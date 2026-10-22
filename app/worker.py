"""Cola Redis y worker que corre el grafo fuera del request.

El POST guarda el estado PENDING y recién ahí hace RPUSH. El worker saca
el id, pasa a RUNNING y ejecuta el grafo en un hilo, para no frenar el
event loop. Si el grafo se pausa, el estado queda AWAITING_APPROVAL. Si
revienta, queda FAILED: el cliente no se queda en un poll eterno.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from datetime import datetime, timezone
from typing import Any

from opentelemetry import trace

from .graph import config_hilo, estado_inicial
from .texto import extraer_texto

logger = logging.getLogger(__name__)

QUEUE_KEY = "techcorp:jobs:queue"
DATA_PREFIX = "techcorp:jobs:data:"
RESUME_PREFIX = "techcorp:jobs:resume-once:"

PENDING = "PENDING"
RUNNING = "RUNNING"
AWAITING_APPROVAL = "AWAITING_APPROVAL"
FAILED = "FAILED"
DONE = "DONE"


def ahora() -> str:
    return datetime.now(timezone.utc).isoformat()


def data_key(job_id: str) -> str:
    return f"{DATA_PREFIX}{job_id}"


def resume_key(job_id: str) -> str:
    return f"{RESUME_PREFIX}{job_id}"


async def guardar(redis, job: dict[str, Any]) -> None:
    job["updated_at"] = ahora()
    await redis.set(data_key(job["job_id"]), json.dumps(job, ensure_ascii=False))


async def leer(redis, job_id: str) -> dict[str, Any] | None:
    crudo = await redis.get(data_key(job_id))
    if not crudo:
        return None
    return json.loads(crudo)


async def actualizar(redis, job_id: str, **cambios: Any) -> dict[str, Any] | None:
    job = await leer(redis, job_id)
    if job is None:
        return None
    job.update(cambios)
    await guardar(redis, job)
    return job


def publico(job: dict[str, Any]) -> dict[str, Any]:
    return {
        "job_id": job["job_id"],
        "status": job["status"],
        "query": job.get("query"),
        "result": job.get("result"),
        "error": job.get("error"),
        "interrupt": job.get("interrupt"),
        "accion": job.get("accion"),
        "critica": job.get("critica"),
        "motivo_critico": job.get("motivo_critico"),
        "contribuciones": job.get("contribuciones") or [],
        "thread_id": job["job_id"],
        "created_at": job.get("created_at"),
        "updated_at": job.get("updated_at"),
    }


async def crear_job(redis, job_id: str, query: str) -> dict[str, Any]:
    """Persiste PENDING antes de encolar. Si no, un poll temprano da 404."""
    job = {
        "job_id": job_id,
        "status": PENDING,
        "query": query,
        "result": None,
        "error": None,
        "interrupt": None,
        "accion": None,
        "critica": None,
        "motivo_critico": None,
        "contribuciones": [],
        "action": "start",
        "aprobado": None,
        "created_at": ahora(),
        "updated_at": ahora(),
    }
    await redis.set(data_key(job_id), json.dumps(job, ensure_ascii=False))
    await redis.rpush(QUEUE_KEY, job_id)
    return job


async def encolar_reanudacion(redis, job_id: str, aprobado: bool) -> str:
    """Devuelve el estado resultante: PENDING si encoló, u otro si no correspondía."""
    job = await leer(redis, job_id)
    if job is None:
        return "missing"
    if job["status"] != AWAITING_APPROVAL:
        return job["status"]

    # Una sola reanudación. Dos clics no disparan dos avisos a RRHH.
    reservado = await redis.set(resume_key(job_id), "si" if aprobado else "no", nx=True)
    if not reservado:
        return job["status"]

    job["status"] = PENDING
    job["action"] = "resume"
    job["aprobado"] = aprobado
    job["interrupt"] = None
    job["error"] = None
    try:
        await guardar(redis, job)
        await redis.rpush(QUEUE_KEY, job_id)
    except Exception:
        await redis.delete(resume_key(job_id))
        raise
    return PENDING


async def marcar_cortados(redis) -> int:
    """Un RUNNING al arrancar es un job que murió a mitad de camino.

    No se reencola: volver a invocar el mismo hilo puede repetir el aviso
    a RRHH. AWAITING_APPROVAL se deja como está: el checkpoint sigue en Redis
    y el approve reanuda después de un reinicio.
    """
    cortados = 0
    async for clave in redis.scan_iter(match=f"{DATA_PREFIX}*"):
        crudo = await redis.get(clave)
        if not crudo:
            continue
        job = json.loads(crudo)
        if job.get("status") != RUNNING:
            continue
        job["status"] = FAILED
        job["error"] = (
            "El worker se reinició mientras corría esta tarea. Enviá un pedido nuevo."
        )
        await guardar(redis, job)
        cortados += 1
        logger.warning("Job %s quedó FAILED por un reinicio", job["job_id"])
    return cortados


def extraer_interrupt(snapshot) -> dict[str, Any] | None:
    interrupciones = list(getattr(snapshot, "interrupts", None) or [])
    if not interrupciones:
        for tarea in getattr(snapshot, "tasks", ()) or ():
            interrupciones.extend(getattr(tarea, "interrupts", ()) or ())
    if interrupciones:
        valor = interrupciones[0].value
        if isinstance(valor, dict):
            return valor
        return {"pregunta": str(valor)}
    if getattr(snapshot, "next", ()):
        return {"pregunta": "El grafo está pausado esperando aprobación."}
    return None


def _correr(graph, entrada, config) -> tuple[Any, dict[str, Any] | None]:
    """Corre en un hilo. La pausa humana no es un fallo: el checkpoint queda en Redis."""
    tracer = trace.get_tracer("techcorp.worker")
    resultado = None
    with tracer.start_as_current_span("ejecutar_tarea") as span:
        span.set_attribute("job.id", config["configurable"]["thread_id"])
        span.set_attribute("session.id", config["configurable"]["thread_id"])
        try:
            resultado = graph.invoke(entrada, config)
        except _pausas():
            logger.info("Grafo en pausa para %s", config["configurable"]["thread_id"])
        return graph.get_state(config), _interrupt_en_resultado(resultado)


def _pausas():
    try:
        from langgraph.errors import GraphInterrupt
    except ImportError:
        return (type("GraphInterrupt", (Exception,), {}),)
    return (GraphInterrupt,)


def _interrupt_en_resultado(resultado: Any) -> dict[str, Any] | None:
    if not isinstance(resultado, dict) or "__interrupt__" not in resultado:
        return None
    bruto = resultado["__interrupt__"]
    item = bruto[0] if isinstance(bruto, (list, tuple)) and bruto else bruto
    valor = getattr(item, "value", item)
    if isinstance(valor, dict):
        return valor
    return {"pregunta": str(valor)}


async def procesar(redis, graph, job_id: str) -> None:
    job = await leer(redis, job_id)
    if job is None:
        logger.error("Job %s no está en Redis. No se procesa.", job_id)
        return

    try:
        config = config_hilo(job_id)
        if job.get("action") == "resume":
            from langgraph.types import Command

            decision = "si" if job.get("aprobado") else "no"
            entrada: Any = Command(resume=decision)
        else:
            entrada = estado_inicial(job["query"])

        snapshot, interrupt_directo = await asyncio.to_thread(_correr, graph, entrada, config)
    except Exception as exc:
        logger.exception("Job %s falló", job_id)
        await actualizar(
            redis,
            job_id,
            status=FAILED,
            error=str(exc)[:2000],
            result=None,
        )
        return

    payload = extraer_interrupt(snapshot) or interrupt_directo
    if getattr(snapshot, "next", ()) or payload:
        motivo = (payload or {}).get("motivo")
        await actualizar(
            redis,
            job_id,
            status=AWAITING_APPROVAL,
            interrupt=payload,
            critica=True,
            motivo_critico=motivo,
            error=None,
        )
        logger.info("Job %s esperando aprobación: %s", job_id, motivo)
        return

    values = snapshot.values or {}
    mensajes = values.get("messages") or []
    respuesta = extraer_texto(mensajes[-1]) if mensajes else ""
    await actualizar(
        redis,
        job_id,
        status=DONE,
        result=respuesta,
        error=None,
        interrupt=None,
        accion=values.get("accion") or None,
        critica=bool(values.get("critica")),
        motivo_critico=values.get("motivo_critico"),
        contribuciones=list(values.get("contribuciones") or []),
    )
    logger.info("Job %s terminó", job_id)


async def worker_loop(redis, graph, stop: asyncio.Event) -> None:
    concurrencia = max(1, int(os.getenv("WORKER_CONCURRENCY", "4")))
    semaforo = asyncio.Semaphore(concurrencia)
    tareas: set[asyncio.Task] = set()
    logger.info("Worker escuchando %s (concurrencia %s)", QUEUE_KEY, concurrencia)

    async def _uno(job_id: str) -> None:
        async with semaforo:
            await procesar(redis, graph, job_id)

    while not stop.is_set():
        item = await redis.blpop(QUEUE_KEY, timeout=1)
        if not item:
            continue
        _, job_id = item
        if isinstance(job_id, bytes):
            job_id = job_id.decode()
        await actualizar(redis, job_id, status=RUNNING, error=None)
        tarea = asyncio.create_task(_uno(job_id))
        tareas.add(tarea)
        tarea.add_done_callback(tareas.discard)

    if tareas:
        await asyncio.wait(tareas, timeout=10)
