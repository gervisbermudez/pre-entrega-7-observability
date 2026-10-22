"""API que encola una consulta y devuelve el job_id al instante.

POST /tasks no corre el agente. GET /tasks/{id} lee Redis.
POST /tasks/{id}/approve reanuda un hilo que quedó pausado.
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import redis.asyncio as aioredis
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, field_validator

from .graph import construir_grafo
from .observability import init_observability
from .worker import (
    PENDING,
    crear_job,
    encolar_reanudacion,
    leer,
    marcar_cortados,
    publico,
    worker_loop,
)

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger(__name__)

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")


class Consulta(BaseModel):
    query: str

    @field_validator("query")
    @classmethod
    def no_vacia(cls, valor: str) -> str:
        texto = valor.strip()
        if not texto:
            raise ValueError("La consulta está vacía")
        return texto


class Decision(BaseModel):
    aprobado: bool


def _abrir_checkpointer(url: str):
    from langgraph.checkpoint.redis import RedisSaver

    contexto = RedisSaver.from_conn_string(url)
    saver = contexto.__enter__()
    try:
        saver.setup()
    except Exception:
        contexto.__exit__(*sys.exc_info())
        raise
    return contexto, saver


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_observability()
    cliente = aioredis.from_url(REDIS_URL, decode_responses=True)
    app.state.redis = cliente
    try:
        await cliente.ping()
    except Exception as exc:
        await cliente.aclose()
        raise RuntimeError(
            f"Redis no responde en {REDIS_URL}. Levantalo con `docker compose up -d`."
        ) from exc

    try:
        contexto, saver = await asyncio.to_thread(_abrir_checkpointer, REDIS_URL)
    except Exception as exc:
        await cliente.aclose()
        raise RuntimeError(
            "No pude crear los índices del checkpointer en Redis. "
            "Hace falta Redis 8 o Redis Stack (RedisJSON y RediSearch). "
            f"Detalle: {exc}"
        ) from exc

    app.state.graph = construir_grafo(saver)
    app.state.checkpointer_cm = contexto
    cortados = await marcar_cortados(cliente)
    if cortados:
        logger.warning("%s jobs quedaron FAILED porque el worker se había cortado.", cortados)

    stop = asyncio.Event()
    tarea = asyncio.create_task(worker_loop(cliente, app.state.graph, stop))
    try:
        yield
    finally:
        stop.set()
        try:
            await asyncio.wait_for(tarea, timeout=15)
        except asyncio.TimeoutError:
            tarea.cancel()
        await cliente.aclose()
        contexto.__exit__(None, None, None)


app = FastAPI(
    title="API del orquestador TechCorp",
    summary="Encola el multi-agente del módulo 6 y consulta el estado por job_id.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/")
async def raiz() -> dict[str, str]:
    return {"docs": "/docs", "health": "/health"}


@app.get("/health")
async def health() -> dict[str, str]:
    try:
        await app.state.redis.ping()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Redis no responde: {exc}") from exc
    return {"status": "ok"}


@app.post("/tasks")
async def crear_tarea(consulta: Consulta) -> dict[str, str]:
    job_id = str(uuid.uuid4())
    # El estado queda escrito antes del RPUSH. Un poll inmediato no da 404.
    job = await crear_job(app.state.redis, job_id, consulta.query)
    return {"job_id": job["job_id"], "status": job["status"]}


@app.get("/tasks/{job_id}")
async def ver_tarea(job_id: str) -> dict:
    job = await leer(app.state.redis, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="No existe ese job_id")
    return publico(job)


@app.post("/tasks/{job_id}/approve")
async def aprobar(job_id: str, decision: Decision) -> dict[str, str]:
    resultado = await encolar_reanudacion(app.state.redis, job_id, decision.aprobado)
    if resultado == "missing":
        raise HTTPException(status_code=404, detail="No existe ese job_id")
    if resultado != PENDING:
        raise HTTPException(
            status_code=409,
            detail=f"El job está en {resultado} y no espera aprobación.",
        )
    return {"job_id": job_id, "status": PENDING}
