"""Pausa humana antes de una acción con efecto.

Una consulta de solo lectura (buscar la política y calcular) termina sola.
Si el pedido pide registrar, notificar, ejecutar un pago o mover un monto
alto, el grafo se detiene en ``interrupt`` y espera ``POST /tasks/{id}/approve``.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path

from langgraph.types import interrupt

from .state import AgentState
from .texto import extraer_texto

UMBRAL_MONTO = 1000
AVISOS = Path(__file__).resolve().parent.parent / "avisos" / "rrhh.jsonl"

# Frases que piden un efecto fuera del grafo. Una pregunta de lectura
# ("¿cuántos días acumula?") no las contiene.
_EFECTOS = (
    "notific",
    "avisá a",
    "avisa a",
    "avisar a",
    "registrá",
    "registra el",
    "registrar el",
    "enviá",
    "envia un",
    "enviar un",
    "aplicá el ajuste",
    "aplica el ajuste",
    "aplicar el ajuste",
    "ejecutá",
    "ejecuta el",
    "ejecutar el",
    "mandá un correo",
    "manda un correo",
    "enviar un correo",
    "reembolso",
    "debitá",
    "debita",
    "transferí",
    "transferi",
    "cargá en el legajo",
    "carga en el legajo",
    "cargar en el legajo",
    "costo muy alto",
    "tarea crítica",
    "tarea critica",
)

_MONEDA = ("usd", "u$s", "$", "dolar", "dólar", "dolares", "dólares", "peso")
_PAGOS = ("pagar", "pagá", "paga ", "pago de", "abonar", "aboná")


def es_critica_texto(pregunta: str) -> tuple[bool, str]:
    """Decide si el pedido necesita un humano antes de actuar."""
    texto = (pregunta or "").lower()
    if any(frase in texto for frase in _EFECTOS):
        return True, "La tarea pide una acción con efecto o un costo muy alto."
    if any(frase in texto for frase in _PAGOS) and _hay_monto_alto(texto):
        return (
            True,
            f"La tarea pide un pago de {UMBRAL_MONTO} USD o más.",
        )
    return False, "Consulta de solo lectura: no requiere aprobación."


def es_critica(state: AgentState) -> tuple[bool, str]:
    mensajes = state.get("messages") or []
    pregunta = extraer_texto(mensajes[0]) if mensajes else ""
    return es_critica_texto(pregunta)


def nodo_evaluar_criticidad(state: AgentState) -> dict:
    critica, motivo = es_critica(state)
    return {"critica": critica, "motivo_critico": motivo, "aprobado": None}


def nodo_aprobacion(state: AgentState) -> dict:
    """Pausa el grafo. Al reanudar, ``interrupt`` devuelve la decisión.

    No hay efectos antes de esta llamada: el aviso a RRHH vive en el nodo
    siguiente, porque al reanudar este nodo vuelve a ejecutarse desde cero.
    """
    mensajes = state.get("messages") or []
    pregunta = extraer_texto(mensajes[0]) if mensajes else ""
    decision = interrupt(
        {
            "pregunta": (
                "Esta tarea quiere ejecutar una acción con efecto. "
                "¿La aprobás?"
            ),
            "motivo": state.get("motivo_critico") or "",
            "pedido": pregunta[:500],
        }
    )
    return {"aprobado": decision_aprobada(decision)}


def nodo_ejecutar(state: AgentState) -> dict:
    """Corre el efecto recién cuando el humano ya aprobó."""
    if not state.get("aprobado"):
        return {"accion": "Rechazado por el revisor. No se notificó a RRHH."}
    mensaje = _armar_aviso(state)
    ruta = registrar_aviso(mensaje)
    return {"accion": f"Notificación enviada a RRHH ({ruta.name}): {mensaje}"}


def decision_aprobada(decision: object) -> bool:
    if isinstance(decision, bool):
        return decision
    if isinstance(decision, dict):
        return bool(decision.get("aprobado"))
    texto = str(decision).strip().lower()
    return texto in {"si", "sí", "true", "1", "aprobar", "aprobado", "yes", "ok"}


def registrar_aviso(mensaje: str) -> Path:
    AVISOS.parent.mkdir(parents=True, exist_ok=True)
    linea = json.dumps(
        {
            "ts": datetime.now(timezone.utc).isoformat(),
            "mensaje": mensaje,
        },
        ensure_ascii=False,
    )
    with AVISOS.open("a", encoding="utf-8") as archivo:
        archivo.write(linea + "\n")
    return AVISOS


def _armar_aviso(state: AgentState) -> str:
    aportes = state.get("contribuciones") or []
    resumen = " | ".join(f"{c['agente']}: {c['aporte'][:180]}" for c in aportes)
    return resumen or "Ajuste de vacaciones sin detalle de aportes."


def _hay_monto_alto(texto: str) -> bool:
    if not any(moneda in texto for moneda in _MONEDA):
        return False
    for crudo in re.findall(r"\d+", texto):
        if int(crudo) >= UMBRAL_MONTO:
            return True
    return False
