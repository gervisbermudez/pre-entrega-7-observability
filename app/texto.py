"""Texto plano para mensajes y el contexto que ve cada especialista."""

from __future__ import annotations

from typing import Any

from langchain_core.messages import BaseMessage

from .state import AgentState


def extraer_texto(contenido: Any) -> str:
    """Gemini a veces devuelve una lista de bloques en vez de un string."""
    if contenido is None:
        return ""
    if isinstance(contenido, BaseMessage):
        return extraer_texto(contenido.content)
    if isinstance(contenido, str):
        return contenido
    if isinstance(contenido, list):
        partes: list[str] = []
        for bloque in contenido:
            if isinstance(bloque, str):
                partes.append(bloque)
            elif isinstance(bloque, dict):
                partes.append(str(bloque.get("text") or ""))
        return "".join(partes).strip()
    return str(contenido).strip()


def contexto_acotado(state: AgentState, instruccion: str) -> str:
    """Pasa la tarea y los aportes ya hechos. No pasa metadata del grafo.

    El especialista no ve ``pasos``, ``next_agent`` ni el historial crudo.
    """
    mensajes = state.get("messages") or []
    pregunta = extraer_texto(mensajes[0]) if mensajes else ""
    bloques = [f"Tarea original:\n{pregunta}"]

    aportes = state.get("contribuciones") or []
    if aportes:
        lineas = [f"- [{c['agente']}] {c['aporte']}" for c in aportes]
        bloques.append("Aportes ya registrados:\n" + "\n".join(lineas))

    bloques.append(f"Tu instrucción:\n{instruccion}")
    return "\n\n".join(bloques)
