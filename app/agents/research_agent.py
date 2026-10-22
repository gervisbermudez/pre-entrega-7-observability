"""Investigador: solo busca en las políticas internas. No calcula."""

from __future__ import annotations

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool

from ..rag import obtener_rag
from ..state import AgentState
from ..texto import contexto_acotado, extraer_texto

_agente = None


@tool
def buscar_en_politicas_internas(pregunta: str) -> str:
    """Busca en las políticas internas de TechCorp (vacaciones, teletrabajo,
    seguridad informática, onboarding). Usala para reglas o números de la empresa."""
    resultados = obtener_rag().obtener_top_k(pregunta)
    if not resultados:
        return "No se encontró información relevante en las políticas."
    return "\n\n".join(
        f"[Fuente: {r['fuente']}] {r['contenido']}" for r in resultados
    )


def _agente_investigador():
    global _agente
    if _agente is None:
        from ..llm import armar_llm

        _agente = create_agent(
            model=armar_llm(),
            tools=[buscar_en_politicas_internas],
            system_prompt=(
                "Sos un agente de investigación. Tu única función es buscar datos "
                "concretos en las políticas internas de TechCorp usando "
                "buscar_en_politicas_internas. NO hagas cálculos ni estimes totales. "
                "Transcribí los tramos con las mismas palabras del documento "
                "(si dice 'menos de 5 años', no escribas 'hasta 5'). "
                "Cerrá con la fuente tal como la devuelve la herramienta "
                "([Fuente: archivo])."
            ),
        )
    return _agente


def nodo_investigador(state: AgentState) -> dict:
    """El investigador ve la tarea y los aportes, no el historial del grafo."""
    tarea = contexto_acotado(
        state,
        "Buscá en las políticas los números y las reglas que hagan falta para "
        "la tarea. No calcules el total. Citá la fuente.",
    )
    resultado = _agente_investigador().invoke(
        {"messages": [HumanMessage(content=tarea)]}
    )
    respuesta = extraer_texto(resultado["messages"][-1])
    return {
        "messages": [AIMessage(content=respuesta, name="investigador")],
        "contribuciones": [{"agente": "investigador", "aporte": respuesta}],
        "pasos": state.get("pasos", 0) + 1,
    }
