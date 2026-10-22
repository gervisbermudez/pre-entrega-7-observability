"""Grafo del módulo 6, con checkpointer de Redis y pausa humana.

El supervisor y los especialistas no cambian de oficio. Cuando el supervisor
cierra, ``evaluar_criticidad`` decide si hace falta un humano antes de
sintetizar. La acción con efecto queda después del ``interrupt``.
"""

from __future__ import annotations

from typing import Literal

from langchain_core.messages import HumanMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from .agents.analyst_agent import nodo_analista
from .agents.research_agent import nodo_investigador
from .hitl import nodo_aprobacion, nodo_ejecutar, nodo_evaluar_criticidad
from .state import AgentState
from .supervisor import nodo_sintesis, nodo_supervisor


def enrutar(state: AgentState) -> Literal["investigador", "analista", "evaluar_criticidad"]:
    if state.get("next_agent") == "FINISH" or state.get("task_completed"):
        return "evaluar_criticidad"
    destino = state.get("next_agent")
    if destino in {"investigador", "analista"}:
        return destino
    return "evaluar_criticidad"


def enrutar_tras_criticidad(state: AgentState) -> Literal["aprobacion_humana", "sintesis"]:
    if state.get("critica"):
        return "aprobacion_humana"
    return "sintesis"


def construir_grafo(checkpointer) -> CompiledStateGraph:
    grafo: StateGraph = StateGraph(AgentState)
    grafo.add_node("supervisor", nodo_supervisor)
    grafo.add_node("investigador", nodo_investigador)
    grafo.add_node("analista", nodo_analista)
    grafo.add_node("evaluar_criticidad", nodo_evaluar_criticidad)
    grafo.add_node("aprobacion_humana", nodo_aprobacion)
    grafo.add_node("ejecutar", nodo_ejecutar)
    grafo.add_node("sintesis", nodo_sintesis)

    grafo.add_edge(START, "supervisor")
    grafo.add_conditional_edges(
        "supervisor",
        enrutar,
        {
            "investigador": "investigador",
            "analista": "analista",
            "evaluar_criticidad": "evaluar_criticidad",
        },
    )
    grafo.add_edge("investigador", "supervisor")
    grafo.add_edge("analista", "supervisor")
    grafo.add_conditional_edges(
        "evaluar_criticidad",
        enrutar_tras_criticidad,
        {
            "aprobacion_humana": "aprobacion_humana",
            "sintesis": "sintesis",
        },
    )
    grafo.add_edge("aprobacion_humana", "ejecutar")
    grafo.add_edge("ejecutar", "sintesis")
    grafo.add_edge("sintesis", END)
    return grafo.compile(checkpointer=checkpointer)


def estado_inicial(pregunta: str) -> AgentState:
    return {
        "messages": [HumanMessage(content=pregunta)],
        "next_agent": None,
        "contribuciones": [],
        "pasos": 0,
        "task_completed": False,
        "razon": "",
        "critica": False,
        "motivo_critico": "",
        "aprobado": None,
        "accion": "",
    }


def config_hilo(job_id: str) -> dict:
    return {
        "configurable": {"thread_id": job_id},
        "recursion_limit": 25,
        "metadata": {"job_id": job_id},
    }
