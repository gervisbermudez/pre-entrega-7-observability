"""Criticidad y pausa humana, sin Redis ni LLM."""

from __future__ import annotations

import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from app.graph import enrutar
from app.hitl import es_critica_texto, nodo_aprobacion, nodo_ejecutar, nodo_evaluar_criticidad
from app.state import AgentState
from app.worker import extraer_interrupt

CONSULTA_LECTURA = (
    "Un empleado va a cumplir 12 años en TechCorp. "
    "¿Cuántos días de vacaciones acumuló en total en esos 12 años?"
)
CONSULTA_CRITICA = (
    "Calculá los días de vacaciones y, cuando tengas el total, "
    "registrá el ajuste en su legajo y notificá a RRHH."
)


def test_lectura_no_es_critica() -> None:
    critica, _motivo = es_critica_texto(CONSULTA_LECTURA)
    assert critica is False


def test_efecto_es_critico() -> None:
    critica, motivo = es_critica_texto(CONSULTA_CRITICA)
    assert critica is True
    assert "efecto" in motivo


def test_pago_alto_es_critico() -> None:
    critica, _motivo = es_critica_texto("Tenés que pagar 2500 USD al proveedor.")
    assert critica is True


def test_pago_chico_no_dispara_el_umbral() -> None:
    critica, _motivo = es_critica_texto("¿La política paga 20 USD de viático?")
    assert critica is False


def _grafo(avisos: Path):
    import app.hitl as hitl

    hitl.AVISOS = avisos

    def fin(state: AgentState) -> dict:
        return {}

    grafo: StateGraph = StateGraph(AgentState)
    grafo.add_node("evaluar_criticidad", nodo_evaluar_criticidad)
    grafo.add_node("aprobacion_humana", nodo_aprobacion)
    grafo.add_node("ejecutar", nodo_ejecutar)
    grafo.add_node("fin", fin)
    grafo.add_edge(START, "evaluar_criticidad")
    grafo.add_conditional_edges(
        "evaluar_criticidad",
        lambda state: "aprobacion_humana" if state.get("critica") else "fin",
        {"aprobacion_humana": "aprobacion_humana", "fin": "fin"},
    )
    grafo.add_edge("aprobacion_humana", "ejecutar")
    grafo.add_edge("ejecutar", "fin")
    grafo.add_edge("fin", END)
    return grafo.compile(checkpointer=MemorySaver())


def _estado(pregunta: str) -> dict:
    return {"messages": [HumanMessage(content=pregunta)]}


def test_lectura_termina_sin_pausa(tmp: Path) -> None:
    app = _grafo(tmp / "rrhh.jsonl")
    config = {"configurable": {"thread_id": "lectura"}}
    app.invoke(_estado(CONSULTA_LECTURA), config)
    snap = app.get_state(config)
    assert snap.next == ()
    assert snap.values["critica"] is False
    assert not (tmp / "rrhh.jsonl").exists()


def test_critica_pausa_y_aprueba(tmp: Path) -> None:
    avisos = tmp / "rrhh.jsonl"
    app = _grafo(avisos)
    config = {"configurable": {"thread_id": "critica-si"}}
    app.invoke(_estado(CONSULTA_CRITICA), config)
    snap = app.get_state(config)
    assert snap.next
    payload = extraer_interrupt(snap)
    assert payload is not None
    assert "aprob" in payload["pregunta"].lower()

    app.invoke(Command(resume="si"), config)
    final = app.get_state(config)
    assert final.next == ()
    assert final.values["aprobado"] is True
    assert "Notificación enviada" in final.values["accion"]
    assert avisos.exists()


def test_critica_rechazada_no_avisa(tmp: Path) -> None:
    avisos = tmp / "rechazo.jsonl"
    app = _grafo(avisos)
    config = {"configurable": {"thread_id": "critica-no"}}
    app.invoke(_estado(CONSULTA_CRITICA), config)
    app.invoke(Command(resume="no"), config)
    final = app.get_state(config)
    assert final.values["aprobado"] is False
    assert final.values["accion"].startswith("Rechazado")
    assert not avisos.exists()


def test_enrutar_manda_el_cierre_a_criticidad() -> None:
    assert enrutar({"next_agent": "FINISH", "task_completed": True}) == "evaluar_criticidad"
    assert enrutar({"next_agent": "investigador"}) == "investigador"
    assert enrutar({"next_agent": "analista"}) == "analista"


def test_extraer_interrupt_desde_el_snapshot() -> None:
    interrupcion = SimpleNamespace(value={"pregunta": "¿La aprobás?", "motivo": "efecto"})
    snap = SimpleNamespace(next=("aprobacion_humana",), interrupts=(interrupcion,), tasks=())
    assert extraer_interrupt(snap)["motivo"] == "efecto"


def main() -> None:
    tmp = RAIZ / "avisos" / "test"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    try:
        test_lectura_no_es_critica()
        test_efecto_es_critico()
        test_pago_alto_es_critico()
        test_pago_chico_no_dispara_el_umbral()
        test_enrutar_manda_el_cierre_a_criticidad()
        test_extraer_interrupt_desde_el_snapshot()
        test_lectura_termina_sin_pausa(tmp)
        test_critica_pausa_y_aprueba(tmp)
        test_critica_rechazada_no_avisa(tmp)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("ok")


if __name__ == "__main__":
    main()
