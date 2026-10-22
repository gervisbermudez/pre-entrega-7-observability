"""Supervisor: elige el próximo nodo y no hace el trabajo de los especialistas.

La rúbrica corta dos fallos típicos: cerrar antes de tener los dos dominios,
y dejar que un especialista se refine sin límite.
"""

from __future__ import annotations

from typing import Literal

from langchain_core.messages import AIMessage
from pydantic import BaseModel, Field

from .state import AgentState
from .texto import extraer_texto

MAX_PASOS = 6
MAX_POR_AGENTE = 2


class DecisionSupervisor(BaseModel):
    next: Literal["investigador", "analista", "FINISH"] = Field(
        description="Próximo agente a invocar, o FINISH si la tarea ya está completa"
    )
    razon: str = Field(description="Breve justificación de la decisión")


SUPERVISOR_PROMPT = """Sos el Supervisor de un equipo con dos especialistas:
- investigador: busca datos concretos en las políticas internas de TechCorp. No calcula.
- analista: calcula con los números que el investigador ya aportó. No busca políticas.

Rúbrica para dar FINISH (las tres tienen que cumplirse):
1. El investigador aportó el dato de la empresa y citó una fuente ([Fuente: ...] o un .txt).
2. El analista calculó después de esa búsqueda y devolvió un número de la herramienta calculadora.
3. Ese número responde la pregunta original. Una escala anual sin el total no alcanza.

Reglas de ruteo:
1. Si falta un dato de la empresa, mandá a 'investigador'. No mandes al analista antes de esa búsqueda.
2. Si ya se buscó el dato pero falta calcular, mandá a 'analista'.
3. Si la búsqueda y el cálculo ya cumplen la rúbrica, respondé 'FINISH'.
4. Si un especialista devolvió un error o un aporte vacío, puede refinar una sola vez.
5. No repitas el mismo agente si su último aporte ya cumple su parte.

Contribuciones hasta ahora:
{contribuciones}
"""


def _ultimo(contribuciones: list[dict[str, str]], agente: str) -> str:
    textos = [c["aporte"] for c in contribuciones if c["agente"] == agente]
    return textos[-1] if textos else ""


def _investigador_ok(aporte: str) -> bool:
    texto = aporte.lower()
    if "no se encontró" in texto or len(aporte.strip()) < 20:
        return False
    tiene_fuente = "fuente:" in texto or ".txt" in texto
    return tiene_fuente and any(caracter.isdigit() for caracter in aporte)


def _analista_ok(aporte: str) -> bool:
    """Acepta el aporte solo si la calculadora devolvió un Resultado."""
    texto = aporte.lower()
    if "error al calcular" in texto or "no llamó a la calculadora" in texto:
        return False
    return "resultado:" in texto and any(caracter.isdigit() for caracter in aporte)


def _aplicar_rubrica(state: AgentState, elegido: str, razon: str) -> tuple[str, str]:
    """Corrige la decisión del modelo si no cumple la rúbrica de suficiencia."""
    contribuciones = list(state.get("contribuciones") or [])
    conteo = {
        "investigador": sum(1 for c in contribuciones if c["agente"] == "investigador"),
        "analista": sum(1 for c in contribuciones if c["agente"] == "analista"),
    }
    inv_ok = _investigador_ok(_ultimo(contribuciones, "investigador"))
    ana_ok = inv_ok and _analista_ok(_ultimo(contribuciones, "analista"))

    if not inv_ok:
        if conteo["investigador"] < MAX_POR_AGENTE:
            if elegido == "investigador":
                return "investigador", razon
            return (
                "investigador",
                "La rúbrica no deja avanzar: falta un dato de la política con fuente.",
            )
        return "FINISH", "El investigador no tiene más intentos. Se cierra con lo que hay."

    if not ana_ok:
        if conteo["analista"] < MAX_POR_AGENTE:
            if elegido == "analista":
                return "analista", razon
            return "analista", "La búsqueda ya cumple. Falta el cálculo del analista."
        return "FINISH", "No hay más reintentos de cálculo. Se cierra con lo que hay."

    if elegido == "FINISH":
        return "FINISH", razon
    return "FINISH", "Ambos especialistas ya cumplieron la rúbrica."


def nodo_supervisor(state: AgentState) -> dict:
    if state.get("pasos", 0) >= MAX_PASOS:
        return {
            "next_agent": "FINISH",
            "task_completed": True,
            "razon": "Se alcanzó el máximo de pasos.",
        }

    contribuciones = state.get("contribuciones") or []
    contribuciones_texto = (
        "\n".join(f"- {c['agente']}: {c['aporte'][:200]}" for c in contribuciones)
        or "(ninguna todavía)"
    )
    pregunta = extraer_texto(state["messages"][0]) if state.get("messages") else ""
    prompt = SUPERVISOR_PROMPT.replace("{contribuciones}", contribuciones_texto)

    from .llm import armar_llm

    decision = armar_llm().with_structured_output(DecisionSupervisor).invoke(
        [
            {"role": "system", "content": prompt},
            {"role": "user", "content": f"Tarea original: {pregunta}"},
        ]
    )
    siguiente, razon = _aplicar_rubrica(state, decision.next, decision.razon)
    if siguiente != decision.next:
        razon = f"El modelo eligió {decision.next}. {razon}"

    return {
        "next_agent": siguiente,
        "task_completed": siguiente == "FINISH",
        "razon": razon,
    }


def nodo_sintesis(state: AgentState) -> dict:
    """Arma la respuesta final. No llama herramientas ni vuelve a delegar."""
    contribuciones = state.get("contribuciones") or []
    contribuciones_texto = "\n".join(
        f"- [{c['agente']}] {c['aporte']}" for c in contribuciones
    ) or "(sin aportes)"
    pregunta = extraer_texto(state["messages"][0]) if state.get("messages") else ""
    cierre = ""
    if state.get("critica"):
        decision = "aprobada" if state.get("aprobado") else "rechazada"
        cierre = (
            f"\n\nLa tarea se marcó como crítica ({state.get('motivo_critico', '')}). "
            f"Decisión humana: {decision}. "
            f"Acción: {state.get('accion') or 'sin acción registrada'}."
        )
    prompt = (
        f"Pregunta original: {pregunta}\n\n"
        f"Aportes del equipo:\n{contribuciones_texto}"
        f"{cierre}\n\n"
        "Redactá una respuesta final clara y breve en español. "
        "Usá solo cifras que aparezcan en los aportes. "
        "Si falta el dato o el cálculo, decilo en vez de inventarlo. "
        "Si hubo una decisión humana, incluila en la respuesta."
    )
    from .llm import armar_llm

    respuesta = armar_llm().invoke(prompt)
    return {"messages": [AIMessage(content=extraer_texto(respuesta), name="supervisor")]}
