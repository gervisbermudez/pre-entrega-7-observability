"""Analista: solo calcula. La calculadora no usa eval()."""

from __future__ import annotations

import ast
import operator as op

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool

from ..state import AgentState
from ..texto import contexto_acotado, extraer_texto

_OPERADORES = {
    ast.Add: op.add,
    ast.Sub: op.sub,
    ast.Mult: op.mul,
    ast.Div: op.truediv,
    ast.Pow: op.pow,
    ast.USub: op.neg,
}

_agente = None


def _evaluar_nodo(nodo: ast.AST):
    if isinstance(nodo, ast.Constant) and isinstance(nodo.value, (int, float)):
        return nodo.value
    if isinstance(nodo, ast.BinOp) and type(nodo.op) in _OPERADORES:
        return _OPERADORES[type(nodo.op)](
            _evaluar_nodo(nodo.left),
            _evaluar_nodo(nodo.right),
        )
    if isinstance(nodo, ast.UnaryOp) and type(nodo.op) in _OPERADORES:
        return _OPERADORES[type(nodo.op)](_evaluar_nodo(nodo.operand))
    raise ValueError("Expresión no permitida")


@tool
def calculadora(expresion: str) -> str:
    """Evalúa una expresión aritmética (+, -, *, /, **, paréntesis).
    Ejemplo: '5*14 + 5*21 + 2*28'. No ejecuta código."""
    try:
        arbol = ast.parse(expresion, mode="eval")
        resultado = _evaluar_nodo(arbol.body)
        return f"Resultado: {resultado}"
    except Exception as exc:
        return f"Error al calcular '{expresion}': {exc}"


def _agente_analista():
    global _agente
    if _agente is None:
        from ..llm import armar_llm

        _agente = create_agent(
            model=armar_llm(),
            tools=[calculadora],
            system_prompt=(
                "Sos un agente de análisis matemático. Tu única función es calcular "
                "con la herramienta calculadora, usando los números que ya figuran "
                "en los aportes. NO busques información nueva y NO inventes cifras "
                "que no estén escritas. Si falta un número de la política, decilo "
                "y no calcules. Tenés que llamar a calculadora: la respuesta tiene "
                "que incluir la expresión y la línea Resultado: que devuelve la herramienta."
            ),
        )
    return _agente


def _lineas_calculadora(mensajes: list) -> list[str]:
    lineas: list[str] = []
    for mensaje in mensajes:
        if not isinstance(mensaje, ToolMessage):
            continue
        if mensaje.name not in {None, "calculadora"}:
            continue
        texto = extraer_texto(mensaje)
        if texto.startswith("Resultado:") or texto.startswith("Error al calcular"):
            lineas.append(texto)
    return lineas


def nodo_analista(state: AgentState) -> dict:
    """El analista recibe los aportes del investigador, no todo el estado."""
    tarea = contexto_acotado(
        state,
        "Calculá con la herramienta calculadora a partir de los números ya "
        "encontrados. No busques políticas ni completes datos que no estén "
        "en los aportes. Incluí la línea Resultado: de la herramienta.",
    )
    resultado = _agente_analista().invoke({"messages": [HumanMessage(content=tarea)]})
    mensajes = resultado["messages"]
    respuesta = extraer_texto(mensajes[-1])
    calculos = [linea for linea in _lineas_calculadora(mensajes) if linea not in respuesta]
    if calculos:
        respuesta = respuesta.rstrip() + "\n" + "\n".join(calculos)
    elif "Resultado:" not in respuesta:
        respuesta = respuesta.rstrip() + "\nNo llamó a la calculadora."
    return {
        "messages": [AIMessage(content=respuesta, name="analista")],
        "contribuciones": [{"agente": "analista", "aporte": respuesta}],
        "pasos": state.get("pasos", 0) + 1,
    }
