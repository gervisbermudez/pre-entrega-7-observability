"""Estado compartido del orquestador.

``contribuciones`` usa ``operator.add`` como reducer: cada especialista
agrega su aporte y no pisa el de los demás. ``messages`` lo hereda de
``MessagesState`` con ``add_messages``.
"""

from __future__ import annotations

import operator
from typing import Annotated, NotRequired

from langgraph.graph import MessagesState


class AgentState(MessagesState):
    """Hereda ``messages`` y suma los campos que el supervisor y el HITL necesitan."""

    next_agent: NotRequired[str | None]
    contribuciones: Annotated[list[dict[str, str]], operator.add]
    pasos: NotRequired[int]
    task_completed: NotRequired[bool]
    razon: NotRequired[str]
    critica: NotRequired[bool]
    motivo_critico: NotRequired[str]
    aprobado: NotRequired[bool | None]
    accion: NotRequired[str]
