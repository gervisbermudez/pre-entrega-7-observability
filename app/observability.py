"""Trazas de cada llamada al modelo hacia Arize Phoenix o LangSmith.

Phoenix corre local (docker compose) y no pide clave. LangSmith se activa
si hay ``LANGSMITH_API_KEY`` y ``OBSERVABILIDAD`` vale ``langsmith`` o ``both``.
El instrumentor de OpenInference envuelve las invocaciones de LangChain,
así que supervisor, especialistas y síntesis quedan en el mismo trazo.
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

_lista = False


def init_observability() -> None:
    global _lista
    if _lista:
        return

    modo = os.getenv("OBSERVABILIDAD", "phoenix").strip().lower()
    if modo in {"langsmith", "both"}:
        os.environ["LANGSMITH_TRACING"] = "true"
        os.environ.setdefault("LANGSMITH_PROJECT", os.getenv("PHOENIX_PROJECT", "techcorp-api"))
        if not os.getenv("LANGSMITH_API_KEY"):
            logger.warning("OBSERVABILIDAD incluye LangSmith pero falta LANGSMITH_API_KEY.")
        else:
            logger.info("LangSmith activo en el proyecto %s.", os.environ["LANGSMITH_PROJECT"])

    if modo in {"phoenix", "both", ""}:
        _iniciar_phoenix()

    _lista = True


def _iniciar_phoenix() -> None:
    endpoint = os.getenv("PHOENIX_COLLECTOR_ENDPOINT", "http://localhost:6006").rstrip("/")
    if not endpoint.endswith("/v1/traces"):
        endpoint = f"{endpoint}/v1/traces"
    os.environ["PHOENIX_COLLECTOR_ENDPOINT"] = endpoint
    proyecto = os.getenv("PHOENIX_PROJECT", "techcorp-api")
    try:
        from openinference.instrumentation.langchain import LangChainInstrumentor
        from phoenix.otel import register

        # Phoenix 20 espera OTLP/HTTP en /v1/traces. Con la URL base el SDK
        # hace POST / y el dashboard responde 405.
        register(
            project_name=proyecto,
            endpoint=endpoint,
            protocol="http/protobuf",
            batch=False,
            verbose=False,
        )
        LangChainInstrumentor().instrument()
    except ImportError:
        logger.warning(
            "Phoenix no está instalado. Seguí con `pip install -r requirements.txt` "
            "si querés trazas en el dashboard."
        )
        return
    except Exception:
        logger.exception("No pude iniciar Phoenix. La API sigue, sin trazas.")
        return
    logger.info("Phoenix recibiendo trazas de %s en %s.", proyecto, endpoint)
