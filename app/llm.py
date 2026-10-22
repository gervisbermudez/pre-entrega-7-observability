"""Modelo de chat. OpenAI por defecto; Gemini si ``LLM_PROVIDER=gemini``."""

from __future__ import annotations

import os

from langchain_core.language_models.chat_models import BaseChatModel


def armar_llm() -> BaseChatModel:
    proveedor = os.getenv("LLM_PROVIDER", "openai").strip().lower()
    if proveedor == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        # El mismo modelo que la pista de clase.
        return ChatGoogleGenerativeAI(model="gemini-3.5-flash", temperature=0)

    from langchain_openai import ChatOpenAI

    return ChatOpenAI(model="gpt-4.1-mini", temperature=0)
