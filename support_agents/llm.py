"""Small LLM interface so every agent can be tested with a scripted fake."""

from __future__ import annotations

import json
import re
from typing import Any, Protocol


class LLM(Protocol):
    def chat(self, system: str, user: str) -> str: ...


class OpenAICompatibleLLM:
    """Works with OpenAI, Groq, OpenRouter, Together, or a local Ollama server."""

    def __init__(self, api_key: str | None, model: str, base_url: str | None = None, temperature: float = 0.2):
        from openai import OpenAI

        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.temperature = temperature

    def chat(self, system: str, user: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        )
        return response.choices[0].message.content or ""


def chat_json(llm: LLM, system: str, user: str) -> dict[str, Any]:
    """Ask for JSON and parse it, tolerating ```json fences and surrounding prose."""
    raw = llm.chat(system + "\n\nRespond with a single JSON object only.", user)
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        raise ValueError(f"Model did not return JSON: {raw[:200]}")
    return json.loads(match.group(0))
