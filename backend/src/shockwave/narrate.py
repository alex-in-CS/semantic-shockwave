"""Ask an LLM to explain *why* each hop of a bridge connects.

Talks to any OpenAI-compatible chat endpoint, which covers both Groq (hosted)
and Ollama (local). Narration is optional: with nothing configured the rest of
the app works and the endpoint reports itself as disabled.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

import httpx

GROQ_BASE_URL = "https://api.groq.com/openai/v1"
GROQ_DEFAULT_MODEL = "openai/gpt-oss-20b"
OLLAMA_DEFAULT_MODEL = "llama3.2"

SYSTEM_PROMPT = (
    "You explain chains of related concepts. For each consecutive pair in the chain, "
    "write one vivid, concrete sentence (max 25 words) saying how the first leads to the "
    "second. Then write a one-sentence summary of the whole journey. Reply with JSON only: "
    '{"steps": ["...", ...], "summary": "..."} with exactly one step per pair, in order.'
)


class NarrationError(RuntimeError):
    pass


@dataclass(frozen=True)
class LLMConfig:
    base_url: str
    model: str
    api_key: str | None = None


@dataclass(frozen=True)
class Narration:
    steps: list[str]
    summary: str


def llm_config_from_env() -> LLMConfig | None:
    """An explicit `SHOCKWAVE_LLM_BASE_URL` wins (e.g. Ollama at http://localhost:11434/v1);
    otherwise a `GROQ_API_KEY` selects Groq. Neither means narration is off."""
    model = os.getenv("SHOCKWAVE_LLM_MODEL")
    if base_url := os.getenv("SHOCKWAVE_LLM_BASE_URL"):
        return LLMConfig(base_url.rstrip("/"), model or OLLAMA_DEFAULT_MODEL,
                         os.getenv("SHOCKWAVE_LLM_API_KEY"))
    if groq_key := os.getenv("GROQ_API_KEY"):
        return LLMConfig(GROQ_BASE_URL, model or GROQ_DEFAULT_MODEL, groq_key)
    return None


def _user_prompt(path: list[str]) -> str:
    hops = zip(path, path[1:], strict=False)
    pairs = "\n".join(f"{i + 1}. {a} -> {b}" for i, (a, b) in enumerate(hops))
    return f"Chain: {' -> '.join(path)}\n\nPairs ({len(path) - 1}):\n{pairs}"


def parse_narration(content: str, expected_steps: int) -> Narration:
    """Validate the model's JSON. Tolerates a code fence around it, nothing else."""
    text = content.strip().removeprefix("```json").removeprefix("```").removesuffix("```")
    try:
        data = json.loads(text)
        steps = [str(s).strip() for s in data["steps"]]
        summary = str(data.get("summary", "")).strip()
    except (ValueError, KeyError, TypeError) as exc:
        raise NarrationError("the model did not return the expected JSON") from exc
    if len(steps) != expected_steps:
        raise NarrationError(f"expected {expected_steps} steps, the model gave {len(steps)}")
    return Narration(steps=steps, summary=summary)


async def narrate(path: list[str], config: LLMConfig, client: httpx.AsyncClient) -> Narration:
    if len(path) < 2:
        return Narration(steps=[], summary="")
    headers = {"Authorization": f"Bearer {config.api_key}"} if config.api_key else {}
    payload = {
        "model": config.model,
        "temperature": 0.7,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": _user_prompt(path)},
        ],
    }
    try:
        res = await client.post(f"{config.base_url}/chat/completions", json=payload,
                                headers=headers, timeout=120.0)
        res.raise_for_status()
        content = res.json()["choices"][0]["message"]["content"]
    except httpx.HTTPStatusError as exc:
        raise NarrationError(f"LLM returned HTTP {exc.response.status_code}") from exc
    except httpx.TimeoutException as exc:
        raise NarrationError("the LLM took too long to answer") from exc
    except httpx.HTTPError as exc:
        raise NarrationError(f"can't reach the LLM at {config.base_url}") from exc
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        raise NarrationError("unexpected response shape from the LLM") from exc
    return parse_narration(content, expected_steps=len(path) - 1)
