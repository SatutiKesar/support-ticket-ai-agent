"""
Thin, provider-agnostic LLM client.

Supports three free-to-use backends (pick one via LLM_PROVIDER in .env):
  - "groq"        : Groq's free-tier API, OpenAI-compatible /chat/completions.
  - "ollama"      : Fully local model served by Ollama (http://localhost:11434).
  - "huggingface" : HuggingFace Inference API free tier.

A 4th provider, "offline", is included purely so the rest of the system can
be smoke-tested with zero setup and zero API keys. It does NOT satisfy the
assessment's "must use an LLM" requirement on its own -- it's a development
convenience, not the intended runtime mode. See README for setup of a real
provider (Groq is the fastest to get a free key for).
"""
import json
import requests

from app import config


class LLMError(Exception):
    pass


def chat(messages: list[dict], temperature: float = 0.0, timeout: int = 30) -> str:
    """
    Send a chat-style message list ([{"role": "system"/"user", "content": ...}])
    to the configured provider and return the assistant's text reply.
    """
    provider = config.LLM_PROVIDER
    if provider == "groq":
        return _chat_groq(messages, temperature, timeout)
    if provider == "ollama":
        return _chat_ollama(messages, temperature, timeout)
    if provider == "huggingface":
        return _chat_huggingface(messages, temperature, timeout)
    if provider == "offline":
        return _chat_offline(messages)
    raise LLMError(
        f"Unknown LLM_PROVIDER '{provider}'. Use one of: groq, ollama, huggingface, offline."
    )


def _chat_groq(messages, temperature, timeout) -> str:
    if not config.GROQ_API_KEY:
        raise LLMError(
            "GROQ_API_KEY is not set. Get a free key at https://console.groq.com/keys "
            "and put it in your .env file."
        )
    headers = {
        "Authorization": f"Bearer {config.GROQ_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": config.GROQ_MODEL,
        "messages": messages,
        "temperature": temperature,
    }
    try:
        resp = requests.post(config.GROQ_URL, headers=headers, json=payload, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]
    except requests.exceptions.RequestException as exc:
        raise LLMError(f"Groq request failed: {exc}") from exc
    except (KeyError, IndexError) as exc:
        raise LLMError(f"Unexpected Groq response shape: {exc}") from exc


def _chat_ollama(messages, temperature, timeout) -> str:
    url = f"{config.OLLAMA_HOST}/api/chat"
    payload = {
        "model": config.OLLAMA_MODEL,
        "messages": messages,
        "stream": False,
        "options": {"temperature": temperature},
    }
    try:
        resp = requests.post(url, json=payload, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        return data["message"]["content"]
    except requests.exceptions.RequestException as exc:
        raise LLMError(
            f"Could not reach Ollama at {config.OLLAMA_HOST}. Is `ollama serve` running "
            f"and have you pulled the model (`ollama pull {config.OLLAMA_MODEL}`)? Details: {exc}"
        ) from exc
    except KeyError as exc:
        raise LLMError(f"Unexpected Ollama response shape: {exc}") from exc


def _chat_huggingface(messages, temperature, timeout) -> str:
    if not config.HF_API_TOKEN:
        raise LLMError(
            "HF_API_TOKEN is not set. Get a free token at https://huggingface.co/settings/tokens "
            "and put it in your .env file."
        )
    headers = {
        "Authorization": f"Bearer {config.HF_API_TOKEN}",
        "Content-Type": "application/json",
    }
    payload = {"model": config.HF_MODEL, "messages": messages, "temperature": temperature}
    try:
        resp = requests.post(config.HF_URL, headers=headers, json=payload, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]
    except requests.exceptions.RequestException as exc:
        raise LLMError(
            f"HuggingFace Inference request failed (model may be cold-starting or "
            f"unavailable on the free tier): {exc}"
        ) from exc
    except (KeyError, IndexError) as exc:
        raise LLMError(f"Unexpected HuggingFace response shape: {exc}") from exc


def _chat_offline(messages) -> str:
    """
    Zero-dependency stub used for local smoke-testing without any API key.
    Recognizes a couple of very simple patterns; anything else returns a
    safe fallback SQL query so the pipeline never crashes.
    """
    question = ""
    for m in reversed(messages):
        if m.get("role") == "user":
            question = m["content"].lower()
            break

    if "sql" in json.dumps(messages).lower() or "SELECT" in json.dumps(messages):
        if "open" in question and "how many" in question:
            sql = "SELECT COUNT(*) AS open_tickets FROM tickets WHERE status = 'Open';"
        else:
            sql = "SELECT * FROM tickets LIMIT 10;"
        return json.dumps({"sql": sql, "explanation": "offline stub: best-effort guess"})

    return "This is the offline stub provider; configure a real LLM_PROVIDER for real answers."
