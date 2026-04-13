"""
Shared utilities for agent nodes — prompt loading and LLM client setup.
"""

import json
import logging
import os
import time
from pathlib import Path

import yaml
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

logger = logging.getLogger(__name__)

PROMPTS_DIR = Path(__file__).parent.parent / "prompts"

# OpenAI config
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o")
MINI_MODEL = os.getenv("OPENAI_MINI_MODEL", "gpt-4o-mini")


def get_openai_client() -> OpenAI:
    """Create an OpenAI client."""
    return OpenAI(api_key=OPENAI_API_KEY)


def load_prompt(template_name: str) -> dict:
    """
    Load a prompt YAML file by node name (e.g., "moderator_v1").

    Returns the full YAML dict with keys: name, version, model,
    system_prompt, user_prompt, etc.
    """
    path = PROMPTS_DIR / f"{template_name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Prompt file not found: {path}")

    with open(path) as f:
        return yaml.safe_load(f)


def call_llm(prompt_config: dict, user_vars: dict, client: OpenAI | None = None) -> dict:
    """
    Call OpenAI with a loaded prompt config and template variables.

    Args:
        prompt_config: Dict from load_prompt() with system_prompt, user_prompt, model.
        user_vars: Dict of {placeholder: value} to fill into the user_prompt template.
        client: Optional pre-created OpenAI client.

    Returns:
        Dict with keys: response (parsed JSON), model, tokens_in, tokens_out, duration_ms.
    """
    if client is None:
        client = get_openai_client()

    # Determine model from prompt config
    model_name = prompt_config.get("model", MINI_MODEL)
    if model_name == "gpt-4o":
        model = CHAT_MODEL
    elif model_name == "gpt-4o-mini":
        model = MINI_MODEL
    else:
        model = model_name

    # Fill user prompt template
    user_prompt = prompt_config["user_prompt"].format(**user_vars)

    start = time.time()
    try:
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": prompt_config["system_prompt"]},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0,
        )
        duration_ms = int((time.time() - start) * 1000)

        raw = completion.choices[0].message.content or "{}"
        usage = completion.usage

        return {
            "response": json.loads(raw),
            "model": model,
            "tokens_in": usage.prompt_tokens if usage else 0,
            "tokens_out": usage.completion_tokens if usage else 0,
            "duration_ms": duration_ms,
        }

    except json.JSONDecodeError:
        logger.error("Failed to parse LLM JSON response: %s", raw)
        return {
            "response": {},
            "model": model,
            "tokens_in": 0,
            "tokens_out": 0,
            "duration_ms": int((time.time() - start) * 1000),
        }
    except Exception as e:
        logger.error("LLM call failed: %s", e)
        raise
