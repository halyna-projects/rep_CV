"""Shared Gemini client/helpers used by both semantic_matching.py
(scoring) and letter_generation.py (writing the ansøgning)."""

import logging
import time

from google import genai
from google.genai import errors as genai_errors
from google.genai import types

from bot.config import GEMINI_API_KEY

logger = logging.getLogger(__name__)

MODEL = "gemini-3.5-flash-lite"

MAX_CV_CHARS = 6000

_client = None


def is_configured() -> bool:
    return bool(GEMINI_API_KEY)


def get_client():
    global _client
    if _client is None:
        # Without an explicit timeout, a stuck Gemini call hangs forever --
        # no exception, nothing in the logs, the bot just never replies.
        # Every other network call in this bot (job sources, Telegram
        # itself) already has one; this was the one gap.
        _client = genai.Client(
            api_key=GEMINI_API_KEY,
            http_options=types.HttpOptions(timeout=45_000),
        )
    return _client


def truncate(text: str, limit: int) -> str:
    text = text or ""
    return text if len(text) <= limit else text[:limit] + "..."


def generate_with_retry(
    prompt: str = None, *, contents=None, json_mode: bool = False, retries: int = 3
):
    """Gemini's free tier returns transient 503s ("high demand") fairly
    often -- retry with backoff before giving up, rather than surfacing
    that as a hard failure for what's usually a one-shot glitch.

    Pass either a plain text `prompt`, or `contents` directly (e.g. a
    [text, image_part] list for multimodal calls -- see
    bot/letter_explainer.py)."""
    client = get_client()
    config = types.GenerateContentConfig(
        response_mime_type="application/json" if json_mode else None
    )
    final_contents = contents if contents is not None else prompt
    last_error = None
    for attempt in range(retries + 1):
        try:
            return client.models.generate_content(
                model=MODEL, contents=final_contents, config=config
            )
        except genai_errors.ServerError as exc:
            last_error = exc
            logger.warning("Gemini call attempt %s failed: %s", attempt + 1, exc)
            if attempt < retries:
                time.sleep(3 * (attempt + 1))
    raise last_error
