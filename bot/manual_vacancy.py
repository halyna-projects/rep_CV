"""Let a person add a job the bot didn't find itself -- a link, a PDF, or
a photo/screenshot of a posting -- and turn it into the same Vacancy shape
the normal search pipeline produces, so match-scoring and application
writing work identically either way."""

import json
import logging
import re

import requests
from google.genai import types

from bot.config import HTTP_USER_AGENT
from bot.gemini_client import generate_with_retry
from bot.sources import Vacancy

logger = logging.getLogger(__name__)

MAX_TEXT_CHARS = 6000

_EXTRACT_INSTRUCTION = (
    "Du får teksten fra et forsøg på et jobopslag -- fra et link, en PDF eller et foto. "
    "Udtræk oplysningerne, og svar KUN med JSON på formen "
    '{"found": bool, "title": str, "company": str, "location": str, "description": str}.\n'
    "- found: false hvis dette tydeligvis IKKE er et jobopslag (fx en fejlside, en "
    "login-side, eller helt urelateret indhold) -- så skal de øvrige felter være tomme "
    "strenge.\n"
    "- title: stillingens titel, som den fremgår.\n"
    '- company: virksomhedens navn, hvis det fremgår ("" hvis ikke).\n'
    '- location: by/sted, hvis det fremgår ("" hvis ikke).\n'
    "- description: jobopslagets indhold (opgaver, krav, kvalifikationer), kort "
    "sammenfattet til de vigtigste punkter, maks ca. 800 tegn."
)

_TAG_RE = re.compile(r"<[^>]+>")
_SCRIPT_STYLE_RE = re.compile(r"<(script|style)\b[^>]*>.*?</\1>", re.I | re.S)


def fetch_url_text(url: str, timeout: int = 20) -> str:
    resp = requests.get(url, headers={"User-Agent": HTTP_USER_AGENT}, timeout=timeout)
    resp.raise_for_status()
    html_text = _SCRIPT_STYLE_RE.sub(" ", resp.text)
    text = _TAG_RE.sub(" ", html_text)
    return re.sub(r"\s+", " ", text).strip()


def _parse_response(response) -> Vacancy | None:
    try:
        data = json.loads(response.text)
    except (json.JSONDecodeError, TypeError, AttributeError):
        logger.exception("Could not parse manual-vacancy extraction response")
        return None
    if not data.get("found") or not str(data.get("title", "")).strip():
        return None
    return Vacancy(
        source="Manuelt tilføjet",
        title=str(data.get("title", "")).strip(),
        company=str(data.get("company", "")).strip(),
        location=str(data.get("location", "")).strip(),
        url="",
        description=str(data.get("description", "")).strip(),
    )


def extract_vacancy_from_text(text: str) -> Vacancy | None:
    prompt = f"{_EXTRACT_INSTRUCTION}\n\nTEKST:\n---\n{text[:MAX_TEXT_CHARS]}\n---"
    response = generate_with_retry(prompt, json_mode=True)
    return _parse_response(response)


def extract_vacancy_from_image(image_bytes: bytes, mime_type: str) -> Vacancy | None:
    response = generate_with_retry(
        contents=[
            _EXTRACT_INSTRUCTION,
            types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
        ],
        json_mode=True,
    )
    return _parse_response(response)
