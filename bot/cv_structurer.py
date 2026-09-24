"""One-shot structuring of an uploaded CV: instead of asking the person to
come up with search keywords themselves, suggest real Danish job-search
terms straight from what's actually in their CV.

This is what makes the autonomous daily search (bot/autonomous.py) possible
without the person having typed keywords in first -- the agent derives its
own starting point from the CV, the same way a human recruiter would skim
a CV and think "what job titles would this person search for".
"""

import json
import logging

from bot.gemini_client import MAX_CV_CHARS, generate_with_retry, is_configured, truncate

logger = logging.getLogger(__name__)

__all__ = ["is_configured", "suggest_keywords"]

PROMPT_TEMPLATE = """You help a job seeker in Denmark find good search keywords for Danish job portals (Jobindex.dk, Jobnet.dk), based on their CV.

CV:
---
{cv_text}
---

Suggest 3-6 short Danish search terms (real Danish job titles / field names actually used in Danish job postings, not English or Ukrainian translations) that genuinely match the concrete experience in this CV -- not aspirational or adjacent fields the person hasn't actually worked in.

Rules:
- Every term must correspond to real, literal experience/skills in the CV -- never invent a specialization the CV doesn't support.
- Prefer concrete Danish job titles ("bogholder", "systemudvikler", "dataanalytiker") over vague category words ("IT", "kontor").
- Order them from most to least central to the CV's actual background.

Respond STRICTLY as a JSON array of strings, e.g.: ["bogholder", "regnskab", "SQL-udvikler"]
"""


def suggest_keywords(cv_text: str) -> list[str]:
    if not is_configured() or not cv_text:
        return []
    prompt = PROMPT_TEMPLATE.format(cv_text=truncate(cv_text, MAX_CV_CHARS))
    try:
        response = generate_with_retry(prompt, json_mode=True)
        data = json.loads(response.text)
        return [str(term).strip() for term in data if str(term).strip()][:6]
    except Exception:
        logger.exception("CV keyword suggestion failed")
        return []
