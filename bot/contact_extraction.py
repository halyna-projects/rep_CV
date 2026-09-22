"""Pull out the bits a Danish jobcenter/kommune log usually asks for
(contact person, phone, email) from a vacancy's free-text description.

This information is almost never in a structured field -- it's buried in
a sentence like "Har du spørgsmål, kontakt Thorvald Kodal på ... eller
+45 42 52 79 99" -- so a regex is fragile for the name in particular.
Gemini extraction is more robust; falls back to "ikke angivet" per field
when nothing is found (never invents a name/number).
"""

import json
import logging

from bot.gemini_client import generate_with_retry, truncate
from bot.sources import Vacancy

logger = logging.getLogger(__name__)

MAX_DESCRIPTION_CHARS = 6000

NOT_FOUND = "ikke angivet"

PROMPT_TEMPLATE = """Find the contact person for questions about this vacancy in the text below: name, phone, email.

VACANCY TEXT:
---
{description}
---

Rules:
- Contact info in Danish job postings is very often placed in a closing section near the END of the text, under a heading like "Rekrutteringsprocessen", "Hør mere om jobbet", "Kontakt", or similar -- read the ENTIRE text carefully, including the end, before concluding that something was not found.
- If something is genuinely not mentioned in the text -- return exactly the string "{not_found}" for that field, never invent it.
- Copy phone and email literally as they appear in the text, without changing anything.
- If several contacts are listed, pick the first/primary one.
- Copy the name/phone/email themselves as-is (they're usually already in Danish/Latin script) -- do not translate them.

Respond STRICTLY as JSON:
{{"contact_name": "<name or {not_found}>", "contact_phone": "<phone or {not_found}>", "contact_email": "<email or {not_found}>"}}
"""


class ContactInfo:
    def __init__(self, name: str, phone: str, email: str):
        self.name = name
        self.phone = phone
        self.email = email


def extract_contact_info(vacancy: Vacancy) -> ContactInfo:
    prompt = PROMPT_TEMPLATE.format(
        description=truncate(vacancy.description, MAX_DESCRIPTION_CHARS),
        not_found=NOT_FOUND,
    )
    try:
        response = generate_with_retry(prompt, json_mode=True)
        data = json.loads(response.text)
        return ContactInfo(
            name=str(data.get("contact_name") or NOT_FOUND),
            phone=str(data.get("contact_phone") or NOT_FOUND),
            email=str(data.get("contact_email") or NOT_FOUND),
        )
    except Exception:
        logger.exception("Contact extraction failed for %s", vacancy.url)
        return ContactInfo(name=NOT_FOUND, phone=NOT_FOUND, email=NOT_FOUND)
