"""Generate a cover letter (ansøgning) for one specific vacancy via Gemini,
grounded in the person's real CV -- same approach used manually earlier in
this project: highlight genuine overlap, name real gaps honestly instead of
inventing experience that isn't in the CV.
"""

from bot.gemini_client import MAX_CV_CHARS, generate_with_retry, truncate
from bot.sources import Vacancy

MAX_DESCRIPTION_CHARS = 4000

PROMPT_TEMPLATE = """You help a job seeker in Denmark write an ansøgning (cover letter) for a specific vacancy, grounded in their real CV.

CANDIDATE'S CV:
---
{cv_text}
---

VACANCY:
Title: {title}
Company: {company}
Location: {location}
Description: {description}

Write the ansøgning in {language}. Requirements:
- Rely ONLY on real facts from the CV -- never invent skills, tools, or experience that aren't there
- If the vacancy requires something the CV doesn't have, mention it honestly but briefly as a willingness to learn quickly -- don't hide it
- Emphasize what genuinely overlaps: concrete experience, tools, industry
- Standard business letter structure: greeting, 3-4 substantive paragraphs, closing
- Don't use generic content-free phrases ("I'm a team player" etc.) -- only concrete points from the CV
- LANGUAGES: if the CV lists several languages under the same combined proficiency level (e.g. "Flydende/modersmål" -- this is ONE combined category "fluent OR native", with no distinction made), don't write "native speaker" for a language unless that's clearly established (foreign citizenship/education in that language, etc.) -- use the neutral "fluent" instead of claiming a language is native when that isn't precisely confirmed
- Respond with ONLY the letter text, no explanation before or after
"""

SUMMARY_PROMPT_TEMPLATE = """You help a job seeker in Denmark write a short, tailored CV summary paragraph (a "Profil" section) for a specific vacancy, grounded in their real CV -- so that both automated ATS keyword screening and a human reader immediately see the relevant overlap.

CANDIDATE'S CV:
---
{cv_text}
---

VACANCY:
Title: {title}
Company: {company}
Location: {location}
Description: {description}

Write the summary in {language}. Requirements:
- Rely ONLY on real facts from the CV -- never invent skills, tools, or experience that aren't there
- Naturally work in the vacancy's own terminology/keywords WHERE they genuinely match something in the CV -- never keyword-stuff a term that doesn't actually apply
- 3-5 sentences, written as a CV "Profile" paragraph (not a cover letter -- no greeting, no closing, no "Dear...")
- If the vacancy emphasizes something the CV doesn't have, don't force it in -- just lead with the strongest genuine overlaps instead
- Respond with ONLY the summary paragraph, no explanation before or after
"""


def _pick_language(vacancy: Vacancy) -> str:
    # Crude heuristic: Danish job ads use these words constantly; English
    # ones (like Collectia's) explicitly say so. Good enough for a first
    # draft -- the person reviews and edits before sending anyway.
    danish_markers = ("stilling", "erfaring", "du har", "vi søger", "ansøgning")
    text = f"{vacancy.title} {vacancy.description}".lower()
    if any(m in text for m in danish_markers):
        return "Danish (Dansk)"
    return "English"


def generate_cover_letter(cv_text: str, vacancy: Vacancy) -> str:
    prompt = PROMPT_TEMPLATE.format(
        cv_text=truncate(cv_text, MAX_CV_CHARS),
        title=vacancy.title,
        company=vacancy.company,
        location=vacancy.location,
        description=truncate(vacancy.description, MAX_DESCRIPTION_CHARS),
        language=_pick_language(vacancy),
    )

    response = generate_with_retry(prompt, json_mode=False)
    return response.text.strip()


def generate_cv_summary(cv_text: str, vacancy: Vacancy) -> str:
    prompt = SUMMARY_PROMPT_TEMPLATE.format(
        cv_text=truncate(cv_text, MAX_CV_CHARS),
        title=vacancy.title,
        company=vacancy.company,
        location=vacancy.location,
        description=truncate(vacancy.description, MAX_DESCRIPTION_CHARS),
        language=_pick_language(vacancy),
    )

    response = generate_with_retry(prompt, json_mode=False)
    return response.text.strip()
