"""Real CV-vs-vacancy matching via Gemini, as opposed to the plain
keyword-overlap fallback in bot/matching.py.

Scores the whole batch of vacancies from one search in a single API call
(rather than one call per vacancy) so this stays fast and friendly to the
free-tier rate limits.
"""

import json

from bot.gemini_client import MAX_CV_CHARS, generate_with_retry, is_configured, truncate
from bot.sources import Vacancy

__all__ = ["is_configured", "semantic_match_batch", "SemanticMatch"]

# Keep prompts a reasonable size: a long vacancy description can otherwise
# blow up token usage across a whole batch for little benefit.
MAX_DESCRIPTION_CHARS = 1500


class SemanticMatch:
    def __init__(self, percent: int, reasoning: str):
        self.percent = percent
        self.reasoning = reasoning


PROMPT_TEMPLATE = """You help a job seeker in Denmark evaluate how well vacancies actually match their CV — not by superficial word overlap, but by substance: the real day-to-day tasks, seniority level, required experience, and the vacancy's language level against the candidate's stated language level.

CANDIDATE'S CV:
---
{cv_text}
---

VACANCIES (score each independently):
{vacancies_block}

For each vacancy, give:
- percent: an integer 0-100, how realistically this person fits this specific position given this CV (judge by the substance of the work, not just keyword overlap; if the role requires a different specialization despite similar-sounding words, give a low percent)
- reasoning: one to two short sentences IN UKRAINIAN (мова — українська) explaining the score (what matches or what is missing). Refer to the person neutrally as "кандидат" (never "кандидатка", and never guess or mention gender based on the name or other CV clues) — the applicant's gender has no bearing on the vacancy match. When the match is partial or zero, if there is a genuine, concrete transferable skill or trait actually present in the CV (e.g. a specific problem-solving habit, a proven fast learning curve, a directly relevant tool used in a different context) — add ONE short clause naming it, clearly separated from the main reasoning (e.g. "...; переноситься вміння швидко освоювати нові системи"). This clause is informational only and must NEVER be used to justify a higher percent than the substance warrants — general technical literacy is not a transferable skill (see the rule above), so leave this out entirely rather than invent one.

IMPORTANT: never credit the candidate with skills or experience that are not literally present in the CV. General exposure to a technology/tool ≠ experience in a specific specialization/role/industry.

GENERAL PRINCIPLE that most often causes inflated scores: the model spots a surface-level technical connection ("both work with data", "both use a computer", "both involve SQL/databases") and counts it as genuine partial overlap, even though the vacancy's REAL tasks require a completely different skill set that is absent from the CV. A surface-level technical connection is NOT the same as relevant experience.

Example 1 (database administration): the CV says "worked with SQL databases, data processing and reporting", WITHOUT the word "administrator" and without tasks like backup/recovery, patching, RMAN, Data Guard, RAC. The vacancy "Oracle Database Administrator" requires exactly these tasks. 80-85% with reasoning "strong experience in administration" — WRONG. Correct: 30-45%, and the reasoning must say plainly these are different roles, even though both "work with Oracle/SQL".

Example 2 (bookkeeping): the CV shows general experience with relational databases/SQL, WITHOUT any mention of bookkeeping, journal entries, VAT, annual reporting. The vacancy "Bogholder" (bookkeeper) requires exactly bookkeeping knowledge (entries, tax, reporting) — the ability to write SQL queries has nothing to do with that knowledge. Giving even 5-15% here with reasoning like "database experience gives an understanding of data structures" is the SAME KIND OF ERROR as example 1: working with data in a database ≠ knowing accounting. Correct: exactly 0%.

Example 3 (job function mismatch — management vs execution): the CV shows a purely technical/execution background (developer, database engineer, QA) with NO product-management, stakeholder-management, or roadmap-ownership experience. The vacancy is a "Product Manager" or similar strategy/ownership role that explicitly requires experience "building or managing" a product, prioritizing a roadmap, and owning business trade-offs — even if the vacancy also uses AI/tech vocabulary the CV shares (AI agents, chatbots, data). Shared subject-matter vocabulary between a management role and an execution role is NOT the same as being qualified for the management role. Do not give 60-80% just because the topic (AI) overlaps; the JOB FUNCTION itself (own the product vs. build the product) is a separate axis from the topic, and a mismatch there caps the score sharply. Correct: 15-35%, with the reasoning naming the function mismatch explicitly (e.g., "кандидат має технічний досвід, але бракує досвіду продуктового менеджменту").

ZERO, NOT A SMALL NUMBER: if the CV contains NOT A SINGLE genuinely relevant skill/knowledge for the vacancy's CORE tasks (only general technical literacy unrelated to the role's substance, or nothing in common at all) — give exactly 0%, not a symbolic 5-10%. A small non-zero percent "just in case, maybe it'll work out" misleads the person just as much as an inflated one. The vacancy stays in the results list regardless (it's a keyword search result); the person decides for themselves whether to apply despite a different profile — your job is to give an honest number, not to decide for them by leaving a "consolation" percent.

Before giving a percent above 20% because of an "adjacent" technical skill — ask yourself: is this skill from the CV actually needed for the vacancy's CORE tasks (not incidental ones, but the ones this position exists for), or is it just general computer/technical literacy that many people have regardless of specialization? If the latter — give 0%, not a low-but-nonzero percent.

CONSISTENCY BETWEEN percent AND reasoning — MANDATORY: if the reasoning says key domain experience/knowledge is MISSING from the CV (e.g., "не вистачає бухгалтерського досвіду", "немає досвіду адміністрування", "потребує іншої спеціалізації") — percent must be 0, even if you also mentioned some adjacent technical detail as a "plus". A medium percent (30-50%) is only acceptable when there is genuine overlap in the role's CORE tasks, and a low non-zero percent (1-20%) only when there is at least a small but literally-present relevant element in the CV (not general technical literacy).

Respond STRICTLY as a JSON array of objects, one per vacancy, in the same order:
[{{"percent": <int>, "reasoning": "<string>"}}, ...]
"""


def semantic_match_batch(
    cv_text: str, vacancies: list[Vacancy]
) -> list[SemanticMatch]:
    if not vacancies:
        return []

    vacancies_block = "\n\n".join(
        f"[{i}] {v.title}\nCompany: {v.company}\nLocation: {v.location}\n"
        f"Description: {truncate(v.description, MAX_DESCRIPTION_CHARS)}"
        for i, v in enumerate(vacancies)
    )

    prompt = PROMPT_TEMPLATE.format(
        cv_text=truncate(cv_text, MAX_CV_CHARS),
        vacancies_block=vacancies_block,
    )

    response = generate_with_retry(prompt, json_mode=True)

    data = json.loads(response.text)
    results = []
    for i in range(len(vacancies)):
        if i < len(data):
            item = data[i]
            results.append(
                SemanticMatch(
                    percent=int(item.get("percent", 0)),
                    reasoning=str(item.get("reasoning", "")),
                )
            )
        else:
            results.append(SemanticMatch(percent=0, reasoning=""))
    return results
