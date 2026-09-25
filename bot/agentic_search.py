"""Pilot: an agentic retry for empty keyword searches.

Everywhere else in this bot, Gemini is called once for a fixed job (score
a batch, write a letter) -- the code decides the whole sequence of steps.
Here it's different: when a keyword search comes back with 0 results,
Gemini is given a real tool (search_keyword) and gets to decide itself
whether to try an alternative Danish term, which term, and whether the
result is good enough to stop -- instead of us hard-coding a synonym list.

Kept deliberately small and capped so it cannot loop: at most
MAX_EXTRA_ATTEMPTS real searches, each term tried at most once, and any
missing/malformed response just stops the loop rather than retrying.
"""

import logging

from google.genai import types

from bot.gemini_client import MODEL, get_client, is_configured
from bot.search import search_keyword as raw_search_keyword
from bot.sources import Vacancy

logger = logging.getLogger(__name__)

MAX_EXTRA_ATTEMPTS = 2

_SEARCH_TOOL = types.FunctionDeclaration(
    name="search_keyword",
    description="Search for Danish job postings with a specific keyword and get the number of results found.",
    parameters={
        "type": "object",
        "properties": {
            "term": {
                "type": "string",
                "description": "An alternative, broader, or synonymous Danish search term.",
            }
        },
        "required": ["term"],
    },
)

_SYSTEM_INSTRUCTION = (
    "You are helping with a job search in Denmark. A search for '{keyword}' returned 0 results. "
    "You may call search_keyword with ONE alternative, broader, or synonymous DANISH term at a "
    "time, at most {max_attempts} times in total. "
    "MANDATORY FORMAT for every response that calls the tool: it MUST contain a text part with "
    "one short sentence IN UKRAINIAN (українською) explaining why you're trying this particular "
    "term, together with the tool call, in that same response. A tool call with no accompanying "
    "text sentence is an invalid response -- always include both parts, never the tool call alone. "
    "If your sentence concludes that '{keyword}' is a typo of or synonym for a real word, "
    "you MUST call the tool with that word in the same response; it is a mistake to write the "
    "conclusion and then not act on it. "
    "If an attempt returns results (count > 0), do NOT call the tool again -- you are done. "
    "IMPORTANT: only call the tool if '{keyword}' can actually be interpreted as a real word or a "
    "typo of a real word (Danish or foreign job title/field). "
    "If '{keyword}' is meaningless text with no reasonable interpretation (random letters, not a "
    "word in any language) -- then say so honestly IN UKRAINIAN and STOP without calling the "
    "tool. Never guess a random word just to have something to suggest."
)


def agentic_keyword_search(keyword: str) -> tuple[list[Vacancy], list[str], str | None]:
    """Returns (extra_vacancies, log, used_term). log is a short list of
    Ukrainian sentences describing what the model tried and why -- useful
    both for debugging and as a transparent trail to show the person.
    used_term is the term that actually found results (or None), so the
    caller can show it as the effective search keyword instead of the
    original."""
    if not is_configured():
        return [], [], None

    client = get_client()
    config = types.GenerateContentConfig(
        tools=[types.Tool(function_declarations=[_SEARCH_TOOL])],
        system_instruction=_SYSTEM_INSTRUCTION.format(
            keyword=keyword, max_attempts=MAX_EXTRA_ATTEMPTS
        ),
    )
    contents = [
        types.Content(
            role="user",
            parts=[types.Part(text=f"Original keyword: '{keyword}'. Suggest an alternative.")],
        )
    ]

    found: list[Vacancy] = []
    log: list[str] = []
    used_term: str | None = None
    tried_terms = {keyword.strip().lower()}

    for _ in range(MAX_EXTRA_ATTEMPTS):
        try:
            response = client.models.generate_content(
                model=MODEL, contents=contents, config=config
            )
        except Exception:
            logger.exception("Agentic search step failed for keyword %r", keyword)
            break

        candidate = response.candidates[0] if response.candidates else None
        if not candidate or not candidate.content or not candidate.content.parts:
            break
        contents.append(candidate.content)

        function_call = None
        had_text = False
        for part in candidate.content.parts:
            if part.text:
                log.append(part.text.strip())
                had_text = True
            if part.function_call:
                function_call = part.function_call

        if not function_call or function_call.name != "search_keyword":
            break  # model chose not to call the tool -- it's done

        term = str(function_call.args.get("term", "")).strip()
        if not term or term.lower() in tried_terms:
            break  # no usable new term -- stop rather than guess
        tried_terms.add(term.lower())

        if not had_text:
            # Gemini occasionally calls the tool without the required
            # explanation sentence despite the instruction below -- fall
            # back to a minimal one so the person never sees a bare
            # result line with no reason attached.
            log.append(f"Пробую схоже слово «{term}».")

        try:
            results = raw_search_keyword(term)
        except Exception:
            logger.exception("Real search failed for agent-suggested term %r", term)
            results = []

        found.extend(results)
        log.append(f"→ Спробував «{term}»: знайдено {len(results)} вакансій по всій Данії.")

        contents.append(
            types.Content(
                role="user",
                parts=[
                    types.Part.from_function_response(
                        name="search_keyword", response={"count": len(results)}
                    )
                ],
            )
        )

        if results:
            used_term = term
            break  # found something -- the hard cap on top of this makes
            # a runaway loop impossible either way

    return found, log, used_term
