import asyncio
import html
import logging
import re
import tempfile
from pathlib import Path

from telegram import (
    Document,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    ReplyKeyboardMarkup,
    Update,
)
from telegram.error import NetworkError, TimedOut
from telegram.ext import ContextTypes

from bot import cv_parser, storage
from bot.danish_cities import resolve_city
from bot.config import ADMIN_TELEGRAM_ID, UPLOADS_DIR
from bot.contact_extraction import extract_contact_info
from bot.letter_explainer import explain_letter_image, explain_letter_text
from bot.letter_generation import generate_cover_letter
from bot.matching import compute_match
from bot.pdf_export import _strip_html, letter_to_pdf, vacancy_to_pdf
from bot.search import search_all
from bot.semantic_matching import is_configured as semantic_matching_configured
from bot.semantic_matching import semantic_match_batch
from bot.translation import translate_to_ukrainian

logger = logging.getLogger(__name__)



async def send_with_retry(update: Update, text: str, retries: int = 2, **kwargs):
    """update.message.reply_text, but survives a flaky connection to Telegram.

    A single dropped connection (seen in practice as httpx.ConnectTimeout /
    telegram.error.TimedOut) used to kill the whole /search silently, with
    the user never finding out anything failed. Retry a couple of times
    with a short backoff before giving up.
    """
    last_error = None
    for attempt in range(retries + 1):
        try:
            return await update.effective_message.reply_text(text, **kwargs)
        except (TimedOut, NetworkError) as exc:
            last_error = exc
            logger.warning("send_with_retry: attempt %s failed: %s", attempt + 1, exc)
            if attempt < retries:
                await asyncio.sleep(2 * (attempt + 1))
    raise last_error

BTN_SEARCH = "🔍 ШУКАТИ ВАКАНСІЇ"
BTN_KEYWORDS = "🔑 Ключові слова"
BTN_LOCATION = "📍 Місто"
BTN_CV = "📄 Моє CV"
BTN_CANCEL = "❌ Скасувати"
BTN_ALL_DENMARK = "🌍 Уся Данія"
BTN_RESET_SEEN = "🔄 Показати вакансії знову"
BTN_EXPLAIN_LETTER = "📨 Пояснити лист/документ\n(будь-яка мова)"

# Required before the Search button appears at all.
REQUIRED_FOR_SEARCH = (BTN_KEYWORDS, BTN_CV)


def _is_ready_for_search(telegram_id: int) -> bool:
    user = storage.get_user(telegram_id) or {}
    has_keywords = bool(storage.get_keywords(telegram_id))
    has_cv = bool(user.get("cv_path"))
    return has_keywords and has_cv


def build_keyboard(telegram_id: int) -> ReplyKeyboardMarkup:
    rows = [[BTN_KEYWORDS, BTN_LOCATION], [BTN_CV], [BTN_RESET_SEEN], [BTN_EXPLAIN_LETTER]]
    if _is_ready_for_search(telegram_id):
        rows.insert(0, [BTN_SEARCH])
    return ReplyKeyboardMarkup(rows, resize_keyboard=True)


# Shown instead of the main menu while the bot is waiting for a specific
# free-text reply (keywords / city), so there's always an obvious way out
# if the person changes their mind instead of typing anything.
CANCEL_KEYBOARD = ReplyKeyboardMarkup([[BTN_CANCEL]], resize_keyboard=True)

# Shown specifically while waiting for a city: Cancel alone only means
# "leave the current filter as it was" (confusing when the person actually
# wants to switch TO nationwide search), so offer that as its own button.
LOCATION_KEYBOARD = ReplyKeyboardMarkup(
    [[BTN_ALL_DENMARK], [BTN_CANCEL]], resize_keyboard=True
)


def _missing_requirements(telegram_id: int) -> list[str]:
    missing = []
    if not storage.get_keywords(telegram_id):
        missing.append(BTN_KEYWORDS)
    user = storage.get_user(telegram_id) or {}
    if not user.get("cv_path"):
        missing.append(BTN_CV)
    return missing


DISCLAIMER = (
    "⚠️ Перед тим як почати, важливо знати:\n\n"
    "• Якщо після старту щось не відповідає одразу — почекайте трохи "
    "або натисніть /start ще раз (буває, бот саме перезапускається).\n\n"
    "• Бот шукає вакансії та готує чернетки листів (ansøgning) — "
    "але <b>сам нічого нікуди не подає</b>. Подати заявку на сайті "
    "компанії за посиланням треба самостійно.\n\n"
    "• Файли, які надішле бот (ansøgning, вакансія та їх переклад), "
    "потрібно самостійно зберегти, прочитати переклад і, за потреби, "
    "відредагувати перед тим, як надсилати.\n\n"
    "• Заповнити лог на Jobnet (для звітності перед комуною) — теж "
    "самостійно, бот цього не робить.\n\n"
    "• ⚠️ <b>Не надсилайте документи з CPR-номером чи іншими надчутливими "
    "даними без потреби.</b> Все, що ви надсилаєте (CV, листи), обробляється "
    "зовнішнім AI-сервісом (Google Gemini) і тимчасово зберігається на сервері "
    "бота. Якщо можливо — закресліть CPR перед відправкою фото/PDF."
)

WELCOME = (
    "Привіт! Я шукаю вакансії на Jobindex.dk, Jobnet.dk та LinkedIn "
    "за вашими ключовими словами.\n\n"
    "Користуйтеся кнопками внизу екрана:\n"
    f"{BTN_KEYWORDS} — задати свої ключові слова через кому\n"
    f"{BTN_LOCATION} — необов'язково, відфільтрувати за містом\n"
    f"{BTN_CV} — завантажити/перевірити своє CV (PDF або Word — просто надішліть файл)\n\n"
    f"Кнопка {BTN_SEARCH} з'явиться, коли заповните ключові слова та CV."
)


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    telegram_id = update.effective_user.id
    storage.ensure_user(telegram_id)
    context.user_data.pop("awaiting", None)
    await update.message.reply_text(DISCLAIMER, parse_mode="HTML")
    await update.message.reply_text(WELCOME, reply_markup=build_keyboard(telegram_id))


async def help_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        WELCOME, reply_markup=build_keyboard(update.effective_user.id)
    )


async def stats(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_user.id != ADMIN_TELEGRAM_ID:
        return
    s = storage.get_stats()
    await update.message.reply_text(
        "📊 Статистика бота\n\n"
        f"Всього унікальних людей, що запускали бота: {s['total']}\n"
        f"Завантажили CV: {s['with_cv']}\n"
        f"Задали ключові слова: {s['with_keywords']}\n"
        f"Вказали місто: {s['with_location']}\n"
        f"Хоч раз запускали пошук: {s['searched']}\n"
        f"Пояснювали лист/документ: {s['used_letter_explain']}"
    )


async def _reply_with_next_step(update: Update, telegram_id: int, done_message: str):
    missing = _missing_requirements(telegram_id)
    if missing:
        done_message += "\n\nЗалишилось заповнити: " + ", ".join(missing)
    await update.message.reply_text(
        done_message, reply_markup=build_keyboard(telegram_id)
    )


async def _save_keywords(update: Update, telegram_id: int, text: str):
    keywords = [k.strip() for k in text.split(",") if k.strip()]
    storage.set_keywords(telegram_id, keywords)
    await _reply_with_next_step(
        update, telegram_id, "Зберіг ключові слова: " + ", ".join(keywords)
    )


async def _prompt_keywords(update: Update, context: ContextTypes.DEFAULT_TYPE):
    telegram_id = update.effective_user.id
    current = storage.get_keywords(telegram_id)
    if current:
        message = (
            "🔵 Поточні ключові слова: "
            + ", ".join(current)
            + "\n\n<b>Щоб змінити — надішліть нові через кому</b>, або натисніть Скасувати."
        )
    else:
        message = (
            "🔵 <b>Надішліть ключові слова через кому</b>, наприклад:\n"
            "sælger (продавець), markedsføring (маркетинг), bogholder (бухгалтер)\n\n"
            "Пошук йде по датських сайтах — краще писати ключові слова "
            "датською, інакше пошук може нічого не знайти.\n\n"
            "Або натисніть Скасувати, якщо передумали."
        )
    await update.message.reply_text(message, reply_markup=CANCEL_KEYBOARD, parse_mode="HTML")
    context.user_data["awaiting"] = "keywords"


async def set_keywords(update: Update, context: ContextTypes.DEFAULT_TYPE):
    telegram_id = update.effective_user.id
    text = " ".join(context.args) if context.args else ""
    if not text:
        await _prompt_keywords(update, context)
        return
    await _save_keywords(update, telegram_id, text)


async def _save_location(update: Update, telegram_id: int, text: str):
    storage.set_location(telegram_id, text)
    message = (
        f"Шукатиму за містом: {text}"
        if text
        else "Фільтр за містом знято — шукаю по всій Данії."
    )
    await _reply_with_next_step(update, telegram_id, message)


async def _resolve_and_save_location(
    update: Update, context: ContextTypes.DEFAULT_TYPE, telegram_id: int, raw_text: str
):
    """Checks the typed city against the official list of 99 Danish
    municipalities (bot/danish_cities.py) before saving, so a typo or an
    old spelling (Århus) doesn't silently turn into a filter that matches
    nothing. Confident matches get auto-corrected; weak matches get a
    clarifying re-prompt instead of being saved outright; a name that
    doesn't match anything at all is still saved (it might be a real
    postal town like "Viby J" that just isn't a kommune), but with a
    heads-up that it's not in the official list.
    """
    if not raw_text:
        await _save_location(update, telegram_id, "")
        return

    canonical, suggestions = resolve_city(raw_text)

    if canonical:
        if canonical.lower() != raw_text.strip().lower():
            await update.message.reply_text(f"Зрозумів як: {canonical}")
        await _save_location(update, telegram_id, canonical)
        return

    if suggestions:
        await update.message.reply_text(
            f"Не знайшов міста «{raw_text}» серед офіційних муніципалітетів Данії.\n\n"
            f"Можливо, ви мали на увазі: {', '.join(suggestions)}?\n\n"
            "Надішліть правильну назву, або натисніть Скасувати, щоб залишити без змін.",
            reply_markup=CANCEL_KEYBOARD,
        )
        context.user_data["awaiting"] = "location"
        return

    await update.message.reply_text(
        f"«{raw_text}» немає в списку офіційних муніципалітетів Данії — "
        "можливо, це район/місто всередині якогось муніципалітету "
        "(таке теж буває), збережу як є, але пошук може нічого не знайти."
    )
    await _save_location(update, telegram_id, raw_text)


async def _prompt_location(update: Update, context: ContextTypes.DEFAULT_TYPE):
    telegram_id = update.effective_user.id
    user = storage.get_user(telegram_id)
    current = (user or {}).get("location") or ""
    if current:
        message = (
            f"Зараз фільтр за містом: {current}\n\n"
            f"<b>Надішліть нову назву міста</b>, натисніть «{BTN_ALL_DENMARK}» "
            "(зняти фільтр зовсім), або «Скасувати», щоб залишити як є."
        )
    else:
        message = (
            "<b>Надішліть назву міста</b>, наприклад: Aarhus\n\n"
            f"Я і так вже шукаю по всій Данії — «{BTN_ALL_DENMARK}» і «Скасувати» "
            "тут роблять те саме."
        )
    await update.message.reply_text(
        message, reply_markup=LOCATION_KEYBOARD, parse_mode="HTML"
    )
    context.user_data["awaiting"] = "location"


async def set_location(update: Update, context: ContextTypes.DEFAULT_TYPE):
    telegram_id = update.effective_user.id
    text = " ".join(context.args) if context.args else ""
    if not text:
        await _prompt_location(update, context)
        return
    await _resolve_and_save_location(update, context, telegram_id, text)


async def _prompt_explain_letter(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "📨 Надішліть документ, який хочете зрозуміти — <b>фото або PDF-файл</b>, "
        "будь-якою мовою (не обов'язково данською, не обов'язково від kommune чи SKAT). "
        "Поясню простими словами українською: від кого лист, що треба зробити і до якого терміну.\n\n"
        "⚠️ Якщо в листі є CPR-номер, адреса чи прізвище — по можливості "
        "закресліть/заретушуйте перед відправкою, або сфотографуйте/скриньте "
        "лише той фрагмент листа, який треба пояснити, без цих даних. "
        "Для пояснення суті листа вони не потрібні.\n\n"
        "Це пояснення від ШІ для орієнтування, не офіційна консультація.",
        reply_markup=CANCEL_KEYBOARD,
        parse_mode="HTML",
    )
    context.user_data["awaiting"] = "letter"


async def handle_plain_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    telegram_id = update.effective_user.id
    text = (update.message.text or "").strip()
    if not text:
        return

    if text == BTN_CANCEL:
        context.user_data.pop("awaiting", None)
        await update.message.reply_text(
            "Добре, скасував.", reply_markup=build_keyboard(telegram_id)
        )
        return
    if text == BTN_ALL_DENMARK:
        context.user_data.pop("awaiting", None)
        await _save_location(update, telegram_id, "")
        return
    if text == BTN_SEARCH:
        await run_search(update, context)
        return
    if text == BTN_KEYWORDS:
        await _prompt_keywords(update, context)
        return
    if text == BTN_LOCATION:
        await _prompt_location(update, context)
        return
    if text == BTN_CV:
        await cv_status(update, context)
        return
    if text == BTN_RESET_SEEN:
        await reset_seen(update, context)
        return
    if text == BTN_EXPLAIN_LETTER:
        await _prompt_explain_letter(update, context)
        return

    awaiting = context.user_data.pop("awaiting", None)
    if awaiting == "location":
        if text.lower() in ("нет", "ні", "no", "-"):
            text = ""
        await _resolve_and_save_location(update, context, telegram_id, text)
        return
    if awaiting == "letter":
        await update.message.reply_text(
            "Це має бути фото або PDF-файл листа, не текст. Надішліть, будь ласка, "
            "документом або фотографією.",
            reply_markup=CANCEL_KEYBOARD,
        )
        context.user_data["awaiting"] = "letter"
        return

    # A bare number (no /apply, no other pending state) almost always means
    # "pick vacancy N from the list I just showed you" -- NEVER let this
    # silently fall through to overwriting real keywords with "1". If
    # LAST_RESULTS is empty (e.g. the bot restarted and lost its in-memory
    # cache since the last /search), say so explicitly instead of guessing.
    if text.isdigit() and awaiting is None:
        if storage.get_last_results(telegram_id):
            await _apply_to_vacancy_core(update, context, int(text))
        else:
            await update.message.reply_text(
                "Не бачу список вакансій (можливо, бот перезапускався) — "
                f"натисніть {BTN_SEARCH} ще раз, потім можна буде "
                "надсилати номер вакансії.",
                reply_markup=build_keyboard(telegram_id),
            )
        return

    # Default: any other free-text message (including the first message
    # after tapping "Ключові слова") is treated as a keywords update.
    await _save_keywords(update, telegram_id, text)


async def handle_document(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if context.user_data.get("awaiting") == "letter":
        await _handle_letter_document(update, context)
        return
    await handle_cv_upload(update, context)


async def _handle_letter_document(update: Update, context: ContextTypes.DEFAULT_TYPE):
    context.user_data.pop("awaiting", None)
    telegram_id = update.effective_user.id
    document: Document = update.message.document
    filename = (document.file_name or "letter").lower()

    if not filename.endswith(".pdf"):
        await update.message.reply_text(
            f"Приймаю лист лише як фото або PDF. Спробуйте ще раз через «{BTN_EXPLAIN_LETTER}».",
            reply_markup=build_keyboard(telegram_id),
        )
        return

    await send_with_retry(update, "Читаю лист...")

    with tempfile.TemporaryDirectory() as tmp_dir:
        dest = Path(tmp_dir) / filename
        tg_file = await document.get_file()
        await tg_file.download_to_drive(custom_path=str(dest))
        try:
            text = await asyncio.to_thread(cv_parser.extract_text, str(dest))
        except Exception:
            text = ""

    if len((text or "").strip()) < 30:
        await update.message.reply_text(
            "Не вдалося прочитати текст з цього PDF (можливо, це скан-зображення). "
            f"Спробуйте надіслати фото листа замість PDF через «{BTN_EXPLAIN_LETTER}».",
            reply_markup=build_keyboard(telegram_id),
        )
        return

    try:
        explanation = await asyncio.to_thread(explain_letter_text, text)
    except Exception:
        logger.exception("Letter explanation (PDF) failed for %s", telegram_id)
        await update.message.reply_text(
            "Не вдалося пояснити лист (збій моделі). Спробуйте ще раз.",
            reply_markup=build_keyboard(telegram_id),
        )
        return

    storage.increment_letters_explained(telegram_id)
    await update.message.reply_text(_strip_html(explanation), reply_markup=build_keyboard(telegram_id))


async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if context.user_data.get("awaiting") != "letter":
        # Stray photo outside the letter-explain flow -- CVs are never
        # uploaded as photos in this bot, so there's nothing useful to do.
        return
    context.user_data.pop("awaiting", None)
    telegram_id = update.effective_user.id

    await send_with_retry(update, "Читаю лист...")

    photo = update.message.photo[-1]  # largest resolution
    tg_file = await photo.get_file()
    photo_bytes = bytes(await tg_file.download_as_bytearray())

    try:
        explanation = await asyncio.to_thread(explain_letter_image, photo_bytes, "image/jpeg")
    except Exception:
        logger.exception("Letter explanation (photo) failed for %s", telegram_id)
        await update.message.reply_text(
            "Не вдалося пояснити лист (збій моделі). Спробуйте ще раз.",
            reply_markup=build_keyboard(telegram_id),
        )
        return

    storage.increment_letters_explained(telegram_id)
    await update.message.reply_text(_strip_html(explanation), reply_markup=build_keyboard(telegram_id))


async def handle_cv_upload(update: Update, context: ContextTypes.DEFAULT_TYPE):
    telegram_id = update.effective_user.id
    document: Document = update.message.document

    allowed_ext = (".pdf", ".doc", ".docx")
    filename = document.file_name or "cv"
    if not filename.lower().endswith(allowed_ext):
        await update.message.reply_text(
            "Надішліть CV у форматі PDF або Word (.pdf, .doc, .docx)."
        )
        return

    user_dir = UPLOADS_DIR / str(telegram_id)
    user_dir.mkdir(exist_ok=True)
    dest_path = user_dir / filename

    tg_file = await document.get_file()
    await tg_file.download_to_drive(custom_path=str(dest_path))

    storage.set_cv_path(telegram_id, str(dest_path))

    try:
        cv_text = cv_parser.extract_text(str(dest_path))
        storage.set_cv_text(telegram_id, cv_text)
    except Exception as exc:
        logger.warning("Could not extract CV text for %s: %s", telegram_id, exc)
        storage.set_cv_text(telegram_id, None)
        await _reply_with_next_step(
            update,
            telegram_id,
            f"CV зберіг: {filename}\n\n"
            "Не вдалося прочитати текст з файлу (для смислового "
            "порівняння з вакансіями) — спробуйте перезберегти його як "
            "звичайний PDF або .docx.",
        )
        return

    await _reply_with_next_step(update, telegram_id, f"CV зберіг: {filename}")


async def cv_status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    telegram_id = update.effective_user.id
    user = storage.get_user(telegram_id)
    if user and user.get("cv_path"):
        message = f"Завантажене CV: {user['cv_path'].split('/')[-1]}"
    else:
        message = "CV ще не завантажено — надішліть файл документом (PDF або Word)."
    await _reply_with_next_step(update, telegram_id, message)


async def reset_seen(update: Update, context: ContextTypes.DEFAULT_TYPE):
    telegram_id = update.effective_user.id
    count = storage.clear_seen(telegram_id)
    if count:
        await update.message.reply_text(f"Забув {count} вже показаних вакансій, шукаю знову...")
    # The button says "show again" -- so show them again immediately,
    # rather than just clearing a flag and leaving the person to guess
    # that they now need to separately press Search.
    await run_search(update, context)


def _matched_keywords(v, keywords: list[str]) -> list[str]:
    """Which of the person's own search keywords literally appear in this
    vacancy's title/description -- shown as a dedicated line so it's clear
    at a glance why it turned up, separate from the % match verdict
    (which may be semantic/Gemini-based and not tied to exact words)."""
    haystack = f"{v.title} {v.description}".lower()
    return [k.strip() for k in keywords if k.strip() and k.strip().lower() in haystack]


def format_vacancy(index: int, v, percent: int, detail: str, keywords: list[str]) -> str:
    title = html.escape(v.title)
    parts = [f"{index}. {title}"]
    meta = " | ".join(html.escape(p) for p in [v.company, v.location, v.source] if p)
    if meta:
        parts.append(meta)

    matched = _matched_keywords(v, keywords)
    if matched:
        parts.append("<b>Ключові слова:</b> " + html.escape(", ".join(matched)))

    detail_escaped = html.escape(detail) if detail else ""
    parts.append(f"Збіг: {percent}%" + (f" — {detail_escaped}" if detail_escaped else ""))
    if v.url:
        url_escaped = html.escape(v.url)
        parts.append(f'<a href="{url_escaped}">{url_escaped}</a>')
    return "\n".join(parts)


# How many vacancies get sent to Gemini for real scoring per search. Bounds
# cost/latency/free-tier rate limits; the final list shown to the user is
# capped further (MAX_RESULTS below) once these are sorted.
SEMANTIC_BATCH_CAP = 30


def _score_vacancies(telegram_id: int, vacancies: list, keywords: list[str]):
    """Returns [(vacancy, percent, detail_string), ...].

    Prefers real semantic scoring via Gemini (bot/semantic_matching.py) when
    it's configured and the user has a parsed CV; falls back to plain
    keyword-overlap matching (bot/matching.py) otherwise.
    """
    cv_text = storage.get_cv_text(telegram_id) if semantic_matching_configured() else None

    if cv_text:
        candidates = vacancies[:SEMANTIC_BATCH_CAP]
        try:
            results = semantic_match_batch(cv_text, candidates)
            return [
                (v, r.percent, r.reasoning) for v, r in zip(candidates, results)
            ]
        except Exception:
            logger.exception(
                "Semantic matching failed for %s, falling back to keyword match",
                telegram_id,
            )

    scored = [(v, compute_match(v, keywords)) for v in vacancies]
    return [
        (v, m.percent, ("за словами: " + ", ".join(m.matched_keywords) if m.matched_keywords else ""))
        for v, m in scored
    ]


# The "type a number to apply" hint is easy to miss as plain text buried in
# a longer message, and Telegram buttons can't be bold -- but message text
# can, via HTML parse mode. Kept as one constant so the wording/formatting
# stays identical everywhere it's shown (before the list, after it, and on
# "Показати список знову").
NUMBER_HINT_HTML = "<b>Щоб отримати ansøgning під вакансію — надішліть її номер</b> (наприклад: 1)."


async def run_search(update: Update, context: ContextTypes.DEFAULT_TYPE):
    telegram_id = update.effective_user.id
    missing = _missing_requirements(telegram_id)
    if missing:
        await update.message.reply_text(
            "Спочатку заповніть: " + ", ".join(missing),
            reply_markup=build_keyboard(telegram_id),
        )
        return

    keywords = storage.get_keywords(telegram_id)
    user = storage.get_user(telegram_id)
    location = (user or {}).get("location") or None

    await send_with_retry(
        update,
        f"Шукаю за словами: {', '.join(keywords)}"
        + (f" у {location}" if location else " по всій Данії")
        + " ...",
    )

    vacancies = await asyncio.to_thread(search_all, keywords, location)

    urls = [v.url for v in vacancies if v.url]
    unseen_urls = storage.filter_unseen(telegram_id, urls)
    new_vacancies = [v for v in vacancies if v.url in unseen_urls or not v.url]

    if not new_vacancies:
        await send_with_retry(
            update,
            "Нових вакансій не знайшлося (або всі вже надсилав раніше). "
            f"Якщо хочете побачити їх знову — натисніть «{BTN_RESET_SEEN}».",
            reply_markup=build_keyboard(telegram_id),
        )
        return

    scored = await asyncio.to_thread(_score_vacancies, telegram_id, new_vacancies, keywords)
    scored.sort(key=lambda triple: triple[1], reverse=True)

    MAX_RESULTS = 20
    to_send = scored[:MAX_RESULTS]
    storage.set_last_results(telegram_id, to_send)

    # Said again at the end of the full list too, but that's easy to miss
    # if the person doesn't scroll past a long list -- show it right above
    # vacancy #1 as well, where they're actually looking.
    await send_with_retry(
        update,
        f"Знайшов {len(to_send)}. {NUMBER_HINT_HTML}",
        parse_mode="HTML",
    )

    sent_urls = await _send_results_chunks(update, to_send, keywords)
    storage.mark_seen(telegram_id, sent_urls)

    footer = (
        f"...і ще {len(new_vacancies) - MAX_RESULTS}. "
        f"Уточніть ключові слова або місто, щоб звузити список."
        if len(new_vacancies) > MAX_RESULTS
        else "Це всі нові вакансії на зараз."
    )
    footer += "\n\n" + NUMBER_HINT_HTML
    await send_with_retry(
        update, footer, reply_markup=build_keyboard(telegram_id), parse_mode="HTML"
    )


async def _send_results_chunks(
    update: Update, to_send: list, keywords: list[str]
) -> list[str]:
    """Sends the numbered vacancy list in chunks of 5. Returns the URLs
    that were actually sent, so the caller can mark them seen even if a
    later chunk fails partway through."""
    sent_urls = []
    for i in range(0, len(to_send), 5):
        chunk = to_send[i : i + 5]
        text = "\n\n".join(
            format_vacancy(i + j + 1, v, percent, detail, keywords)
            for j, (v, percent, detail) in enumerate(chunk)
        )
        await send_with_retry(update, text, disable_web_page_preview=True, parse_mode="HTML")
        sent_urls.extend(v.url for v, percent, detail in chunk if v.url)
    return sent_urls


def _apply_keyboard(telegram_id: int, index: int):
    results = storage.get_last_results(telegram_id)
    row = []
    if index < len(results):
        row.append(
            InlineKeyboardButton(
                f"➡️ Наступна (№{index + 1})", callback_data=f"apply:{index + 1}"
            )
        )
    buttons = [row] if row else []
    buttons.append(
        [InlineKeyboardButton("📋 Показати список знову", callback_data="relist")]
    )
    return InlineKeyboardMarkup(buttons)


async def _apply_to_vacancy_core(update: Update, context: ContextTypes.DEFAULT_TYPE, index: int):
    telegram_id = update.effective_user.id
    message = update.effective_message

    results = storage.get_last_results(telegram_id)
    if not results:
        await message.reply_text(f"Спочатку запустіть пошук — натисніть {BTN_SEARCH}.")
        return
    if not (1 <= index <= len(results)):
        await message.reply_text(f"Немає вакансії №{index} — в останньому списку їх {len(results)}.")
        return

    cv_text = storage.get_cv_text(telegram_id)
    if not cv_text:
        await message.reply_text(f"Не знайшов текст вашого CV — надішліть файл ще раз через {BTN_CV}.")
        return

    if not semantic_matching_configured():
        await message.reply_text("Генерація листів зараз недоступна (не налаштовано доступ до моделі).")
        return

    vacancy, percent, detail = results[index - 1]
    await send_with_retry(update, f"Пишу ansøgning для «{vacancy.title}»...")

    try:
        letter = await asyncio.to_thread(generate_cover_letter, cv_text, vacancy)
    except Exception:
        logger.exception("Letter generation failed for %s / %s", telegram_id, vacancy.url)
        await message.reply_text("Не вдалося згенерувати лист (збій на боці моделі). Спробуйте ще раз.")
        return

    await send_with_retry(
        update,
        f"{vacancy.title} — {vacancy.company}\n{vacancy.url}\n\n{letter}",
    )

    contact = await asyncio.to_thread(extract_contact_info, vacancy)

    with tempfile.TemporaryDirectory() as tmp_dir:
        safe_name = re.sub(r"[^\w\-]+", "_", vacancy.title)[:60] or "vacancy"
        try:
            letter_pdf_path = Path(tmp_dir) / "letter.pdf"
            letter_to_pdf(letter, vacancy, str(letter_pdf_path))
            with open(letter_pdf_path, "rb") as f:
                await message.reply_document(
                    document=f,
                    filename=f"ansogning_{safe_name}.pdf",
                    caption="Ansøgning у PDF — можна зберегти і роздрукувати.",
                )

            vacancy_pdf_path = Path(tmp_dir) / "vacancy.pdf"
            vacancy_to_pdf(vacancy, str(vacancy_pdf_path), contact=contact)
            with open(vacancy_pdf_path, "rb") as f:
                await message.reply_document(
                    document=f,
                    filename=f"vakansiya_{safe_name}.pdf",
                    caption="Вакансія у PDF — зверху зведення для логу (контакт, телефон, email, посилання).",
                )
        except Exception:
            logger.exception("PDF export failed for %s / %s", telegram_id, vacancy.url)
            await message.reply_text("Лист готовий, але не вдалося зробити PDF-файли — спробуйте /apply ще раз.")
            return

        # Translations are a bonus on top of the two Danish documents
        # above, not a core deliverable -- if Gemini hiccups here, don't
        # fail the whole /apply when the letter+PDFs already went out.
        try:
            letter_ua = await asyncio.to_thread(translate_to_ukrainian, letter)
            letter_ua_path = Path(tmp_dir) / "letter_ua.txt"
            letter_ua_path.write_text(letter_ua, encoding="utf-8")
            with open(letter_ua_path, "rb") as f:
                await message.reply_document(
                    document=f,
                    filename=f"ansogning_UA_{safe_name}.txt",
                    caption="Переклад ansøgning українською — для перевірки, можна редагувати.",
                )

            vacancy_text = f"{vacancy.title}\n\n{_strip_html(vacancy.description)}"
            vacancy_ua = await asyncio.to_thread(translate_to_ukrainian, vacancy_text)
            # Appended after translation, not before -- a link run through
            # the translation model risks coming back mangled. This was
            # missing entirely before: the UA file had no link at all, so
            # returning to a saved vacancy later meant digging through the
            # Danish PDF instead, where the link isn't even a real
            # clickable/copyable hyperlink, just plain text.
            if vacancy.url:
                vacancy_ua += f"\n\nПосилання: {vacancy.url}"
            vacancy_ua_path = Path(tmp_dir) / "vacancy_ua.txt"
            vacancy_ua_path.write_text(vacancy_ua, encoding="utf-8")
            with open(vacancy_ua_path, "rb") as f:
                await message.reply_document(
                    document=f,
                    filename=f"vakansiya_UA_{safe_name}.txt",
                    caption="Переклад вакансії українською — для перевірки, як є.",
                )
        except Exception:
            logger.exception("Translation failed for %s / %s", telegram_id, vacancy.url)
            await message.reply_text(
                "Не вдалося зробити переклад українською (два файли на датській вище вже готові)."
            )

    await message.reply_text(
        "Що далі?",
        reply_markup=_apply_keyboard(telegram_id, index),
    )


async def apply_to_vacancy(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args or not context.args[0].isdigit():
        await update.message.reply_text(
            "Вкажіть номер вакансії з останнього списку, наприклад: /apply 3"
        )
        return
    await _apply_to_vacancy_core(update, context, int(context.args[0]))


async def handle_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    telegram_id = update.effective_user.id
    data = query.data or ""

    if data.startswith("apply:"):
        index = int(data.split(":", 1)[1])
        await _apply_to_vacancy_core(update, context, index)
        return

    if data == "relist":
        results = storage.get_last_results(telegram_id)
        if not results:
            await update.effective_message.reply_text(
                f"Список порожній — натисніть {BTN_SEARCH}.",
                reply_markup=build_keyboard(telegram_id),
            )
            return
        await update.effective_message.reply_text(
            f"Список з {len(results)}. {NUMBER_HINT_HTML}",
            parse_mode="HTML",
        )
        await _send_results_chunks(update, results, storage.get_keywords(telegram_id))
        await update.effective_message.reply_text(
            "Це весь список вище.",
            reply_markup=build_keyboard(telegram_id),
        )
