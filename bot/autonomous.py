"""Pilot: the bot runs its own search on a schedule, instead of waiting for
a button press -- scoped to just the admin for now, to learn how this
behaves (frequency, noise, cost) before ever considering it for every user.

Reuses run_search() as-is rather than re-implementing the search pipeline,
by handing it a minimal stand-in for an incoming Update -- run_search only
ever calls .effective_user.id / .message.reply_text / .effective_message.reply_text
on it, never anything Telegram-update-specific.
"""

import logging

from telegram.ext import ContextTypes

from bot.config import ADMIN_TELEGRAM_ID

logger = logging.getLogger(__name__)

# How often the agent checks on its own. Deliberately not too frequent
# while this is admin-only and unproven -- easy to shorten once it's clear
# it behaves well (doesn't spam, doesn't re-announce old results).
CHECK_INTERVAL_SECONDS = 6 * 60 * 60  # every 6 hours

# Below this match %, a result isn't worth an unprompted interruption --
# a manual /search still shows everything, this only trims the
# autonomous run.
MIN_AUTONOMOUS_MATCH_PERCENT = 50


class _ChatMessageProxy:
    """Stands in for update.message / update.effective_message: routes
    reply_text to a fresh message in the admin's chat instead of an actual
    reply, since there's no incoming message to reply to here."""

    def __init__(self, bot, chat_id):
        self._bot = bot
        self._chat_id = chat_id

    async def reply_text(self, text, **kwargs):
        return await self._bot.send_message(chat_id=self._chat_id, text=text, **kwargs)


class _FakeUser:
    def __init__(self, telegram_id):
        self.id = telegram_id


class _FakeUpdate:
    def __init__(self, bot, telegram_id):
        proxy = _ChatMessageProxy(bot, telegram_id)
        self.effective_user = _FakeUser(telegram_id)
        self.message = proxy
        self.effective_message = proxy


async def autonomous_check(context: ContextTypes.DEFAULT_TYPE):
    from bot.handlers import run_search  # local import: avoids a circular import at module load

    try:
        fake_update = _FakeUpdate(context.bot, ADMIN_TELEGRAM_ID)
        found_something = await run_search(
            fake_update,
            context,
            silent_when_empty=True,
            min_percent=MIN_AUTONOMOUS_MATCH_PERCENT,
        )
        if not found_something:
            # A short heartbeat rather than full silence -- while this is
            # still a pilot, the person needs proof the scheduled check is
            # actually firing, not just an absence of news that could
            # equally mean the job died.
            await context.bot.send_message(
                chat_id=ADMIN_TELEGRAM_ID,
                text="🤖 Автоматична перевірка: нових вакансій немає.",
            )
    except Exception:
        logger.exception("Autonomous search run failed")
