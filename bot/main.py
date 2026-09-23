import logging

from telegram import Update
from telegram.ext import (
    Application,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from bot import handlers, storage
from bot.config import TELEGRAM_BOT_TOKEN

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)

logger = logging.getLogger(__name__)


async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE):
    # Without this, an exception (e.g. a transient network timeout talking
    # to Telegram) just gets logged and the user is left staring at
    # nothing with no idea anything went wrong - see the ConnectTimeout
    # that silently killed a /search mid-results.
    logger.exception("Unhandled exception while processing update", exc_info=context.error)

    if isinstance(update, Update) and update.effective_message:
        try:
            await update.effective_message.reply_text(
                "Щось пішло не так (схоже, збій мережі). Спробуйте ще раз."
            )
        except Exception:
            logger.exception("Could not even notify the user about the earlier error")


def main():
    storage.init_db()

    app = (
        Application.builder()
        .token(TELEGRAM_BOT_TOKEN)
        .connect_timeout(20)
        .read_timeout(20)
        .get_updates_connect_timeout(20)
        .get_updates_read_timeout(20)
        .build()
    )

    app.add_handler(CommandHandler("start", handlers.start))
    app.add_handler(CommandHandler("help", handlers.help_cmd))
    app.add_handler(CommandHandler("keywords", handlers.set_keywords))
    app.add_handler(CommandHandler("location", handlers.set_location))
    app.add_handler(CommandHandler("cv", handlers.cv_status))
    app.add_handler(CommandHandler("search", handlers.run_search))
    app.add_handler(CommandHandler("reset", handlers.reset_seen))
    app.add_handler(CommandHandler("apply", handlers.apply_to_vacancy))
    app.add_handler(CommandHandler("stats", handlers.stats))
    app.add_handler(CommandHandler("grant", handlers.grant))
    app.add_handler(CommandHandler("revoke", handlers.revoke))
    app.add_handler(CallbackQueryHandler(handlers.handle_callback))
    app.add_handler(MessageHandler(filters.Document.ALL, handlers.handle_document))
    app.add_handler(MessageHandler(filters.PHOTO, handlers.handle_photo))
    app.add_handler(
        MessageHandler(filters.TEXT & ~filters.COMMAND, handlers.handle_plain_text)
    )
    app.add_error_handler(error_handler)

    logging.info("Bot starting (polling)...")
    app.run_polling()


if __name__ == "__main__":
    main()
