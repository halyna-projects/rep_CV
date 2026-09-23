import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

TELEGRAM_BOT_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]

# Optional: without this, the bot falls back to plain keyword matching
# instead of real semantic CV-vs-vacancy comparison (see bot/semantic_matching.py).
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

DB_PATH = DATA_DIR / "bot.db"

UPLOADS_DIR = BASE_DIR / "uploads"
UPLOADS_DIR.mkdir(exist_ok=True)

HTTP_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)

# how many vacancies each source returns per keyword before dedup/limiting
RESULTS_PER_KEYWORD = 30

# Telegram numeric ID allowed to run /stats -- everyone else gets no reply.
ADMIN_TELEGRAM_ID = 2104700983

# Own test accounts excluded from /stats counts (not real users) --
# includes the admin's own account, so /stats shows only actual testers.
EXCLUDED_TELEGRAM_IDS = {7644385945, ADMIN_TELEGRAM_ID}

# New users get this many free AI actions (search/manual-add/apply) before
# being asked to wait for the admin to grant them full access via /grant --
# keeps an unexpected wave of new people from running up the Gemini bill
# before the admin has actually agreed to let them use the bot.
FREE_TRIAL_AI_ACTIONS = 2
