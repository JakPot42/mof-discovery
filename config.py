import os

DEMO_MODE = os.getenv("DEMO_MODE", "True").lower() in ("1", "true", "yes")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
DATABASE_URL = os.getenv("DATABASE_URL", "mof.db")
CLAUDE_MODEL = "claude-haiku-4-5-20251001"
TOP_K_DEFAULT = 10
