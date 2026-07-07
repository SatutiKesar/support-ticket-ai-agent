"""
Central configuration, read from environment variables (see .env.example).
"""
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

# --- Data ---
CSV_PATH = os.getenv("TICKETS_CSV", str(BASE_DIR / "data" / "support_tickets.csv"))
SQLITE_PATH = os.getenv("TICKETS_DB", str(BASE_DIR / "data" / "tickets.db"))
TABLE_NAME = "tickets"

# --- LLM provider ---
# One of: "groq", "ollama", "huggingface", "offline"
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq").lower()

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3")

HF_API_TOKEN = os.getenv("HF_API_TOKEN", "")
HF_MODEL = os.getenv("HF_MODEL", "meta-llama/Meta-Llama-3-8B-Instruct")
HF_URL = f"https://api-inference.huggingface.co/models/{HF_MODEL}/v1/chat/completions"

# --- Anomaly detection ---
# "now"  -> use real wall-clock time as "today" when computing ticket age
# "data" -> use the most recent created_at timestamp found in the CSV as "today"
#           (recommended for this historical dataset, since its dates are in the past)
ANOMALY_REFERENCE_TIME = os.getenv("ANOMALY_REFERENCE_TIME", "data").lower()
STALE_HOURS_THRESHOLD = float(os.getenv("STALE_HOURS_THRESHOLD", "24"))
IQR_MULTIPLIER = float(os.getenv("IQR_MULTIPLIER", "1.5"))
HIGH_PRIORITY_LEVELS = {"High", "Critical"}
OPEN_STATUSES_EXCLUDED = {"Resolved", "Closed"}

# --- API ---
CORS_ALLOW_ORIGINS = os.getenv("CORS_ALLOW_ORIGINS", "*").split(",")
