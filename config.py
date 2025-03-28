import os

# API Configuration
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "")

# Database Configuration
CHROMA_DB_PATH = "./chroma_db"
COLLECTION_NAME = "research_papers"

# Rate Limiting Configuration
MAX_CALLS_PER_MINUTE = 10
MAX_RETRIES = 5
INITIAL_RETRY_DELAY = 2

# Server Configuration
SERVER_HOST = "0.0.0.0"
SERVER_PORT = 8000
