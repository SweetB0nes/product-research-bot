import os
from dotenv import load_dotenv

load_dotenv()

# Конфигурация модели
MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct" 
USE_QUANTIZATION = False  

# Параметры RAG
FAISS_INDEX_PATH = "research_faiss_index"
EMBEDDING_MODEL = "intfloat/multilingual-e5-base"
TOP_K_RESULTS = 5

# Параметры генерации
MAX_NEW_TOKENS = 400    
TEMPERATURE = 0.2    
TOP_P = 0.85

# Telegram
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")

# Промты
SYSTEM_PROMPT = """
Ты экспертный HR-ассистент по онбордингу. Отвечай на вопросы, используя только предоставленный контекст.
Всегда придерживайся структуры:
1. Ключевые тезисы
2. Сравнительные данные
3. Практические рекомендации
4. Источники: [№]
"""

USER_PROMPT_TEMPLATE = """
Контекст:
{context}

Вопрос: {question}
"""