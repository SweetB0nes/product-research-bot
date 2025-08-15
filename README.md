# 🤖RAG-система (Retrieval-Augmented Generation) для продуктового исследования с интеграцией в Telegram

## ✨ Особенности
- Поиск информации в предварительно подготовленной базе знаний
- Генерация структурированных ответов на русском языке
- Поддержка работы как на GPU, так и на CPU
- Форматирование ответов с источниками информации

## ⚙️ Установка и запуск
1. Клонировать репозиторий:

        git clone https://github.com/your-username/product-research-bot.git
    
        cd product-research-bot

2. Создать виртуальное окружение и активировать его:

        python -m venv venv
    Для Windows:
   
        venv\Scripts\activate
    Для Linux/macOS:
   
        source venv/bin/activate

4. Установить зависимости:

        pip install -r requirements.txt

5. Создать векторный индекс (база знаний):

        python data_preparation/create_faiss_index.py 

    При необходимости обогатить исчтоники

6. Создать файл .env:

    TELEGRAM_BOT_TOKEN="ваш_телеграм_токен"
    HF_API_TOKEN="ваш_hugging_face_токен"  # Для доступа к модели Qwen

7. Запустить бота: 

        python bot/main.py  

## 🔧 Настройка параметров
- MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"  # Используемая LLM
- USE_QUANTIZATION = False                 # Использование квантования
- TOP_K_RESULTS = 5                        # Кол-во извлекаемых документов
- MAX_NEW_TOKENS = 400                     # Макс. длина ответа
- TEMPERATURE = 0.2                        # Креативность ответов
- TOP_P = 0.85                             # Качество генерации

## 🔑 Получение токенов
1. Telegram Bot Token:
    - Создайте бота через @BotFather
    - Скопируйте токен вида 1234567890:ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghi

2. Hugging Face Token:
    - Зарегистрируйтесь на huggingface.co
    - Перейдите в Settings → Access Tokens
    - Создайте новый токен с ролью "Read"
