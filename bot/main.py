import os
import asyncio
import logging
from aiogram import Bot, Dispatcher
from aiogram.filters import Command
from aiogram.types import Message
from aiogram.client.default import DefaultBotProperties
from dotenv import load_dotenv

from rag_engine import generate_response

# Загрузка конфигурации
load_dotenv()
TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")

# Инициализация бота без форматирования
bot = Bot(TOKEN)
dp = Dispatcher()

# Настройка логгирования
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def split_message(text: str, max_len: int = 4000) -> list:
    """Сплит если сообщение длинное"""
    parts = []
    while len(text) > max_len:
        split_index = text.rfind('\n', 0, max_len)
        if split_index == -1:
            split_index = max_len
        parts.append(text[:split_index])
        text = text[split_index:].lstrip()
    parts.append(text)
    return parts

@dp.message(Command("start"))
async def cmd_start(message: Message):
    """ /start """
    welcome_msg = (
        "🔍 Добро пожаловать в бот!\n\n"
        "Я помогу вам анализировать тренды, онбординг пользователей и конкурентную среду.\n\n"
        "Примеры запросов:\n"
        "• Какие боли в онбординге выявлены в 2025 году?\n"
        "• Сравните UX-тренды 2024 и 2025 годов\n"
        "• Какие методики продуктового роста популярны в 2025?\n\n"
        "Просто задайте вопрос, и я предоставлю ответ на основе актуальных исследований!"
    )
    await message.answer(welcome_msg)

@dp.message()
async def handle_query(message: Message):
    """Обработка текстовых запросов с индикацией процесса"""
    try:
        user_id = message.from_user.id
        query = message.text
        
        logger.info(f"Новый запрос от {user_id}: {query}")
        
        # Показать статус "печатает"
        await message.bot.send_chat_action(
            chat_id=message.chat.id, 
            action="typing"
        )
        
        # Отправляем сообщение о поиске информации
        search_msg = await message.answer("🔍 Ищу информацию в базе знаний...")
        
        # Запуск генерации ответа в отдельном потоке
        result = await asyncio.to_thread(generate_response, query)
        
        # Удаляем сообщение о поиске перед отправкой ответа
        try:
            await message.bot.delete_message(
                chat_id=message.chat.id,
                message_id=search_msg.message_id
            )
        except Exception as e:
            logger.warning(f"Не удалось удалить сообщение поиска: {str(e)}")
        
        if not result or not result[0].strip():
            await message.answer("⚠️ Не удалось найти информацию по вашему запросу. Попробуйте переформулировать вопрос.")
            return
            
        answer, sources = result
        
        # Форматирование ответа
        response_text = f"{answer}\n\n🔍 Источники:\n" + "\n".join(sources)
        
        # Разбиваем сообщение на части если оно слишком длинное
        if len(response_text) > 4000:
            message_parts = split_message(response_text)
            for part in message_parts:
                await message.answer(part)
        else:
            await message.answer(response_text)
        
    except Exception as e:
        logger.error(f"Ошибка обработки запроса: {str(e)}")
        await message.answer("🚫 Произошла внутренняя ошибка. Пожалуйста, попробуйте позже.")
        
        # Пытаемся удалить сообщение поиска и в случае ошибки
        try:
            if 'search_msg' in locals():
                await message.bot.delete_message(
                    chat_id=message.chat.id,
                    message_id=search_msg.message_id
                )
        except:
            pass

async def main():
    """Запуск бота"""
    logger.info("Запуск бота продуктовых исследований...")
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())