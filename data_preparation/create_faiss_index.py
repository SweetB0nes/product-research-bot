import os
import logging
from langchain_community.document_loaders import WebBaseLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

def load_documents():
    """Загрузка документов из русскоязычных источников по онбордингу"""
    sources = [
        # Основные источники на русском (2024-2025)
        "https://www.kickidler.com/ru/info/onbording-sotrudnikov-vsyo-chto-vazhno-znat-biznesu-dlya-uspeshnoj-adaptaczii-novichkov",
        "https://grandawards.ru/blog/onbording/",
        "https://tech-recruiter.ru/blog/onboarding-sotrudnikov-2025",
        "https://didit.me/ru/blog/what-is-digital-onboarding-key-strategies-for-attracting-new-customers-in-2024",
        "https://pritula.academy/adaptation",
        "https://hrpp.quorumconference.ru/",
        
        # Дополнительные материалы
        "https://hr-elearning.ru/blog/onboarding-sotrudnikov-2025",
        "https://www.insales.ru/blogs/university/onboarding-sotrudnikov",
        "https://blog.skillfactory.ru/onboarding-sotrudnikov-v-it/"
    ]
    
    docs = []
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
        "Accept-Language": "ru-RU,ru;q=0.9,en-US;q=0.8,en;q=0.7"
    }
    
    for source in sources:
        try:
            logger.info(f"Загрузка источника: {source}")
            loader = WebBaseLoader(
                web_path=source,
                requests_kwargs={"headers": headers}
            )
            web_docs = loader.load()
            
            for doc in web_docs:
                doc.metadata["source"] = source
                doc.metadata["language"] = "ru"
            
            docs.extend(web_docs)
            logger.info(f"Успешно загружено документов: {len(web_docs)}")
            
        except Exception as e:
            logger.error(f"Ошибка загрузки {source}: {str(e)}")
    
    logger.info(f"Всего загружено документов: {len(docs)}")
    return docs

def create_vector_index():
    """Создание FAISS векторного индекса на CPU"""
    docs = load_documents()
    
    # Разбивка на чанки
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    chunks = text_splitter.split_documents(docs)
    logger.info(f"Всего чанков: {len(chunks)}")
    
    embeddings = HuggingFaceEmbeddings(
        model_name="intfloat/multilingual-e5-base",
        model_kwargs={'device': 'cpu'}
    )
    
    vector_store = FAISS.from_documents(chunks, embeddings)
    
    index_path = "research_faiss_index"
    vector_store.save_local(index_path)
    logger.info(f"FAISS индекс сохранен в {index_path}")

if __name__ == "__main__":
    create_vector_index()