import torch
import logging
import re
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_community.llms import HuggingFacePipeline
from config import *

# Настройка логгирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Проверка GPU
logger.info("Initializing RAG engine...")
logger.info(f"CUDA available: {torch.cuda.is_available()}")

# Инициализация модели и токенизатора
llm = None
vector_store = None

try:
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME, 
        trust_remote_code=True
    )
    
    if torch.cuda.is_available():
        device = "cuda"
        torch_dtype = torch.float16
        logger.info("Using GPU acceleration")
    else:
        device = "cpu"
        torch_dtype = torch.float32
        logger.info("Using CPU mode")
    
    model_kwargs = {
        "device_map": "auto" if device == "cuda" else None,
        "torch_dtype": torch_dtype,
    }
    
    if device == "cpu":
        model_kwargs["device_map"] = None
        model_kwargs["low_cpu_mem_usage"] = True
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        **model_kwargs
    )
    
    if device == "cpu":
        model = model.to("cpu")
    
    logger.info(f"Model loaded: {MODEL_NAME}")
    
    text_generation_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=MAX_NEW_TOKENS,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True
    )
    
    llm = HuggingFacePipeline(pipeline=text_generation_pipeline)
    logger.info("Text generation pipeline created")
    
except Exception as e:
    logger.error(f"Error loading model: {str(e)}")

try:
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={'device': 'cpu'}
    )
    
    vector_store = FAISS.load_local(
        FAISS_INDEX_PATH, 
        embeddings, 
        allow_dangerous_deserialization=True
    )
    logger.info("Vector store loaded")
except Exception as e:
    logger.error(f"Error loading vector store: {str(e)}")

def clean_response(text: str) -> str:
    """
    Очистка ответа от системных тегов и артефактов генерации
    
    Параметры:
        text (str): Текст ответа от модели
        
    Возвращает:
        str: Очищенный текст
    """
    text = re.sub(r'<\|[^>]+\|>', '', text)
    
    text = re.sub(r'\b(system|ассистент|assistant|user|контекст):?\s*', '', text, flags=re.IGNORECASE)
    
    text = re.sub(r'[\n\s]*(ответ|проблема|рекомендация|источник)[:\d]*[\s]*', '', text, flags=re.IGNORECASE)
    
    text = re.sub(r'\n{3,}', '\n\n', text)  # >2 переносов → 2 переноса
    text = re.sub(r'[ \t]{2,}', ' ', text)   # Множественные пробелы → 1
    text = text.strip()
    
    return text

def generate_response(query):
    if llm is None or vector_store is None:
        return None
        
    try:
        relevant_docs = vector_store.similarity_search(query, k=TOP_K_RESULTS)
        
        context = "\n".join([f"[{i+1}] {doc.page_content}" 
                            for i, doc in enumerate(relevant_docs)])
        
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user", 
                "content": USER_PROMPT_TEMPLATE.format(
                    context=context,
                    question=query
                )
            }
        ]
        
        tokenizer.chat_template = "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
        formatted_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
        
        response_start = len(inputs.input_ids[0])
        decoded = tokenizer.decode(outputs[0][response_start:], skip_special_tokens=True)
        
        clean_answer = clean_response(decoded)
        
        sources = [f"[{i+1}] {doc.metadata['source']}" 
                  for i, doc in enumerate(relevant_docs)]
        
        return clean_answer, sources
        
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        return None