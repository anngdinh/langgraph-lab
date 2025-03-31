from dotenv import load_dotenv
import os
_ = load_dotenv(override=True)  # read local .env file

################################################################
from langchain_google_genai import ChatGoogleGenerativeAI
gemini_flash = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    api_key=os.getenv('GEMINI_API_KEY'),
    temperature=0,
    # verbose=True,
)

def New_Gemini_Flash():
    return ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        api_key=os.getenv('GEMINI_API_KEY'),
        temperature=0,
        # verbose=True,
    )

gemini_pro = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro",
    api_key=os.getenv('GEMINI_API_KEY'),
    temperature=0,
    # verbose=True,
)

def New_Gemini_Pro():
    return ChatGoogleGenerativeAI(
        model="gemini-1.5-pro",
        api_key=os.getenv('GEMINI_API_KEY'),
        temperature=0,
    )

from langchain_deepseek import ChatDeepSeek
deepseek = ChatDeepSeek(
    model="deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
    api_key=os.getenv('DEEPSEEK_API_KEY'),
    api_base="https://model-000.ai-gateway.vngcloud.tech/deepseek/chat/completions",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

from langchain_ollama import ChatOllama
ollama = ChatOllama(
    model="llama3.2",
    # api_key=os.getenv('OLLAMA_API_KEY'),
    temperature=0,
    verbose=True
)
