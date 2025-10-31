import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()

@dataclass
class Config:
    # API Anahtarları
    ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY")

    # Modeller
    EMBEDDING_MODEL: str = "intfloat/multilingual-e5-large"
    RAG_LLM_MODEL: str = "claude-3-5-sonnet-20240620"
    EVALUATION_LLM_MODEL: str = "claude-3-haiku-20240307" # Değerlendirme için daha hızlı bir model

    # Depolama
    STORAGE_DIR: str = "./storage"

    def __post_init__(self):
        if not self.ANTHROPIC_API_KEY:
            raise EnvironmentError("ANTHROPIC_API_KEY .env dosyasında bulunamadı")
        print("Config yüklendi. RAG LLM: Sonnet, Değerlendirme LLM: Haiku.")