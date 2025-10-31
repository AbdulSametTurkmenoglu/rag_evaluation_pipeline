from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from langchain_anthropic import ChatAnthropic
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_community.embeddings import HuggingFaceEmbeddings
from datasets import Dataset

from .config import Config


def configure_ragas_dependencies(cfg: Config):
    """RAGAS'ın değerlendirme yapmak için kullanacağı LLM ve Embedding'leri ayarlar."""

    # Değerlendirme için LLM (Daha hızlı ve ucuz bir model)
    langchain_llm = ChatAnthropic(
        model=cfg.EVALUATION_LLM_MODEL,
        api_key=cfg.ANTHROPIC_API_KEY
    )
    ragas_llm = LangchainLLMWrapper(langchain_llm)

    # RAGAS'ın metrikleri hesaplamak için kullanacağı Embedding modeli
    hf_embeddings = HuggingFaceEmbeddings(model_name=cfg.EMBEDDING_MODEL)
    ragas_embeddings = LangchainEmbeddingsWrapper(hf_embeddings)

    return ragas_llm, ragas_embeddings


def run_ragas_evaluation(eval_dataset: Dataset, cfg: Config):
    """
    RAGAS değerlendirme sürecini başlatır.
    """
    ragas_llm, ragas_embeddings = configure_ragas_dependencies(cfg)

    metrics = [
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
    ]

    print("\nRAGAS değerlendirmesi başlatılıyor...")
    print(f"Değerlendirme LLM'i: {cfg.EVALUATION_LLM_MODEL}")

    results = evaluate(
        dataset=eval_dataset,
        metrics=metrics,
        llm=ragas_llm,
        embeddings=ragas_embeddings
    )

    return results