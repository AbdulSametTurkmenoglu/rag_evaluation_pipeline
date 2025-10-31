import os
from llama_index.core import (
    Document,
    VectorStoreIndex,
    StorageContext,
    load_index_from_storage,
    Settings
)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.anthropic import Anthropic
from .config import Config


def build_or_load_index(source_documents: list[Document], cfg: Config) -> VectorStoreIndex:
    """Diskten index yükler veya yoksa yenisini oluşturur."""

    # Embedding modelini LlamaIndex Settings'e global olarak ata
    Settings.embed_model = HuggingFaceEmbedding(model_name=cfg.EMBEDDING_MODEL)

    required_file = os.path.join(cfg.STORAGE_DIR, "docstore.json")

    if not os.path.exists(required_file):
        print(f"'{cfg.STORAGE_DIR}' dizininde index bulunamadı. Yeni index oluşturuluyor...")
        index = VectorStoreIndex.from_documents(source_documents)
        index.storage_context.persist(persist_dir=cfg.STORAGE_DIR)
        print(f"Index, {len(source_documents)} belge ile oluşturuldu ve kaydedildi.")
    else:
        print(f"'{cfg.STORAGE_DIR}' dizininde mevcut index bulundu. Yükleniyor...")
        storage_context = StorageContext.from_defaults(persist_dir=cfg.STORAGE_DIR)
        index = load_index_from_storage(storage_context)
        print("Index başarıyla yüklendi.")
    return index


def query_rag_pipeline(index: VectorStoreIndex, query_str: str, cfg: Config):
    """
    Test edilecek RAG pipeline'ını (Query Engine) çalıştırır.
    """
    llm = Anthropic(
        model=cfg.RAG_LLM_MODEL,
        api_key=cfg.ANTHROPIC_API_KEY,
        temperature=0.1,
        max_tokens=1024
    )
    query_engine = index.as_query_engine(llm=llm, similarity_top_k=2)

    response = query_engine.query(query_str)

    # RAGAS için gerekli çıktıları formatla
    answer = str(response)
    contexts = [node.get_content() for node in response.source_nodes]

    return answer, contexts