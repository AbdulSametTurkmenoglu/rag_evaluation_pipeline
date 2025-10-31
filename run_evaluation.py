import sys
from datasets import Dataset
from src.config import Config
from src.data_loader import load_corpus, load_test_questions
from src.rag_system import build_or_load_index, query_rag_pipeline
from src.evaluation import run_ragas_evaluation
from src.reporting import print_evaluation_report


def main():
    try:
        cfg = Config()
    except EnvironmentError as e:
        print(f"Hata: {e}")
        print(".env dosyasını kontrol edin. ANTHROPIC_API_KEY gerekli.")
        sys.exit(1)

    # 1. Index için dokümanları yükle
    corpus_documents = load_corpus(path="data/corpus.jsonl")

    # 2. RAG sisteminin indexini oluştur/yükle
    index = build_or_load_index(source_documents=corpus_documents, cfg=cfg)

    # 3. Değerlendirme sorularını yükle
    test_questions = load_test_questions(path="data/test_questions.jsonl")

    print("\nTest sorguları RAG sistemiyle çalıştırılıyor...")
    eval_data = []
    for item in test_questions:
        question = item["question"]
        ground_truth = item["ground_truth"]

        # 4. RAG sisteminden cevap ve bağlam al
        answer, contexts = query_rag_pipeline(index, question, cfg)

        eval_data.append({
            "question": question,
            "answer": answer,
            "contexts": contexts,
            "ground_truth": ground_truth,
        })
        print(f"Soru '{question[:30]}...' cevaplandı.")

    # 5. RAGAS için Dataset objesi oluştur
    eval_dataset = Dataset.from_list(eval_data)

    # 6. RAGAS değerlendirmesini çalıştır
    results = run_ragas_evaluation(eval_dataset, cfg=cfg)
    results_df = results.to_pandas()

    # 7. Sonuçları raporla
    print_evaluation_report(results_df)


if __name__ == "__main__":
    main()