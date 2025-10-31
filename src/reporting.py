import pandas as pd


def print_evaluation_report(results_df: pd.DataFrame):
    """RAGAS sonuçlarını detaylı ve özet olarak terminale basar."""

    print("\n" + "=" * 100)
    print(" RAGAS DEĞERLENDİRME SONUÇLARI ".center(100))
    print("=" * 100 + "\n")

    for idx, row in results_df.iterrows():
        print(f" SORU {idx + 1}: {row['question']}")
        print("-" * 100)

        answer = str(row['answer'])
        print(f"  Cevap: {answer[:90]}..." if len(answer) > 90 else f"  Cevap: {answer}")
        print(f"  Ground Truth: {row['ground_truth']}")
        print(f"  Alınan Belgeler: {len(row['contexts'])} adet")
        print()
        print(f"  METRIK SONUÇLARI (Puan 0.0 - 1.0):")
        print(f"    • Faithfulness (Sadakat):      {row['faithfulness']:.4f}")
        print(f"    • Answer Relevancy (İlgililik): {row['answer_relevancy']:.4f}")
        print(f"    • Context Precision (Kesinlik): {row['context_precision']:.4f}")
        print(f"    • Context Recall (Hatırlama):  {row['context_recall']:.4f}")
        print("\n")

    print("=" * 100)
    print(" ÖZET İSTATİSTİKLER ".center(100))
    print("=" * 100 + "\n")

    metrics_summary = {
        'Faithfulness': 'faithfulness',
        'Answer Relevancy': 'answer_relevancy',
        'Context Precision': 'context_precision',
        'Context Recall': 'context_recall'
    }

    for metric_name, metric_col in metrics_summary.items():
        if metric_col in results_df.columns:
            avg = results_df[metric_col].mean()
            print(f"  {metric_name:<20}: {avg:.4f}")

    print("\n" + "=" * 100)
    print("\n Değerlendirme tamamlandı!\n")