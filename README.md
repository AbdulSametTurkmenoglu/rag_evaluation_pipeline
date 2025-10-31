# RAG Değerlendirme Pipeline'ı (RAGAS)

Bu proje, bir RAG (Retrieval-Augmented Generation) sisteminin kalitesini, `ragas` kütüphanesi ve `Anthropic` (Claude) modelleri kullanarak otomatik olarak değerlendiren bir test pipeline'ıdır.

Sistem, bir "test seti" (`test_questions.jsonl`) ve bir "bilgi havuzu" (`corpus.jsonl`) kullanarak, RAG sisteminin performansını 4 ana metrik üzerinden ölçer.

##  Değerlendirme Metrikleri

Bu pipeline, `ragas` kullanarak aşağıdaki 4 kritik metriği ölçer:

1.  **Faithfulness (Cevap Sadakati):**
    * *Soru:* Üretilen cevap, bulunan bağlam (context) tarafından ne kadar destekleniyor?
    * *Ölçüm:* Cevaptaki hangi ifadelerin bağlamda karşılığı *olmadığını* (halüsinasyon) tespit eder.

2.  **Answer Relevancy (Cevap İlgililiği):**
    * *Soru:* Üretilen cevap, sorulan soruyla ne kadar ilgili?
    * *Ölçüm:* Cevabın, sorudan ne kadar saptığını veya gereksiz bilgi içerip içermediğini analiz eder.

3.  **Context Precision (Bağlam Kesinliği):**
    * *Soru:* Bulunan bağlamın (context) ne kadarı cevabı oluşturmak için *gerçekten* gerekliydi?
    * *Ölçüm:* Bağlamdaki "gürültüyü" (noise) ölçer. Yüksek puan, sistemin sadece ilgili belgeleri bulduğunu gösterir.

4.  **Context Recall (Bağlam Hatırlaması):**
    * *Soru:* Bulunan bağlam, "ideal" cevabı (ground truth) oluşturmak için yeterli bilgiyi içeriyor mu?
    * *Ölçüm:* Sistemin, doğru cevabı verebilmek için gerekli olan bilgiyi bulup bulamadığını ölçer.





##  Kurulum

### 1. Ön Gereksinimler
* **Ollama:** `nomic-embed-text` modelinin çalışması için gereklidir (veya `src/config.py`'den değiştirin).
    ```bash
    ollama pull nomic-embed-text
    ```
* **Anthropic API Key:** Claude modellerini kullanmak için bir API anahtarı gereklidir.

### 2. Yerel Kurulum

1.  **Depoyu Klonlama:**
    ```bash
    git clone [https://github.com/AbdulSametTurkmenoglu/rag_degerlendirme_pipeline.git](https://github.com/AbdulSametTurkmenoglu/rag_degerlendirme_pipeline.git)
    cd ragas
    ```

2.  **Sanal Ortam ve Kütüphaneler:**
    ```bash
    python -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt
    ```

3.  **.env Dosyası:**
    `.env.example` dosyasını `.env` olarak kopyalayın ve `ANTHROPIC_API_KEY`'inizi girin.

##  Kullanım

Tüm kurulum tamamlandıktan sonra, değerlendirme pipeline'ını çalıştırmak için:

```bash
python run_evaluation.py
```

### Süreç Adımları:
Script çalıştığında sırayla şunları yapacaktır:

1.  `src/config.py`'den API anahtarlarını ve model adlarını okuyacaktır.
2.  `data/corpus.jsonl`'deki dokümanları kullanarak `storage/` klasöründe bir LlamaIndex (FAISS) oluşturacak veya varsa yükleyecektir.
3.  `data/test_questions.jsonl`'deki her bir soruyu, `src/rag_system.py`'de tanımlı RAG pipeline'ına soracaktır.
4.  RAG sisteminden dönen `answer` (cevap) ve `contexts` (bağlam) bilgilerini toplayacaktır.
5.  Bu bilgileri (`question`, `answer`, `contexts`, `ground_truth`) `ragas`'a gönderecektir.
6.  `src/reporting.py`'deki fonksiyonu kullanarak terminale detaylı bir rapor basacaktır.

### Örnek Rapor Çıktısı

```
====================================================================================================
 RAGAS DEĞERLENDİRME SONUÇLARI
====================================================================================================

 SORU 1: Suç ve Ceza romanında Raskolnikov'un suçu nedir?
----------------------------------------------------------------------------------------------------
  Cevap: Raskolnikov'un işlediği suç, bir tefeci kadını öldürmesidir.
  Ground Truth: Raskolnikov bir tefeci kadını öldürür.
  Alınan Belgeler: 1 adet

  METRIK SONUÇLARI (Puan 0.0 - 1.0):
    • Faithfulness (Sadakat):      1.0000
    • Answer Relevancy (İlgililik): 0.9850
    • Context Precision (Kesinlik): 1.0000
    • Context Recall (Hatırlama):  1.0000

====================================================================================================
 ÖZET İSTATİSTİKLER
====================================================================================================

  Faithfulness        : 0.9500
  Answer Relevancy    : 0.9925
  Context Precision   : 1.0000
  Context Recall      : 1.0000

====================================================================================================

 Değerlendirme tamamlandı!
```