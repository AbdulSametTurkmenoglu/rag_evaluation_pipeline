import json
from typing import List, Dict
from llama_index.core import Document

def load_corpus(path: str = "data/corpus.jsonl") -> List[Document]:
    """İndekslenecek dokümanları .jsonl dosyasından yükler."""
    documents = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            documents.append(Document(text=data["text"]))
    print(f"Corpus için {len(documents)} doküman yüklendi.")
    return documents

def load_test_questions(path: str = "data/test_questions.jsonl") -> List[Dict]:
    """Değerlendirme için soruları ve ground truth'ları yükler."""
    questions = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            questions.append(json.loads(line))
    print(f"Değerlendirme için {len(questions)} soru yüklendi.")
    return questions