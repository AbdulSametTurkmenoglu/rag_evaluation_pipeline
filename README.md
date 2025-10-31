# RAG Evaluation Pipeline (RAGAS)

An automated testing pipeline for evaluating RAG (Retrieval-Augmented Generation) system quality using the `ragas` library and `Anthropic` (Claude) models.

The system measures RAG performance across 4 core metrics using a test set (`test_questions.jsonl`) and a knowledge corpus (`corpus.jsonl`).

##  Evaluation Metrics

This pipeline uses `ragas` to measure 4 critical metrics:

1. **Faithfulness (Answer Fidelity)**
   - *Question:* How well is the generated answer supported by the retrieved context?
   - *Measures:* Detects statements in the answer that are *not* backed by context (hallucinations)

2. **Answer Relevancy**
   - *Question:* How relevant is the generated answer to the asked question?
   - *Measures:* Analyzes whether the answer deviates from the question or contains unnecessary information

3. **Context Precision**
   - *Question:* How much of the retrieved context was *actually* necessary to generate the answer?
   - *Measures:* Noise in the context. High score indicates the system retrieved only relevant documents

4. **Context Recall**
   - *Question:* Does the retrieved context contain sufficient information to generate the "ideal" answer (ground truth)?
   - *Measures:* Whether the system can find the necessary information to provide the correct answer

##  Prerequisites

- **Ollama**: Required for `nomic-embed-text` model (or modify in `src/config.py`)
- **Anthropic API Key**: Required for Claude models
- Python 3.8+

##  Installation

**Install Ollama embedding model:**
```bash
ollama pull nomic-embed-text
```

**Clone and setup:**
```bash
# Clone the repository
git clone https://github.com/AbdulSametTurkmenoglu/rag_evaluation_pipeline.git
cd ragas

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .\.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Setup environment variables
cp .env.example .env
# Edit .env and add your ANTHROPIC_API_KEY
```

##  Usage

**Run the evaluation pipeline:**
```bash
python run_evaluation.py
```

### Pipeline Steps

When executed, the script will:

1. Load API keys and model names from `src/config.py`
2. Create or load a LlamaIndex (FAISS) from documents in `data/corpus.jsonl` (stored in `storage/`)
3. Query each question from `data/test_questions.jsonl` through the RAG pipeline defined in `src/rag_system.py`
4. Collect `answer` and `contexts` returned by the RAG system
5. Send the collected data (`question`, `answer`, `contexts`, `ground_truth`) to `ragas` for evaluation
6. Print a detailed report to terminal using `src/reporting.py`

### Sample Report Output
```
====================================================================================================
 RAGAS EVALUATION RESULTS
====================================================================================================

 QUESTION 1: What is Raskolnikov's crime in Crime and Punishment?
----------------------------------------------------------------------------------------------------
  Answer: Raskolnikov's crime is murdering a pawnbroker woman.
  Ground Truth: Raskolnikov murders a pawnbroker woman.
  Retrieved Documents: 1

  METRIC RESULTS (Score 0.0 - 1.0):
    • Faithfulness:         1.0000
    • Answer Relevancy:     0.9850
    • Context Precision:    1.0000
    • Context Recall:       1.0000

====================================================================================================
 SUMMARY STATISTICS
====================================================================================================

  Faithfulness        : 0.9500
  Answer Relevancy    : 0.9925
  Context Precision   : 1.0000
  Context Recall      : 1.0000

====================================================================================================

 Evaluation completed!
```


##  Configuration

Edit `src/config.py` to customize:
- Embedding model (default: `nomic-embed-text` via Ollama)
- LLM model (default: Claude via Anthropic)
- Evaluation metrics
- Storage paths

