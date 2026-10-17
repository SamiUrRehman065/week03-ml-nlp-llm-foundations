# Week 03: Machine Learning, NLP & LLM Foundations

A modular, production-ready implementation of foundational Machine Learning, Natural Language Processing (NLP), and Large Language Model (LLM) representation techniques.

---

## 📋 Table of Contents
1. [Overview](#-overview)
2. [Project Architecture](#-project-architecture)
3. [Environment Setup & Installation](#-environment-setup--installation)
4. [How to Run](#-how-to-run)
   - [Unified Execution](#unified-execution)
   - [Phase-by-Phase Execution](#phase-by-phase-execution)
   - [Direct Module Execution](#direct-module-execution)
5. [Phase Deep-Dives](#-phase-deep-dives)
   - [Phase 01: Text Classification (Naive Bayes)](#phase-01-text-classification-naive-bayes)
   - [Phase 02: NLP Text Preprocessing](#phase-02-nlp-text-preprocessing)
   - [Phase 03: Tokenization Suite (Codes 01–11)](#phase-03-tokenization-suite-codes-0111)
   - [Phase 04: Classification Pipelines & Model Comparison](#phase-04-classification-pipelines--model-comparison)
   - [Phase 05: Sentence Embeddings & Semantic Similarity](#phase-05-sentence-embeddings--semantic-similarity)
   - [Phase 06: Semantic Search Engine](#phase-06-semantic-search-engine)
6. [Theoretical Concepts & Checklist Answers](#-theoretical-concepts--checklist-answers)
7. [Evidence & Outputs](#-evidence--outputs)

---

## 🌟 Overview

This repository satisfies all learning outcomes and checklist deliverables specified for **Week 03**:
- **Classical Text Classification:** Bag-of-Words and TF-IDF feature extraction with Multinomial Naive Bayes, Linear Support Vector Machines (LinearSVC), and Logistic Regression.
- **Rule-Based Preprocessing:** Text normalization, whitespace collapsing, punctuation filtering, and case normalization using Regular Expressions.
- **Tokenization Algorithms:** Word-level, character-level, WordPiece (BERT), Byte-Pair Encoding (GPT-2), and Unigram Language Model training with Google's `SentencePiece`.
- **Dense Neural Representations:** Contextual embeddings generated with `sentence-transformers/all-MiniLM-L6-v2`.
- **Semantic Retrieval:** Cosine-similarity-based semantic search engine evaluated across multiple queries.

---

## 📂 Project Architecture

```
Week_03_ML_NLP_LLM_Foundations/
├── README.md                          # Comprehensive project documentation & theory
├── requirements.txt                   # All dependencies and package versions
├── main.py                            # Master runner for all phases or individual phases
│
├── data/
│   ├── support_tickets.csv            # University support ticket dataset (16 samples, 4 categories)
│   └── spm_training.txt               # Text corpus for SentencePiece tokenizer training
│
├── phase01_text_classification/
│   ├── __init__.py
│   └── naive_bayes_classifier.py      # CountVectorizer + MultinomialNB with evaluation
│
├── phase02_text_preprocessing/
│   ├── __init__.py
│   └── clean_text.py                  # Regex text cleaner with multi-sentence test suite
│
├── phase03_tokenization/
│   ├── __init__.py
│   ├── code01_bert_tokens.py          # Code 01: BERT tokenize & encode
│   ├── code02_bert_special_tokens.py  # Code 02: BERT special tokens & decoding
│   ├── code03_bpe_pair_counter.py     # Code 03: BPE adjacent character pair counter
│   ├── code04_char_vocab_mapping.py   # Code 04: Character tokenization & manual dictionary map
│   ├── code05_compare_tokenizers.py   # Code 05: Comparing BERT vs GPT-2 token counts
│   ├── code06_sentencepiece_train.py  # Code 06: SentencePiece Unigram training & inference
│   ├── code07_special_tokens_info.py  # Code 07: [CLS], [SEP] token properties and IDs
│   ├── code08_char_vs_tokens.py       # Code 08: Character count vs Token count compression ratio
│   ├── code09_whitespace_tokenizer.py # Code 09: Basic lowercase split tokenizer
│   ├── code10_vocab_unk_handling.py   # Code 10: Fixed vocabulary with <UNK> out-of-vocabulary fallback
│   ├── code11_bert_subwords.py        # Code 11: BERT subword (##) decomposition
│   └── tokenization_suite.py          # Unified runner executing Codes 01 through 11
│
├── phase04_classification_pipelines/
│   ├── __init__.py
│   ├── code01_pipeline_classifier.py  # Code 01: Pipeline(TfidfVectorizer, LogisticRegression) on CSV
│   └── code02_model_comparisons.py    # Code 02: Benchmarking LogisticRegression, LinearSVC, Naive Bayes
│
├── phase05_embeddings_similarity/
│   ├── __init__.py
│   └── semantic_similarity.py         # SentenceTransformer + Cosine Similarity calculations
│
├── phase06_semantic_search/
│   ├── __init__.py
│   └── document_search.py             # Semantic search engine with top-k document ranking
│
└── outputs/
    ├── phase01_output.txt             # Verified output trace for Phase 01
    ├── phase02_output.txt             # Verified output trace for Phase 02
    ├── phase03_output.txt             # Verified output trace for Phase 03
    ├── phase04_output.txt             # Verified output trace for Phase 04
    ├── phase05_output.txt             # Verified output trace for Phase 05
    ├── phase06_output.txt             # Verified output trace for Phase 06
    └── complete_run_output.txt        # Full console log of all phases executed via main.py
```

---

## ⚙️ Environment Setup & Installation

### 1. Create a Virtual Environment (Recommended)
On Windows (PowerShell / Command Prompt):
```bash
python -m venv venv
venv\Scripts\activate
```
On macOS / Linux:
```bash
python3 -m venv venv
source venv/bin/activate
```

### 2. Install Required Dependencies
Install all specified packages using `requirements.txt`:
```bash
pip install -r requirements.txt
```

The dependencies include:
- `sentencepiece`: Google's unsupervised subword tokenizer library.
- `sentence-transformers`: Dense sentence and document embedding framework.
- `transformers`: Hugging Face library providing BERT, GPT-2, and modern LLM tokenizers.
- `torch`: Deep learning tensor backend for transformer models.
- `scikit-learn`: Machine learning estimators, metrics, pipelines, and vectorizers.
- `pandas` & `numpy`: Tabular data processing and numerical arrays.
- `pypdf`, `python-docx`, `openpyxl`, `python-dotenv`, `fastapi`, `uvicorn`, `pydantic`.

---

## 🚀 How to Run

### Unified Execution
To execute all 6 phases end-to-end, run:
```bash
python main.py
```
*(or explicitly with `python main.py --all`)*

### Phase-by-Phase Execution
You can selectively run any single phase using the `--phase` argument:
```bash
python main.py --phase 1    # Phase 01: Naive Bayes Text Classification
python main.py --phase 2    # Phase 02: NLP Text Preprocessing
python main.py --phase 3    # Phase 03: Tokenization Suite (Codes 01 to 11)
python main.py --phase 4    # Phase 04: Classification Pipelines & Comparisons
python main.py --phase 5    # Phase 05: Sentence Embeddings & Semantic Similarity
python main.py --phase 6    # Phase 06: Semantic Search Engine
```

### Direct Module Execution
Every script is completely self-contained and can also be executed directly:
```bash
# Phase 01
python phase01_text_classification/naive_bayes_classifier.py

# Phase 02
python phase02_text_preprocessing/clean_text.py

# Phase 03 individual demonstrations
python phase03_tokenization/code01_bert_tokens.py
python phase03_tokenization/code05_compare_tokenizers.py
python phase03_tokenization/code06_sentencepiece_train.py
python phase03_tokenization/tokenization_suite.py

# Phase 04
python phase04_classification_pipelines/code01_pipeline_classifier.py
python phase04_classification_pipelines/code02_model_comparisons.py

# Phase 05
python phase05_embeddings_similarity/semantic_similarity.py

# Phase 06
python phase06_semantic_search/document_search.py
```

---

## 🔍 Phase Deep-Dives

### Phase 01: Text Classification (Naive Bayes)
- **Script:** `phase01_text_classification/naive_bayes_classifier.py`
- **Technique:** Bag-of-Words via `CountVectorizer`, Multinomial Naive Bayes (`MultinomialNB`).
- **Core Functionality:**
  - Converts raw text documents into a word occurrence matrix.
  - Fits prior probabilities $P(c)$ and term likelihoods $P(w|c)$.
  - Performs stratified train-test splitting and generates accuracy and classification report.
  - Predicts sentiment for new, unseen input (`"I really love this course"` -> `positive`).

### Phase 02: NLP Text Preprocessing
- **Script:** `phase02_text_preprocessing/clean_text.py`
- **Technique:** Regular Expressions (`re`) and string normalization.
- **Operations Performed:**
  1. `text.strip()`: Trims leading/trailing whitespace.
  2. `text.lower()`: Normalizes all characters to lowercase.
  3. `re.sub(r"\s+", " ", text)`: Collapses repeated tabs, spaces, and newlines into single spaces.
  4. `re.sub(r"[^\w\s]", "", text)`: Filters out punctuation and special symbols.
- **Test Suite:** Evaluated across 6 test sentences with varying casing, punctuation, and spacing.

### Phase 03: Tokenization Suite (Codes 01–11)
- **Suite Runner:** `phase03_tokenization/tokenization_suite.py`
- **Modules:**
  - **Code 01 (`code01_bert_tokens.py`):** Uses `AutoTokenizer.from_pretrained("bert-base-uncased")` to tokenize and encode sentences into numerical IDs.
  - **Code 02 (`code02_bert_special_tokens.py`):** Adds special delimiter tokens (`[CLS]`, `[SEP]`) during encoding and decodes IDs back into readable text.
  - **Code 03 (`code03_bpe_pair_counter.py`):** Demonstrates Byte-Pair Encoding pair frequency counting using `collections.Counter` on vocabulary terms `"low lower lowest"`.
  - **Code 04 (`code04_char_vocab_mapping.py`):** Character-level tokenization mapping `'hello'` against manual character dictionary `{'h': 0, 'e': 1, 'l': 2, 'o': 3}`.
  - **Code 05 (`code05_compare_tokenizers.py`):** Benchmarks WordPiece (`bert-base-uncased`: 7 tokens) vs Byte-Pair Encoding (`gpt2`: 9 tokens) on identical text.
  - **Code 06 (`code06_sentencepiece_train.py`):** Trains an unsupervised Unigram language model with `SentencePieceTrainer` from scratch on `training.txt` (vocab size: 50) and performs encoding/decoding.
  - **Code 07 (`code07_special_tokens_info.py`):** Inspects `cls_token` (`[CLS]`, ID: 101) and `sep_token` (`[SEP]`, ID: 102).
  - **Code 08 (`code08_char_vs_tokens.py`):** Calculates compression ratio comparing character count (86 chars) against token count (11 tokens).
  - **Code 09 (`code09_whitespace_tokenizer.py`):** Basic whitespace splitting tokenizer (`tokenize(text: str) -> list[str]`).
  - **Code 10 (`code10_vocab_unk_handling.py`):** Fixed dictionary lookup with `<UNK>` (ID: 0) fallback for unknown/out-of-vocabulary words.
  - **Code 11 (`code11_bert_subwords.py`):** BERT WordPiece subword prefix (`##`) behavior demonstration.

### Phase 04: Classification Pipelines & Model Comparison
- **Scripts:** `code01_pipeline_classifier.py` & `code02_model_comparisons.py`
- **Dataset:** `data/support_tickets.csv` (16 university support tickets across 4 departments: `account`, `technical`, `finance`, `admission`).
- **Pipeline Architecture:**
  - `TfidfVectorizer`: Calculates Term Frequency-Inverse Document Frequency features.
  - `LogisticRegression(max_iter=1000)`: Optimizes multi-class cross-entropy loss.
- **Model Comparisons:** Benchmarks `LogisticRegression`, `LinearSVC`, and `MultinomialNB` on balanced sentiment data, demonstrating 100% precision/recall on held-out test splits.

### Phase 05: Sentence Embeddings & Semantic Similarity
- **Script:** `phase05_embeddings_similarity/semantic_similarity.py`
- **Model:** `sentence-transformers/all-MiniLM-L6-v2` (produces 384-dimensional dense vectors).
- **Metric:** Cosine Similarity $\cos(\theta) = \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\|_2 \|\mathbf{v}\|_2}$.
- **Findings:**
  - `"How do I reset my password?"` vs `"I forgot my password."` -> **0.7712** (Strong Semantic Alignment).
  - `"How do I reset my password?"` vs `"The weather is very hot today."` -> **0.0881** (Semantically Unrelated).

### Phase 06: Semantic Search Engine
- **Script:** `phase06_semantic_search/document_search.py`
- **Class:** `SemanticSearchEngine`
  - Encodes document catalog into a stored vector index.
  - Maps real-time user natural language queries into the shared latent space.
  - Computes cosine similarity across all indexed records and returns ranked results.
- **Evaluated Queries:**
  1. *"Where can I register my courses?"* -> **Matched:** "Students can register courses through the student portal." (Score: 0.7061)
  2. *"When will final exam timetable and date sheets be announced?"* -> **Matched:** "The examination office publishes examination schedules." (Score: 0.6845)
  3. *"I need help resolving my university tuition fee balance."* -> **Matched:** "The finance office handles tuition fee issues." (Score: 0.7410)
  4. *"How do I access academic journals, books, and study materials?"* -> **Matched:** "The library provides digital research resources." (Score: 0.6923)

---

## 🧠 Theoretical Concepts & Checklist Answers

### 1. Difference Between Training and Inference
- **Training:** The optimization phase where model parameters are computed from labeled training examples. In scikit-learn, calling `fit()` or `fit_transform()` causes the vectorizer to construct the vocabulary and count frequencies, while the classifier learns weights or class probability distributions.
- **Inference:** The prediction phase where the trained model processes novel input without modifying its internal parameters. The vectorizer calls `transform()` (using strictly the pre-established vocabulary), and the classifier computes the forward-pass prediction ($\hat{y} = \arg\max P(y|x)$).

### 2. Why Text Preprocessing is Useful
- **Vocabulary Compaction:** Lowercasing prevents duplicated tokens for identical words (e.g. avoiding three entries for "Exam", "exam", and "EXAM").
- **Noise Filtering:** Removing extraneous punctuation (e.g., `!!!`, `???`, `---`) prevents symbol attachment to words, ensuring "portal" and "portal!" resolve to the identical token.
- **Normalization:** Collapsing inconsistent whitespace guarantees uniform splitting across varied data sources.
- **Enhanced Generalization:** Higher lexical overlap between training and inference data improves out-of-sample accuracy.

### 3. Special Tokens and Subword Tokenization
- **Special Tokens:** Non-natural-language control tokens utilized by Transformer models:
  - `[CLS]` (ID: 101 in BERT): Marks sequence beginning; its final hidden state is utilized as aggregate sequence representation.
  - `[SEP]` (ID: 102 in BERT): Delimits sentences in single or paired text inputs.
  - `[PAD]` (ID: 0): Pads sequences within batches to equal tensor shapes.
  - `[UNK]` (ID: 100): Denotes out-of-vocabulary tokens.
- **Subword Tokenization (WordPiece / BPE / Unigram):**
  - Solves the fixed-vocabulary dilemma by splitting rare words into constituent morphological fragments (e.g. `transforming` -> `transform` + `##ing`).
  - Retains compact vocabulary size (30,000–50,000) while preventing out-of-vocabulary failures on rare or misspelled words.

### 4. Word Counts vs. Dense Semantic Embeddings
- **Word Counts / TF-IDF (Sparse):**
  - High-dimensional vectors (dimension equal to vocabulary size $V$, often 10,000+).
  - Vectors are orthogonal; synonyms ("car" and "automobile") share zero dot-product overlap.
  - Insensitive to word ordering, syntactic roles, or contextual semantics.
- **Dense Semantic Embeddings (Sentence Transformers):**
  - Low-dimensional continuous representations (e.g., 384 dimensions).
  - Contextual and deep: semantically related statements map to proximate vectors even with zero shared vocabulary.
  - Captures paraphrase, intent, and subtle semantic nuance via deep Transformer attention layers.

---

## 📊 Evidence & Outputs

All execution traces and terminal outputs have been captured and saved inside the `outputs/` folder:
- [`outputs/phase01_output.txt`](file:///outputs/phase01_output.txt)
- [`outputs/phase02_output.txt`](file:///outputs/phase02_output.txt)
- [`outputs/phase03_output.txt`](file:///outputs/phase03_output.txt)
- [`outputs/phase04_output.txt`](file:///outputs/phase04_output.txt)
- [`outputs/phase05_output.txt`](file:///outputs/phase05_output.txt)
- [`outputs/phase06_output.txt`](file:///outputs/phase06_output.txt)
- [`outputs/complete_run_output.txt`](file:///outputs/complete_run_output.txt)

---

## 🎓 Summary
All required phases and checklist tasks are completely implemented, documented, structured, and validated for error-free execution.
