# LiverTox-RAG-LLM
RAG-LLM implementation for Drug induced liver injury based on LiverTox data

## Overview

A Retrieval-Augmented Generation (RAG) system for drug-induced liver injury (DILI) assessment using the NIH LiverTox database. This system achieves hallucination-free responses by grounding large language models in authoritative hepatotoxicity data.


## Publication

This code accompanies the following publication:
> Rao A, Cholankeril G, Flores A, Sood G, White TE, Kanwal F, El-Serag HB. Development of retrieval-augmented generation–based large language model for drug-induced liver injury using LiverTox data. *Hepatology Communications*. 2026;10(3):e00895. doi: [10.1097/HC9.0000000000000895](https://doi.org/10.1097/HC9.0000000000000895)

## Installation

```bash
git clone https://github.com/[your-username]/LiverTox-RAG-LLM.git
cd LiverTox-RAG-LLM
pip install -r requirements.txt
```

## Quick Start

### 1. Process LiverTox Database

```python
python main.py --train
```

This will:
- Parse LiverTox drug monographs
- Generate semantic chunks
- Create BioBERT embeddings
- Build FAISS index

### 2. Query the System

```python
python main.py --test
```

Example queries:
- "What is the hepatotoxicity pattern of acetaminophen?"
- "What are the risk factors for isoniazid hepatotoxicity?"
- "What is the mechanism of liver injury from statins?"

## Project Structure

```
LiverTox-RAG-LLM/
├── livertoxrag/
│   ├── pipeline.py           # Main orchestration
│   ├── rag_llm.py            # RAG implementation
│   ├── chunker.py            # Document processing
│   ├── embedder.py           # BioBERT embeddings
│   ├── vector_store.py       # FAISS index management
│   ├── data_loader.py        # NXML file parsing
│   └── evaluator.py          # Evaluation metrics
├── data/
│   └── raw_nxml/             # LiverTox source files
├── embeddings/               # Generated embeddings
├── evaluation/               # Evaluation results
│   └── questions.csv         # Evaluation questions
├── examples/
│   └── query_example.py      # Usage example
└── main.py                   # Entry point
```

## System Architecture

```
LiverTox Data -> Parsing -> Chunking -> Embedding -> FAISS Index
                                                          |
User Query -> Query Embedding -> Semantic Search -> Retrieved Chunks
                                                          |
                                              Context + LLM -> Response
```

## Configuration

### Section Prioritization

The system prioritizes clinically relevant sections:

```python
SECTION_PRIORITY = {
    "hepatotoxicity": 0,      # Highest priority
    "mechanism_of_injury": 0,
    "outcome_and_management": 0,
    "introduction": 1,
    "background": 1,
    "case_report": 2,
}
```

### Model Configuration

Supported models:
- OpenAI: GPT-4, GPT-3.5
- Anthropic: Claude 3
- Google: Gemini Pro
- Local: Llama 3, Mistral (via llama_cpp)

Set API keys as environment variables:
```bash
export OPENAI_API_KEY="your-key"
export ANTHROPIC_API_KEY="your-key"
```

## Data Preparation

### LiverTox Database

1. Download LiverTox NXML files
2. Place in `data/raw_nxml/`

## Evaluation

Run evaluation:

```bash
python main.py --test
```

This evaluates the system on clinical DILI questions.

## Citation

If you use this code, please cite:

```bibtex
@article{rao2026livertox,
  title={Development of retrieval-augmented generation–based large language model for drug-induced liver injury using LiverTox data},
  author={Rao, Ashwin and Cholankeril, George and Flores, Avegail and Sood, Gagan and White, Terrence E and Kanwal, Fasiha and El-Serag, Hashem B},
  journal={Hepatology Communications},
  volume={10},
  number={3},
  pages={e00895},
  year={2026},
  publisher={Wolters Kluwer},
  doi={10.1097/HC9.0000000000000895}
}
```

## License

MIT License - See LICENSE file for details

## Disclaimer

This system is for research purposes only and should not be used for clinical decision-making without expert consultation.
