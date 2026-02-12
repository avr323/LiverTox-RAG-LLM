# livertoxrag/__init__.py

# Expose key functions for easy imports
from livertoxrag.data_loader import load_nxml_files
from livertoxrag.chunker import chunk_text
from livertoxrag.embedder import embed_chunks
from livertoxrag.vector_store import build_faiss_index, save_index, load_index
from livertoxrag.rag_llm import answer_question
from livertoxrag.evaluator import run_evaluation
from livertoxrag.pipeline import run_train, run_test

__version__ = "0.1.0"