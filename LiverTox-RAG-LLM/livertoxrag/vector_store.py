# livertoxrag/vector_store.py

import faiss
import numpy as np
import os

def build_faiss_index(embeddings):
    """
    Builds a FAISS index from embeddings using inner product similarity.
    
    Args:
        embeddings (numpy.ndarray): Matrix of embedding vectors
        
    Returns:
        faiss.IndexFlatIP: FAISS index for similarity search
    """
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(np.array(embeddings).astype("float32"))
    return index

def save_index(index, path):
    """
    Saves a FAISS index to disk.
    
    Args:
        index (faiss.Index): FAISS index to save
        path (str): File path to save the index to
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(path), exist_ok=True)
    faiss.write_index(index, path)

def load_index(path):
    """
    Loads a FAISS index from disk.
    
    Args:
        path (str): File path to load the index from
        
    Returns:
        faiss.Index: Loaded FAISS index
        
    Raises:
        FileNotFoundError: If the index file does not exist
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"FAISS index not found at {path}")
    return faiss.read_index(path)