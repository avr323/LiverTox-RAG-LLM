# livertoxrag/embedder.py

from sentence_transformers import SentenceTransformer

def embed_chunks(chunks):
    """
    Embeds chunks using BioBERT-style clinical model.

    Args:
        chunks (list): List of chunk dictionaries containing 'text', 'doc_id', 'chunk_id', and 'section' keys

    Returns:
        tuple: (embeddings, metadata)
            - embeddings: numpy array of embedded vectors
            - metadata: dictionary mapping indices to chunk metadata
    """
    model = SentenceTransformer('pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb')

    texts = [chunk["text"] for chunk in chunks]
    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True
    )

    # Create metadata dictionary with string keys for consistent access
    metadata = {
        str(i): {
            "doc_id": chunk["doc_id"],
            "chunk_id": chunk["chunk_id"],
            "section": chunk["section"],
            "text": chunk["text"]
        } for i, chunk in enumerate(chunks)
    }

    return embeddings, metadata