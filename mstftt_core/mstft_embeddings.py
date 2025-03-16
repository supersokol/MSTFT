# mstftt_core/mstft_embeddings.py

import numpy as np
import hashlib

def get_embedding(text, embedding_size=32):
    """
    Generate a deterministic embedding for a given text using SHA-256.
    The SHA-256 hash digest (32 bytes) is converted into a fixed-length vector.
    
    Args:
        text (str): The input text.
        embedding_size (int): Desired length of the embedding vector (default: 32).
    
    Returns:
        numpy.ndarray: A normalized embedding vector of length `embedding_size`.
    """
    # Compute SHA-256 hash of the text
    hash_digest = hashlib.sha256(text.encode('utf-8')).digest()
    # Convert the 32-byte digest to a numpy array of uint8 and then to float32
    vec = np.frombuffer(hash_digest, dtype=np.uint8).astype(np.float32)
    # Adjust vector length if needed
    if embedding_size < len(vec):
        vec = vec[:embedding_size]
    elif embedding_size > len(vec):
        vec = np.pad(vec, (0, embedding_size - len(vec)), 'constant')
    # Normalize the vector to unit length
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    return vec

def cosine_similarity(a, b):
    """
    Compute the cosine similarity between two vectors.
    
    Args:
        a (array-like): First vector.
        b (array-like): Second vector.
        
    Returns:
        float: Cosine similarity value.
    """
    a = np.array(a)
    b = np.array(b)
    if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
        return 0.0
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def calculate_similarity(query_emb, indexed_sections):
    """
    Calculate cosine similarity between a query embedding and each embedding in an indexed sections dictionary.
    
    Args:
        query_emb (numpy.ndarray): Embedding vector for the query.
        indexed_sections (dict): Dictionary with keys as identifiers and values as embedding vectors.
        
    Returns:
        list: Sorted list of tuples (key, similarity) in descending order of similarity.
    """
    similarities = []
    for key, emb in indexed_sections.items():
        sim = cosine_similarity(query_emb, emb)
        similarities.append((key, sim))
    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities
