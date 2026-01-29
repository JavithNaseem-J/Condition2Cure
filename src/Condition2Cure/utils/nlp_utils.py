import re
import numpy as np
from sentence_transformers import SentenceTransformer


print("Loading Sentence Transformer model...")
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
print("Model loaded!")


def clean_text(text: str) -> str:

    if not text or not isinstance(text, str):
        return ""
    
    # Step 1: Lowercase
    text = text.lower()
    
    # Step 2: Remove HTML tags like <br>, <p>, etc.
    text = re.sub(r'<.*?>', '', text)
    
    # Step 3: Keep only letters, numbers, and basic punctuation
    text = re.sub(r'[^a-zA-Z0-9\s\.\,\!\?]', '', text)
    
    # Step 4: Remove extra whitespace
    text = ' '.join(text.split())
    
    return text.strip()


def get_embedding(text: str) -> np.ndarray:

    cleaned = clean_text(text)
    
    # Generate embedding using BERT
    embedding = embedding_model.encode(
        cleaned,
        convert_to_numpy=True,
        show_progress_bar=False
    )
    
    return embedding


def get_embeddings_batch(texts: list) -> np.ndarray:

    # Clean all texts
    cleaned_texts = [clean_text(t) for t in texts]
    
    # Generate embeddings in batch
    embeddings = embedding_model.encode(
        cleaned_texts,
        convert_to_numpy=True,
        show_progress_bar=True,
        batch_size=32
    )
    
    return embeddings
