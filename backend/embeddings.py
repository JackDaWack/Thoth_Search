import os
import json
import numpy as np
from sentence_transformers import SentenceTransformer as st

DATA_PATH = os.path.join(os.path.dirname(__file__), 'data')
DOCS_FILE = os.path.join(DATA_PATH, 'documents.json')
VECS_FILE = os.path.join(DATA_PATH, 'vectors.npy')

#Data set to add later.
doc_section = []

def generate_embeddings():
    os.makedirs(DATA_PATH, exist_ok=True)

    # Save metadata
    with open(DOCS_FILE, 'w') as f:
        json.dump(sample_docs, f, indent=2)

    # Generate embeddings
    texts = [doc["text"] for doc in sample_docs]
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(texts, convert_to_numpy=True)

    # Save embeddings
    np.save(VECS_FILE, embeddings)

if __name__ == "__main__":
    generate_embeddings()
