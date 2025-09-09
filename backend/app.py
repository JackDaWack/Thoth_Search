from fastapi import FastAPI
import os
import json
import numpy as np
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer as st

DATA_PATH = os.path.join(os.path.dirname(__file__), 'data')
DOCS_FILE = os.path.join(DATA_PATH, 'documents.json')
VECS_FILE = os.path.join(DATA_PATH, 'vectors.npy')

app = FastAPI()
model = st('all-MiniLM-L6-v2')

class Search_Query(BaseModel):
    query_text: str

@app.get("/")
def read_root():
    return {"message": "Welcome to Thoth Search API!"}

@app.get("/documents")
def get_documents():
    try:
        with open(DOCS_FILE, 'r') as f:
            documents = json.load(f)
    except FileNotFoundError:
        return {"error": "Documents file not found."}
    except Exception as e:
        return {"error": str(e)}
    return {"documents": documents}

@app.post("/search")
def search(query: Search_Query):
    try:
        with open(DOCS_FILE, 'r') as f:
            documents = json.load(f)
        vectors = np.load(VECS_FILE)
    except FileNotFoundError:
        return {"error": "Data files not found. Please generate embeddings first."}
    except Exception as e:
        return {"error": str(e)}

    if len(vectors) != len(documents):
        return {"error": "Data inconsistency: number of documents and vectors do not match."}

    try:
        query_vec = model.encode([query.query_text], convert_to_numpy=True)[0]
    except Exception as e:
        return {"error": f"Error generating query embedding: {str(e)}"}

    similarities = np.dot(vectors, query_vec) / (np.linalg.norm(vectors, axis=1) * np.linalg.norm(query_vec))
    top_indices = np.argsort(similarities)[-3:][::-1]
    results = [documents[i] for i in top_indices]

    return {"results": results}