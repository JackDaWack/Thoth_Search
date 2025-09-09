from fastapi import FastAPI
import os
import json
import numpy as np

DATA_PATH = os.path.join(os.path.dirname(__file__), 'data')
DOCS_FILE = os.path.join(DATA_PATH, 'documents.json')
VECS_FILE = os.path.join(DATA_PATH, 'vectors.npy')

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Welcome to Thoth Search API!"}

@app.get("/documents")
def get_documents():
    with open(DOCS_FILE, 'r') as f:
        documents = json.load(f)
    return {"documents": documents}