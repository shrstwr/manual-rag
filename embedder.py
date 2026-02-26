from sentence_transformers import SentenceTransformer
import faiss
from config import model_name, INDEX_PATH
import numpy as np
import os

class Embedder:
    def __init__(self):
        self.model=SentenceTransformer(model_name)
        self.index= None
    
    def encode_chunks(self,chunks):
        embeddings=self.model.encode(chunks,convert_to_numpy=True,normalize_embeddings=True)
        return embeddings.astype("float32")
    
    def indexer(self,embeddings):
        dims=embeddings.shape[1]
        self.index=faiss.IndexFlatIP(dims)
        self.index.add(embeddings)

    def save_indexes(self):
        faiss.write_index(self.index, INDEX_PATH)

    def load_index(self):
        if(os.path.exists(INDEX_PATH)):
            self.index=faiss.read_index(INDEX_PATH)
            return True
        return False

    def encode_query(self,query):
        return self.model.encode([f"query:{query}"],convert_to_numpy=True,normalize_embeddings=True).astype("float32")
    


        