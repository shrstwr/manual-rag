from ingester import pdf_loader, chunker
from embedder import Embedder
import config 
import os
import numpy as np
from retriever import retrieve
from textifier import textify
import faiss

def main():
    
    engine=Embedder()
    isIndex=engine.load_index()
    if(isIndex and os.path.exists(config.CHUNKS_PATH)):
        chunks=np.load(config.CHUNKS_PATH,allow_pickle=True)
        print("Loading saved indexes.....")
    else:
        text=pdf_loader(config.pdf_path)
        chunks=chunker(text)

        embeddings=engine.encode_chunks(chunks)
        engine.indexer(embeddings)

        engine.save_indexes()
        np.save(config.CHUNKS_PATH,np.array(chunks,dtype=object))

    while(True):

        query=input("Enter your query: ")
        if(query.lower()=="exit"):
            break

        results=retrieve(query,engine,chunks)
        if not results:
            print("No relevant information found.")
        else:
            print(f"Mistral is working on your thoughtful query: {query}")
            answer=textify(results,query)
            print(answer,"\n\n")

if __name__ == "__main__":
    main()
                
        

        

