# Manual RAG System (From Scratch)

A fully manual implementation of a Retrieval-Augmented Generation (RAG) pipeline without using frameworks like LangChain.

## Features
- PDF ingestion
- Custom chunking logic
- Embedding generation
- FAISS vector indexing
- L2 and Inner Product similarity search
- Threshold-based relevance filtering
- LLM-based contextual response generation

## Architecture
1. Extract text from PDF
2. Chunk text into segments
3. Generate embeddings
4. Store in FAISS
5. Retrieve top-k relevant chunks
6. Apply similarity threshold
7. Generate answer via LLM

## Tech Stack
- Python
- FAISS
- NumPy
- PyPDF
- OpenAI / Ollama

## Why This Project?
Built to understand:
- Vector similarity mechanics
- Embedding normalization effects
- L2 vs Inner Product behavior
- Hallucination mitigation via thresholding