# Manual Retrieval-Augmented Generation (RAG) System

A from-scratch implementation of a Retrieval-Augmented Generation (RAG) pipeline built **without LangChain or high-level orchestration frameworks**, focusing on understanding vector similarity, embedding behavior, and retrieval mechanics at a systems level.

---

## Overview

This project implements a complete RAG workflow:

- PDF ingestion  
- Custom chunking  
- Embedding generation  
- FAISS-based vector indexing  
- Inner Product similarity search  
- Threshold-based relevance filtering  
- Local LLM-based answer generation  

The goal was to deeply understand how retrieval systems operate internally rather than relying on abstraction layers.

---

## Key Features

- Manual PDF parsing using PyPDF  
- Custom fixed-size chunking strategy  
- Embedding generation pipeline  
- FAISS vector store using **Inner Product search**  
- Embedding normalization to approximate cosine similarity  
- Score thresholding to reduce hallucinated responses  
- Local LLM integration via Ollama  

---

## Architecture

1. Extract text from PDF  
2. Apply deterministic chunking logic  
3. Generate embeddings for each chunk  
4. Normalize embeddings  
5. Store vectors in FAISS index  
6. Perform Inner Product similarity search  
7. Apply similarity threshold to filter irrelevant chunks  
8. Pass retrieved context to local LLM for response generation (Mistral)

---

## Technical Design Decisions

### Inner Product + Normalization

Embeddings are normalized before indexing.  
This allows Inner Product search to behave like cosine similarity while maintaining FAISS efficiency.

This ensures retrieval is driven by semantic alignment rather than vector magnitude.

### Threshold-Based Retrieval

Instead of blindly returning top-k results, a similarity threshold is applied:

- Prevents low-relevance context injection  
- Reduces hallucination risk  
- Avoids generating answers when supporting evidence is weak  

### Framework-Free Implementation

The pipeline was intentionally implemented without LangChain to gain deeper understanding of:

- Vector similarity mechanics  
- Embedding normalization effects  
- FAISS indexing behavior  
- Retrieval-to-generation coupling  
- Error propagation in RAG systems  

---

## Tech Stack

- Python  
- FAISS  
- NumPy  
- PyPDF  
- Ollama (Local LLM Runtime)  

---

## What This Project Demonstrates

- Practical understanding of vector similarity search  
- Manual FAISS index construction  
- Cosine-style retrieval via embedding normalization  
- Controlled retrieval filtering strategies  
- End-to-end RAG pipeline implementation  
- Local LLM integration without external API dependency  
