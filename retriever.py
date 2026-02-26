from config import top_k,threshold

def retrieve(query,engine,chunks):
    query_encoding=engine.encode_query(query)
    D, I = engine.index.search(query_encoding,top_k)
    best_score=D[0][0]
    if(best_score<threshold):
        return []
    retrieved_chunks=[chunks[idx] for idx in I[0]]
    return retrieved_chunks
