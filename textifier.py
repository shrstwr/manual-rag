import ollama

def textify(s_chunks,query):
    context="\n\n".join(s_chunks)
    prompt=f"""
        Answer the question ONLY using the context below.
        If the answer is not in the context, say "Insufficient information."

        Context:
        {s_chunks}

        Question:
        {query}

        Answer:
    """
    response=ollama.chat(
        model="mistral",
        messages=[{"role":"user","content":prompt}]
        
    )
    return response['message']['content']
