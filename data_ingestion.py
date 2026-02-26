from pypdf import PdfReader 
from sentence_transformers import SentenceTransformer
import numpy

reader=PdfReader("sample.pdf")
text=""

for page in reader.pages:
    extracted=page.extract_text(extraction_mode="layout", layout_mode_space_vertically=False)

    if(extracted):
        text+=extracted

'''print(len(text))
print(text[:1000])'''

# ======== CHUNKING LOGIC ========

chunks=[]
chunk_size=300
overlap=40
step_s=chunk_size-overlap

words = text.split()

for i in range(0,len(words),step_s):
    sentence=' '.join(words[i:i+chunk_size])
    chunks.append(sentence)

'''print(len(chunks))

chunk0=chunks[0].split()
chunk1=chunks[1].split()

print(chunk0[-50:],"\n",chunk1[0:50])'''



# ======== EMBEDDING LOGIC ========

model = SentenceTransformer('BAAI/bge-small-en-v1.5')
embeddings=model.encode(chunks,convert_to_numpy=True,normalize_embeddings=True)

print("Encoding done \n",embeddings.shape)

import faiss
dims=embeddings.shape[1]
index=faiss.IndexFlatIP(dims)
index.add(embeddings.astype("float32"))
'''print(index.ntotal)'''


query=["What is self attention?",
        "What problem does the transformer solve?",
        "Explain positional encoding.",
        "What is convolution in CNNs?"
        ] 

query_encoding=model.encode(query,convert_to_numpy=True,normalize_embeddings=True).astype("float32")

print("Query shape:", query_encoding.shape)

D, I = index.search(query_encoding,2)
print(I.shape,D.shape)
# [query.size,k]

threshold=0.50

import ollama


for i in range(len(query)):
    best_score=D[i][0]
    if(best_score<threshold):
        print("No relevant information on the query.")
    else:
        for rank,idx in enumerate(I[i]):
            context="\n\n".join([chunks[idx] for idx in I[i]] )
            
            prompt=f"""
Answer the question ONLY using the context below.
If the answer is not in the context, say "Insufficient information."

Context:
{context}

Question:
{query[i]}

Answer:
"""

    response=ollama.chat(
        model="mistral",
        messages=[{"role":"user","content":prompt}]
    )
    print("Question: ",query[i])
    print("answer: ", response['message']['content'])
    print("="*80)

            


# ======== LLM INTEGRATION ========



