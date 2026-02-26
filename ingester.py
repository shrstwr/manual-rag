from pypdf import PdfReader
from config import chunk_size,overlap,CHUNKS_PATH
import numpy as np

def pdf_loader(path:str)->str:
    reader=PdfReader("sample.pdf")
    text=" "
    for page in reader.pages:
        extracted=page.extract_text(extraction_mode="layout", layout_mode_space_vertically=False)

        if(extracted):
            text+=extracted
    return text

def chunker(text:str):
    words=text.split()
    chunks=[]
    step_s=chunk_size-overlap

    for i in range(0,len(words),step_s):
        sentence=' '.join(words[i:i+chunk_size-overlap])
        chunks.append(sentence)

    np.save(CHUNKS_PATH,np.array(chunks,dtype=object))    
    return chunks
