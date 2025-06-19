from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle

MODEL_NAME = "all-MiniLM-L6-v2"
model = SentenceTransformer(MODEL_NAME)

def cargar_fragmentos(path, max_chars=500):
    with open(path, "r", encoding="utf-8") as f:
        texto = f.read()
    return [texto[i:i+max_chars] for i in range(0, len(texto), max_chars)]

def generar_embeddings(textos):
    vectores = model.encode(textos, show_progress_bar=True)
    return np.array(vectores)

def guardar_index(embeddings, textos, path="faiss_index.bin", pkl="embeddings.pkl"):
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    faiss.write_index(index, path)
    with open(pkl, "wb") as f:
        pickle.dump(textos, f)

def cargar_index(path="faiss_index.bin", pkl="embeddings.pkl"):
    index = faiss.read_index(path)
    with open(pkl, "rb") as f:
        textos = pickle.load(f)
    return index, textos

def buscar_similares(pregunta, index, textos, k=3):
    emb = model.encode([pregunta])
    distancias, indices = index.search(np.array(emb), k)
    return [textos[i] for i in indices[0]]
