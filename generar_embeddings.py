import requests
import json

API_KEY = "TU_API_KEY_DEEPSEEK"  # <-- Reemplázala por tu key real
ARCHIVO = "reglamento_demo.txt"
URL = "https://api.deepseek.com/v1/embeddings"


def dividir_texto(texto, max_longitud=300):
    partes = []
    oraciones = texto.split('.')
    buffer = ''
    for o in oraciones:
        if len(buffer) + len(o) < max_longitud:
            buffer += o + '.'
        else:
            partes.append(buffer.strip())
            buffer = o + '.'
    if buffer:
        partes.append(buffer.strip())
    return partes


def generar_embeddings(textos):
    resultados = []
    for texto in textos:
        payload = {
            "model": "deepseek-embedding-001",
            "input": texto
        }
        headers = {
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json"
        }
        r = requests.post(URL, headers=headers, json=payload)
        data = r.json()
        vector = data["data"][0]["embedding"]
        resultados.append({"texto": texto, "vector": vector})
    return resultados


if __name__ == "__main__":
    with open(ARCHIVO, "r", encoding="utf-8") as f:
        texto = f.read()

    partes = dividir_texto(texto)
    embeddings = generar_embeddings(partes)

    with open("vectores.json", "w", encoding="utf-8") as f:
        json.dump(embeddings, f, ensure_ascii=False, indent=2)

    print("✅ Embeddings generados y guardados en vectores.json")
