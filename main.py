from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import faiss
import pickle
from sentence_transformers import SentenceTransformer
import requests

# Configuración
HF_TOKEN = os.getenv("HF_TOKEN")
MODEL_URL = "https://api-inference.huggingface.co/models/tiiuae/falcon-7b-instruct"

# Carga los fragmentos y el índice
with open("fragments.pkl", "rb") as f:
    fragments = pickle.load(f)

index = faiss.read_index("reglamento.index")
embedding_model = SentenceTransformer("BAAI/bge-small-en-v1.5")

app = Flask(__name__)
CORS(app,
     origins=["https://app.tecnoeducando.edu.pe"],
     methods=["GET", "POST", "OPTIONS"],
     allow_headers=["Content-Type", "X-Token"],
     supports_credentials=True)

TOKEN_PERMITIDO = "e398a7d3-dc9f-4ef9-bb29-07bff1672ef1"

@app.route("/consulta", methods=["POST", "OPTIONS"])
def consulta():
    if request.method == "OPTIONS":
        return jsonify({}), 200

    if request.headers.get("X-Token") != TOKEN_PERMITIDO:
        return jsonify({"error": "No autorizado"}), 403

    data = request.get_json()
    pregunta = data.get("pregunta", "")

    # Buscar contexto relevante
    pregunta_vec = embedding_model.encode([pregunta])
    D, I = index.search(pregunta_vec, k=5)
    contexto = "\n\n".join([fragments[i] for i in I[0]])

    prompt = f"""Responde con base únicamente en el siguiente reglamento. Sé claro y directo.

--- CONTEXTO ---
{contexto}
--- FIN ---

Pregunta: {pregunta}
"""

    try:
        response = requests.post(
            MODEL_URL,
            headers={
                "Authorization": f"Bearer {HF_TOKEN}",
                "Content-Type": "application/json"
            },
            json={
                "inputs": prompt,
                "parameters": {
                    "temperature": 0.7,
                    "max_new_tokens": 512
                }
            }
        )
        data = response.json()
        if isinstance(data, list):
            respuesta = data[0].get("generated_text", "No se pudo generar respuesta.")
            return jsonify({"respuesta": respuesta})
        else:
            return jsonify({"error": data}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
