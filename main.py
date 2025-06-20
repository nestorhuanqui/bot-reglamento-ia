from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import faiss
import pickle
from sentence_transformers import SentenceTransformer
import requests

# === CONFIGURACIÓN ===
HUGGINGFACE_TOKEN = os.getenv("HF_TOKEN")  # Variable de entorno en Render
FALCON_API_URL = "https://api-inference.huggingface.co/models/tiiuae/falcon-7b-instruct"
MODEL_NAME = "tiiuae/falcon-7b-instruct"

# === CARGA EMBEDDINGS Y FRAGMENTOS ===
with open("fragments.pkl", "rb") as f:
    fragments = pickle.load(f)
index = faiss.read_index("reglamento.index")

# Modelo de embeddings
embedding_model = SentenceTransformer("BAAI/bge-small-en-v1.5")

# === FLASK SETUP ===
app = Flask(__name__)
CORS(app,
     resources={r"/consulta": {"origins": "https://app.tecnoeducando.edu.pe"}},
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
    pregunta = data.get("pregunta", "").strip()
    if not pregunta:
        return jsonify({"error": "Pregunta vacía"}), 400

    # Buscar contexto relevante
    pregunta_vec = embedding_model.encode([pregunta])
    D, I = index.search(pregunta_vec, k=5)
    contexto = "\n\n".join([fragments[i] for i in I[0]])

    # Prompt optimizado para Falcon
    prompt = f"""Responde basándote estrictamente en el siguiente reglamento. Si no está en el reglamento, responde "No se encuentra en el reglamento".

--- REGLAMENTO ---
{contexto}
--- FIN ---

Pregunta: {pregunta}
Respuesta:"""

    try:
        response = requests.post(
            FALCON_API_URL,
            headers={
                "Authorization": f"Bearer {HUGGINGFACE_TOKEN}",
                "Content-Type": "application/json"
            },
            json={
                "inputs": prompt,
                "parameters": {
                    "max_new_tokens": 300,
                    "temperature": 0.3,
                    "do_sample": False
                }
            }
        )

        result = response.json()
        if isinstance(result, list) and "generated_text" in result[0]:
            return jsonify({"respuesta": result[0]["generated_text"].split("Respuesta:")[-1].strip()})
        else:
            return jsonify({"error": result}), 500

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
