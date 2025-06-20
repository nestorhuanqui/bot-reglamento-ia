from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import faiss
import pickle
from sentence_transformers import SentenceTransformer
import requests

# Configuración
DEESEEK_API_KEY = os.getenv("DEESEEK_API_KEY")  # ⚠️ Asegúrate de poner esto en Render
MODEL_NAME = "deepseek-ai/deepseek-coder-6.7b-instruct"  # o el modelo que uses

# Carga los fragmentos y el índice
with open("fragments.pkl", "rb") as f:
    fragments = pickle.load(f)

index = faiss.read_index("reglamento.index")

# Modelo de embeddings
embedding_model = SentenceTransformer("BAAI/bge-small-en-v1.5")

# Flask
app = Flask(__name__)
CORS(app,
     resources={r"/consulta": {"origins": "https://app.tecnoeducando.edu.pe"}},
     methods=["GET", "POST", "OPTIONS"],
     allow_headers=["Content-Type", "X-Token"],
     supports_credentials=True)

# Token de seguridad
TOKEN_PERMITIDO = "e398a7d3-dc9f-4ef9-bb29-07bff1672ef1"

@app.route("/consulta", methods=["POST", "OPTIONS"])
def consulta():
    if request.method == "OPTIONS":
        return jsonify({}), 200

    if request.headers.get("X-Token") != TOKEN_PERMITIDO:
        return jsonify({"error": "No autorizado"}), 403

    data = request.get_json()
    pregunta = data.get("pregunta", "")

    # Embedding de la pregunta
    pregunta_vec = embedding_model.encode([pregunta])
    D, I = index.search(pregunta_vec, k=5)
    contexto = "\n\n".join([fragments[i] for i in I[0]])

    prompt = f"""
Responde con base en el siguiente reglamento. No inventes información.

--- CONTEXTO ---
{contexto}
--- FIN ---

Pregunta: {pregunta}
"""

    try:
        response = requests.post(
            "https://api.deepseek.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {DEESEEK_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": MODEL_NAME,
                "messages": [
                    {"role": "user", "content": prompt}
                ]
            }
        )

        data = response.json()
        if "choices" in data:
            respuesta = data["choices"][0]["message"]["content"]
            return jsonify({"respuesta": respuesta})
        else:
            return jsonify({"error": data}), 500

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
