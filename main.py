from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import faiss
import pickle
from sentence_transformers import SentenceTransformer
import requests

# === CONFIGURACIÓN ===
DEESEEK_API_KEY = os.getenv("DEESEEK_API_KEY")
DEESEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"
MODEL_NAME = "deepseek-chat"

# === CARGA FRAGMENTOS Y FAISS ===
with open("fragments.pkl", "rb") as f:
    fragments = pickle.load(f)
index = faiss.read_index("reglamento.index")
embedding_model = SentenceTransformer("BAAI/bge-small-en-v1.5")

# === APP FLASK ===
app = Flask(__name__)
CORS(app,
     resources={r"/consulta": {"origins": ["https://app.tecnoeducando.edu.pe"]}},
     supports_credentials=True,
     methods=["GET", "POST", "OPTIONS"],
     allow_headers=["Content-Type", "X-Token"])

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

    # Prompt optimizado
    prompt = f"""Eres un asistente que responde exclusivamente con base en el siguiente reglamento.
Si la información no está en el reglamento, responde: "No se encuentra en el reglamento".

--- CONTEXTO ---
{contexto}
--- FIN ---

Pregunta: {pregunta}
Respuesta:"""

    try:
        response = requests.post(
            DEESEEK_API_URL,
            headers={
                "Authorization": f"Bearer {DEESEEK_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": MODEL_NAME,
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.3,
                "max_tokens": 300
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
