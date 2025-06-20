from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import faiss
import pickle
from sentence_transformers import SentenceTransformer
import requests

# === CONFIGURACIÓN ===
API_URL = "https://api-inference.huggingface.co/models/tiiuae/falcon-7b-instruct"
HF_TOKEN = os.getenv("HF_TOKEN")  # ⚠️ Debes ponerlo en las Environment Variables en Render

# === CARGAR FRAGMENTOS E ÍNDICE FAISS ===
with open("fragments.pkl", "rb") as f:
    fragments = pickle.load(f)

index = faiss.read_index("reglamento.index")

embedding_model = SentenceTransformer("BAAI/bge-small-en-v1.5")

# === FLASK + CORS ===
app = Flask(__name__)
CORS(app,
     origins=["https://app.tecnoeducando.edu.pe"],
     methods=["GET", "POST", "OPTIONS"],
     allow_headers=["Content-Type", "X-Token"],
     supports_credentials=True)

# === TOKEN DE AUTORIZACIÓN PARA EL FRONTEND ===
TOKEN_PERMITIDO = "e398a7d3-dc9f-4ef9-bb29-07bff1672ef1"

# === RUTA PARA CONSULTA DE EMBEDDINGS ===
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

    # Vector de la pregunta
    pregunta_vec = embedding_model.encode([pregunta])
    D, I = index.search(pregunta_vec, k=5)
    contexto = "\n\n".join([fragments[i] for i in I[0]])

    # Crear prompt
    prompt = f"""
Responde con base en el siguiente reglamento. No inventes información.

--- CONTEXTO ---
{contexto}
--- FIN ---

Pregunta: {pregunta}
"""

    try:
        response = requests.post(
            API_URL,
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

        print("STATUS:", response.status_code)
        print("TEXT:", response.text)

        if response.status_code == 200:
            data = response.json()
            if isinstance(data, list):
                respuesta = data[0].get("generated_text", "Sin respuesta generada.")
                return jsonify({"respuesta": respuesta})
            else:
                return jsonify({"error": "Error inesperado", "detalle": data}), 500
        else:
            return jsonify({"error": f"Respuesta HTTP {response.status_code}", "detalle": response.text}), 500

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# === RUTA PARA VERIFICAR QUE EL BACKEND FUNCIONA ===
@app.route("/ping", methods=["GET"])
def ping():
    return jsonify({"mensaje": "Backend activo en bot.tecnoeducando.edu.pe"})

# === INICIO DEL SERVIDOR FLASK ===
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
