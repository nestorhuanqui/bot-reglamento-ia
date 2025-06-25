from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import faiss
import pickle
import requests
from sentence_transformers import SentenceTransformer

# === CONFIGURACIÓN GENERAL ===
API_URL = "https://api.deepseek.com/v1/chat/completions"
DEEPSEEK_TOKEN = "Bearer sk-baff1039951046d0be1ebc634657b59d"  # Reemplázalo por tu token real
MODEL_NAME = "deepseek-chat"
TOKEN_PERMITIDO = "e398a7d3-dc9f-4ef9-bb29-07bff1672ef1"

# === MODELO DE EMBEDDINGS ===
embedding_model = SentenceTransformer("BAAI/bge-small-en-v1.5")

# === FLASK ===
app = Flask(__name__)
CORS(app,
     resources={r"/consulta": {"origins": ["https://app.tecnoeducando.edu.pe"]}},
     supports_credentials=True,
     methods=["GET", "POST", "OPTIONS"],
     allow_headers=["Content-Type", "X-Token"])

# === RUTA /consulta ===
@app.route("/consulta", methods=["POST", "OPTIONS"])
def consulta():
    if request.method == "OPTIONS":
        response = jsonify({})
        response.headers.add("Access-Control-Allow-Origin", "https://app.tecnoeducando.edu.pe")
        response.headers.add("Access-Control-Allow-Headers", "Content-Type, X-Token")
        response.headers.add("Access-Control-Allow-Methods", "POST, OPTIONS")
        return response, 200

    if request.headers.get("X-Token") != TOKEN_PERMITIDO:
        return jsonify({"error": "No autorizado"}), 403

    data = request.get_json()
    pregunta = data.get("pregunta", "").strip()
    if not pregunta:
        return jsonify({"error": "Pregunta vacía"}), 400

    # Cargar index y fragmentos
    try:
        with open("fragments.pkl", "rb") as f:
            fragments = pickle.load(f)
        index = faiss.read_index("reglamento.index")
    except:
        return jsonify({"error": "No hay documento cargado"}), 500

    # Embedding + búsqueda
    pregunta_vec = embedding_model.encode([pregunta])
    D, I = index.search(pregunta_vec, k=5)
    contexto = "\n\n".join([fragments[i] for i in I[0]])

    # Prompt para DeepSeek
    prompt = f"""Eres un asistente que responde exclusivamente con base en el siguiente reglamento. 
Si la información no se encuentra en el reglamento, responde únicamente: "No se encuentra en el reglamento".

--- CONTEXTO ---
{contexto}
--- FIN ---

Pregunta: {pregunta}
Respuesta:"""

    # Llamada al modelo DeepSeek
    try:
        res = requests.post(
            API_URL,
            headers={
                "Authorization": DEEPSEEK_TOKEN,
                "Content-Type": "application/json"
            },
            json={
                "model": MODEL_NAME,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.3
            }
        )
        r = res.json()
        texto = r["choices"][0]["message"]["content"]
        return jsonify({"respuesta": texto})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# === EJECUCIÓN LOCAL ===
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
