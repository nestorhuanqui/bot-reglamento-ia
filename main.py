from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import faiss
import pickle
import requests
from sentence_transformers import SentenceTransformer

# === CONFIGURACIÓN ===
DEEPSEEK_API_KEY = os.getenv("DEESEEK_API_KEY")
API_URL = "https://api.deepseek.com/v1/chat/completions"
MODEL_NAME = "deepseek-chat"
TOKEN_PERMITIDO = "e398a7d3-dc9f-4ef9-bb29-07bff1672ef1"

# === FLASK SETUP ===
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": ["https://app.tecnoeducando.edu.pe"]}})

# Middleware global para forzar headers CORS en TODAS las respuestas
@app.after_request
def agregar_cors(response):
    response.headers.add("Access-Control-Allow-Origin", "https://app.tecnoeducando.edu.pe")
    response.headers.add("Access-Control-Allow-Headers", "Content-Type, X-Token")
    response.headers.add("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
    return response

# === CARGAR MODELO DE EMBEDDINGS ===
embedding_model = SentenceTransformer("BAAI/bge-small-en-v1.5")

# === RUTA /consulta ===
@app.route("/consulta", methods=["POST", "OPTIONS"])
def consulta():
    if request.method == "OPTIONS":
        return jsonify({}), 200

    if request.headers.get("X-Token") != TOKEN_PERMITIDO:
        return jsonify({"error": "No autorizado"}), 403

    pregunta = request.get_json().get("pregunta", "").strip()
    if not pregunta:
        return jsonify({"error": "Pregunta vacía"}), 400

    # Cargar vector store
    try:
        with open("fragments.pkl", "rb") as f:
            fragments = pickle.load(f)
        index = faiss.read_index("reglamento.index")
    except:
        return jsonify({"error": "No hay documento cargado"}), 500

    # Vectorizar y buscar contexto
    pregunta_vec = embedding_model.encode([pregunta])
    D, I = index.search(pregunta_vec, k=5)
    contexto = "\n\n".join([fragments[i] for i in I[0]])

    # PROMPT con enfoque conversacional natural
    prompt = f"""
Actúa como un asistente amable y preciso. Usa solamente la información del siguiente reglamento para responder.
Si no encuentras la información en el texto, responde: "No se encuentra en el reglamento".

--- CONTEXTO ---
{contexto}
--- FIN ---

Pregunta: {pregunta}
Respuesta:"""

    try:
        res = requests.post(API_URL,
            headers={
                "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": MODEL_NAME,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.6
            })

        r = res.json()
        texto = r["choices"][0]["message"]["content"]
        return jsonify({"respuesta": texto})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# === MAIN ===
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
