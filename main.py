from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import faiss
import pickle
import requests
from sentence_transformers import SentenceTransformer

# === CONFIGURACIÓN ===
DEESEEK_TOKEN = os.getenv("DEESEEK_API_KEY")
API_URL = "https://api.deepseek.com/v1/chat/completions"
MODEL_NAME = "deepseek-chat"
TOKEN_PERMITIDO = "e398a7d3-dc9f-4ef9-bb29-07bff1672ef1"

# === EMBEDDINGS ===
embedding_model = SentenceTransformer("BAAI/bge-small-en-v1.5")

# === FLASK APP ===
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

    pregunta = request.get_json().get("pregunta", "").strip()
    if not pregunta:
        return jsonify({"error": "Pregunta vacía"}), 400

    try:
        with open("fragments.pkl", "rb") as f:
            fragments = pickle.load(f)
        index = faiss.read_index("reglamento.index")
    except:
        return jsonify({"error": "No hay documento cargado"}), 500

    pregunta_vec = embedding_model.encode([pregunta])
    D, I = index.search(pregunta_vec, k=5)
    contexto = "\n\n".join([fragments[i] for i in I[0]])

    prompt = f"""Eres un asistente amable y profesional que responde exclusivamente con base en el siguiente reglamento.
Responde con un tono cercano, claro y cordial, como si hablaras con una persona interesada. 
Si la información no se encuentra en el reglamento, responde únicamente: "No se encuentra en el reglamento".

--- CONTEXTO ---
{contexto}
--- FIN ---

Pregunta: {pregunta}
Respuesta:"""

    try:
        res = requests.post(API_URL,
                            headers={
                                "Authorization": f"Bearer {DEESEEK_TOKEN}",
                                "Content-Type": "application/json"
                            },
                            json={
                                "model": MODEL_NAME,
                                "messages": [{"role": "user", "content": prompt}],
                                "temperature": 0.3
                            })
        r = res.json()
        texto = r["choices"][0]["message"]["content"]
        return jsonify({"respuesta": texto})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# === EJECUCIÓN LOCAL ===
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
