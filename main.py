from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import faiss
import pickle
import requests
from sentence_transformers import SentenceTransformer

# === CONFIGURACIÓN ===
DEESEEK_API_KEY = os.getenv("HF_TOKEN")  # ⚠️ Asegúrate de configurarlo en Render
API_URL = "https://api.deepseek.com/v1/chat/completions"
MODEL_NAME = "deepseek-chat"
TOKEN_PERMITIDO = "e398a7d3-dc9f-4ef9-bb29-07bff1672ef1"

# === INICIALIZACIÓN ===
embedding_model = SentenceTransformer("BAAI/bge-small-en-v1.5")

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
        return jsonify({}), 200

    if request.headers.get("X-Token") != TOKEN_PERMITIDO:
        return jsonify({"error": "No autorizado"}), 403

    data = request.get_json()
    pregunta = data.get("pregunta", "").strip()
    if not pregunta:
        return jsonify({"error": "Pregunta vacía"}), 400

    # === CARGAR BASE EMBEDDINGS ===
    try:
        with open("fragments.pkl", "rb") as f:
            fragments = pickle.load(f)
        index = faiss.read_index("reglamento.index")
    except:
        return jsonify({"error": "No hay documento cargado"}), 500

    # === BUSCAR CONTEXTO RELACIONADO ===
    pregunta_vec = embedding_model.encode([pregunta])
    D, I = index.search(pregunta_vec, k=5)
    contexto = "\n\n".join([fragments[i] for i in I[0]])

    # === PROMPT PARA DEEPSEEK ===
    prompt = f"""Eres un asistente que responde exclusivamente con base en el siguiente reglamento. 
Si la información no se encuentra en el reglamento, responde únicamente: "No se encuentra en el reglamento".

--- CONTEXTO ---
{contexto}
--- FIN ---

Pregunta: {pregunta}
Respuesta:"""

    # === CONSULTAR DEEPSEEK ===
    try:
        headers = {
            "Authorization": f"Bearer {DEEPSEEK_TOKEN}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": MODEL_NAME,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.3
        }

        res = requests.post(API_URL, headers=headers, json=payload)
        r = res.json()

        if "choices" in r:
            respuesta = r["choices"][0]["message"]["content"]
            return jsonify({"respuesta": respuesta})
        elif "error" in r:
            return jsonify({"error": r["error"]}), 500
        else:
            return jsonify({"error": "Respuesta inesperada del modelo"}), 500

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# === EJECUCIÓN LOCAL ===
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
