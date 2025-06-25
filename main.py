from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import faiss
import pickle
import requests
from sentence_transformers import SentenceTransformer

# === CONFIGURACIÓN GENERAL ===
API_URL = "https://api.deepseek.com/v1/chat/completions"
MODEL_NAME = "deepseek-chat"
TOKEN_PERMITIDO = "e398a7d3-dc9f-4ef9-bb29-07bff1672ef1"
DEEPSEEK_TOKEN = os.getenv("DEEPSEEK_TOKEN")  # Tu token de API DeepSeek (colocado como variable de entorno en Render)

# === FLASK SETUP ===
app = Flask(__name__)
CORS(app,
     resources={
         r"/consulta": {"origins": ["https://app.tecnoeducando.edu.pe"]}
     },
     supports_credentials=True,
     methods=["GET", "POST", "OPTIONS"],
     allow_headers=["Content-Type", "X-Token"]
)

# === MODELO DE EMBEDDINGS ===
embedding_model = SentenceTransformer("BAAI/bge-small-en-v1.5")

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
        response = jsonify({"error": "No autorizado"})
        response.headers.add("Access-Control-Allow-Origin", "https://app.tecnoeducando.edu.pe")
        return response, 403

    data = request.get_json()
    pregunta = data.get("pregunta", "").strip()
    if not pregunta:
        response = jsonify({"error": "Pregunta vacía"})
        response.headers.add("Access-Control-Allow-Origin", "https://app.tecnoeducando.edu.pe")
        return response, 400

    try:
        with open("fragments.pkl", "rb") as f:
            fragments = pickle.load(f)
        index = faiss.read_index("reglamento.index")
    except Exception as e:
        response = jsonify({"error": "No hay documento cargado"})
        response.headers.add("Access-Control-Allow-Origin", "https://app.tecnoeducando.edu.pe")
        return response, 500

    pregunta_vec = embedding_model.encode([pregunta])
    D, I = index.search(pregunta_vec, k=5)
    contexto = "\n\n".join([fragments[i] for i in I[0]])

    prompt = f"""Eres un asistente que responde exclusivamente con base en el siguiente reglamento. 
Si la información no se encuentra en el reglamento, responde únicamente: "No se encuentra en el reglamento".

--- CONTEXTO ---
{contexto}
--- FIN ---

Pregunta: {pregunta}
Respuesta:"""

    try:
        res = requests.post(API_URL,
                            headers={
                                "Authorization": f"Bearer {DEEPSEEK_TOKEN}",
                                "Content-Type": "application/json"
                            },
                            json={
                                "model": MODEL_NAME,
                                "messages": [{"role": "user", "content": prompt}],
                                "temperature": 0.3
                            })
        r = res.json()
        texto = r["choices"][0]["message"]["content"]

        response = jsonify({"respuesta": texto})
        response.headers.add("Access-Control-Allow-Origin", "https://app.tecnoeducando.edu.pe")
        return response
    except Exception as e:
        response = jsonify({"error": str(e)})
        response.headers.add("Access-Control-Allow-Origin", "https://app.tecnoeducando.edu.pe")
        return response, 500


# === EJECUCIÓN LOCAL ===
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
