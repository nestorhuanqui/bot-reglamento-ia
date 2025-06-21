from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import faiss
import pickle
from sentence_transformers import SentenceTransformer
import requests

# === CONFIGURACIÓN ===
HUGGINGFACE_TOKEN = os.getenv("HF_TOKEN")  # Configura esto en Render
API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.1"

# === CARGA DE EMBEDDINGS Y FRAGMENTOS ===
with open("fragments.pkl", "rb") as f:
    fragments = pickle.load(f)

index = faiss.read_index("reglamento.index")
embedding_model = SentenceTransformer("BAAI/bge-small-en-v1.5")

# === FLASK APP ===
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

    # Búsqueda semántica
    pregunta_vec = embedding_model.encode([pregunta])
    D, I = index.search(pregunta_vec, k=5)
    contexto = "\n\n".join([fragments[i] for i in I[0]])

    prompt = f"""Responde de forma clara basándote exclusivamente en el siguiente reglamento. 
Si la información no está en el reglamento, responde: "No se encuentra en el reglamento".

--- REGLAMENTO ---
{contexto}
--- FIN ---

Pregunta: {pregunta}
Respuesta:"""

    try:
        response = requests.post(
            API_URL,
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
            texto = result[0]["generated_text"].split("Respuesta:")[-1].strip()
            return jsonify({"respuesta": texto})
        else:
            return jsonify({"error": result}), 500

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)

# === RUTA DE VERIFICACIÓN DEL ESTADO DEL MODELO EN HUGGING FACE ===
@app.route("/modelo-status", methods=["GET"])
def modelo_status():
    try:
        response = requests.get(
            FALCON_API_URL,
            headers={
                "Authorization": f"Bearer {HUGGINGFACE_TOKEN}"
            }
        )

        if response.status_code == 200:
            return jsonify({"estado": "Disponible ✅"})
        elif response.status_code == 503:
            return jsonify({"estado": "⏳ Cargando modelo en Hugging Face... (503)"}), 503
        elif response.status_code == 401:
            return jsonify({"estado": "❌ Token inválido o sin permisos (401)"}), 401
        else:
            return jsonify({
                "estado": f"⚠️ Error desconocido",
                "código": response.status_code,
                "detalle": response.text
            }), response.status_code

    except Exception as e:
        return jsonify({"error": str(e)}), 500

