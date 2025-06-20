from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import faiss
import pickle
from sentence_transformers import SentenceTransformer
import requests

# Configuración
HF_TOKEN = os.getenv("HF_TOKEN")  # Token Hugging Face (en Render)
MODEL_URL = "https://api-inference.huggingface.co/models/HuggingFaceH4/zephyr-7b-alpha"

# Carga embeddings y fragmentos
with open("fragments.pkl", "rb") as f:
    fragments = pickle.load(f)

index = faiss.read_index("reglamento.index")
embedding_model = SentenceTransformer("BAAI/bge-small-en-v1.5")

# Flask app
app = Flask(__name__)
CORS(app, supports_credentials=True, resources={
    r"/*": {
        "origins": ["https://app.tecnoeducando.edu.pe"],
        "methods": ["POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "X-Token"]
    }
})

# Seguridad
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
Responde con base en el siguiente reglamento. Sé claro y no inventes información.

--- CONTEXTO ---
{contexto}
--- FIN ---

Pregunta: {pregunta}
"""

    try:
        response = requests.post(
            MODEL_URL,
            headers={
                "Authorization": f"Bearer {HF_TOKEN}",
                "Content-Type": "application/json"
            },
            json={"inputs": prompt}
        )

        data = response.json()
        if isinstance(data, list) and "generated_text" in data[0]:
            return jsonify({"respuesta": data[0]["generated_text"].split("Pregunta:")[-1].strip()})
        else:
            return jsonify({"error": data}), 500

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
