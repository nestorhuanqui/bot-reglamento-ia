from flask import Flask, request, jsonify
from flask_cors import CORS
from utils import cargar_index, buscar_similares
import httpx
import os

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"
TOKEN_PERMITIDO = "e398a7d3-dc9f-4ef9-bb29-07bff1672ef1"

app = Flask(__name__)
CORS(app, resources={r"/consulta": {"origins": "*"}})

# Carga FAISS y textos
index, textos = cargar_index()

@app.route("/consulta", methods=["POST"])
def consulta():
    if request.headers.get("X-Token") != TOKEN_PERMITIDO:
        return jsonify({"error": "No autorizado"}), 403

    data = request.get_json()
    pregunta = data.get("pregunta", "")

    fragmentos = buscar_similares(pregunta, index, textos)
    contexto = "\n---\n".join(fragmentos)

    prompt = f"""
Responde según el reglamento a continuación. No inventes.

--- Reglamento ---
{contexto}
--- Fin ---

Pregunta: {pregunta}
"""

    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "deepseek-chat",
        "messages": [{"role": "user", "content": prompt}]
    }

    try:
        response = httpx.post(DEEPSEEK_API_URL, headers=headers, json=payload)
        response.raise_for_status()
        result = response.json()
        respuesta = result["choices"][0]["message"]["content"]
        return jsonify({"respuesta": respuesta})
    except Exception as e:
        return jsonify({"error": f"Error al contactar DeepSeek: {e}"}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
