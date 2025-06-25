from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import faiss
import pickle
import requests
from sentence_transformers import SentenceTransformer

# === CONFIGURACIÓN GENERAL ===
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")  # Asegúrate que esté definido en Render
API_URL = "https://api.deepseek.com/v1/chat/completions"
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

    try:
        with open("fragments.pkl", "rb") as f:
            fragments = pickle.load(f)
        index = faiss.read_index("reglamento.index")
    except Exception:
        return jsonify({"error": "No hay documento cargado"}), 500

    # === BÚSQUEDA SEMÁNTICA ===
    pregunta_vec = embedding_model.encode([pregunta])
    D, I = index.search(pregunta_vec, k=5)
    contexto = "\n\n".join([fragments[i] for i in I[0]])

    # === PROMPT MEJORADO CON INSTRUCCIONES HUMANAS ===
    prompt = f"""Eres un asistente virtual amable, claro y profesional. Responde únicamente con base en el siguiente reglamento. 
Si la pregunta usa sinónimos o expresiones similares a las del reglamento, intenta interpretarlas correctamente. 
Si no encuentras información relacionada en el reglamento, responde exactamente: "No se encuentra en el reglamento".

--- CONTEXTO ---
{contexto}
--- FIN ---

Pregunta: {pregunta}
Respuesta:"""

    # === LLAMADA A DEEPSEEK ===
    try:
        res = requests.post(API_URL,
                            headers={
                                "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
                                "Content-Type": "application/json"
                            },
                            json={
                                "model": MODEL_NAME,
                                "messages": [{"role": "user", "content": prompt}],
                                "temperature": 0.3
                            })

        r = res.json()
        if "choices" in r:
            texto = r["choices"][0]["message"]["content"].strip()
            return jsonify({"respuesta": texto})
        else:
            return jsonify({"error": r}), 500

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# === MAIN ===
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
