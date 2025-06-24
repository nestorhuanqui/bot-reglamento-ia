from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import faiss
import pickle
import requests
from sentence_transformers import SentenceTransformer
import tempfile
import docx
import PyPDF2
from werkzeug.utils import secure_filename

# === CONFIGURACIÓN ===
HUGGINGFACE_TOKEN = os.getenv("HF_TOKEN")
API_URL = "https://api.deepseek.com/v1/chat/completions"
MODEL_NAME = "deepseek-chat"
TOKEN_PERMITIDO = "e398a7d3-dc9f-4ef9-bb29-07bff1672ef1"

# === MODELO DE EMBEDDINGS ===
embedding_model = SentenceTransformer("BAAI/bge-small-en-v1.5")

# === FLASK ===
app = Flask(__name__)
CORS(app,
     resources={r"/consulta": {"origins": ["https://app.tecnoeducando.edu.pe"]},
                r"/subir-doc": {"origins": ["https://app.tecnoeducando.edu.pe"]}},
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
                                "Authorization": f"Bearer {HUGGINGFACE_TOKEN}",
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

# === RUTA /subir-doc ===
@app.route("/subir-doc", methods=["POST", "OPTIONS"])
def subir_doc():
    if request.method == "OPTIONS":
        response = jsonify({})
        response.headers.add("Access-Control-Allow-Origin", "https://app.tecnoeducando.edu.pe")
        response.headers.add("Access-Control-Allow-Headers", "Content-Type, X-Token")
        response.headers.add("Access-Control-Allow-Methods", "POST, OPTIONS")
        return response, 200

    if request.headers.get("X-Token") != TOKEN_PERMITIDO:
        return jsonify({"error": "No autorizado"}), 403

    file = request.files.get("documento")
    if not file:
        return jsonify({"error": "Archivo no recibido"}), 400

    ext = file.filename.split(".")[-1].lower()
    temp_path = os.path.join(tempfile.gettempdir(), secure_filename(file.filename))
    file.save(temp_path)

    try:
        if ext == "txt":
            with open(temp_path, "r", encoding="utf-8") as f:
                texto = f.read()
        elif ext == "pdf":
            texto = ""
            with open(temp_path, "rb") as f:
                reader = PyPDF2.PdfReader(f)
                for page in reader.pages:
                    texto += page.extract_text() + "\n"
        elif ext == "docx":
            doc = docx.Document(temp_path)
            texto = "\n".join(p.text for p in doc.paragraphs)
        else:
            return jsonify({"error": "Formato no soportado"}), 400

        fragmentos = [frag.strip() for frag in texto.split("\n\n") if frag.strip()]
        vectores = embedding_model.encode(fragmentos, convert_to_numpy=True)
        index = faiss.IndexFlatL2(vectores.shape[1])
        index.add(vectores)

        faiss.write_index(index, "reglamento.index")
        with open("fragments.pkl", "wb") as f:
            pickle.dump(fragmentos, f)

        return jsonify({"mensaje": f"Documento cargado. Fragmentos: {len(fragmentos)}"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# === MAIN ===
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
