from flask import Flask, request, jsonify
from flask_cors import CORS
import openai, os

app = Flask(__name__)

# Configuración CORS: solo tu frontend
CORS(app,
     origins=["https://bot.tecnoeducando.edu.pe"],
     methods=["GET","POST","OPTIONS"],
     allow_headers=["Content-Type","X-Token"],
     supports_credentials=True
)

# Clave OpenAI desde variable de entorno
openai.api_key = os.getenv("OPENAI_API_KEY")

# Token de seguridad
TOKEN_PERMITIDO = "e398a7d3-dc9f-4ef9-bb29-07bff1672ef1"

# Carga reglamento
with open("reglamento.txt", "r", encoding="utf-8") as f:
    reglamento = f.read()

@app.route("/consulta", methods=["POST"])
def consulta():
    if request.headers.get("X-Token") != TOKEN_PERMITIDO:
        return jsonify({"error": "No autorizado"}), 403
    data = request.get_json() or {}
    pregunta = data.get("pregunta","").strip()
    if not pregunta:
        return jsonify({"error":"Pregunta vacía"}),400
    prompt = f"""Responde SOLO con base en este reglamento (no inventes nada):
    
--- Reglamento ---
{reglamento}
--- Fin ---

Pregunta: {pregunta}
"""
    try:
        resp = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role":"user","content":prompt}]
        )
        return jsonify({"respuesta": resp.choices[0].message.content})
    except Exception as e:
        return jsonify({"error": f"OpenAI error: {str(e)}"}),500

if __name__=="__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
