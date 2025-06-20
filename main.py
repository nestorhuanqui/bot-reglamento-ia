from flask import Flask, request, jsonify
from flask_cors import CORS
import openai, os

# Configura tu API key
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Carga reglamento
with open("reglamento.txt", "r", encoding="utf-8") as f:
    reglamento = f.read()

app = Flask(__name__)
CORS(app,
     origins=["https://app.tecnoeducando.edu.pe"],
     methods=["GET", "POST", "OPTIONS"],
     allow_headers=["Content-Type", "X-Token"],
     supports_credentials=True)

TOKEN_PERMITIDO = "e398a7d3-dc9f-4ef9-bb29-07bff1672ef1"

@app.route("/consulta", methods=["POST"])
def consulta():
    if request.headers.get("X-Token") != TOKEN_PERMITIDO:
        return jsonify({"error": "No autorizado"}), 403

    pregunta = request.json.get("pregunta", "")
    prompt = f"""
Responde con base en el reglamento del colegio. No inventes respuestas.

--- Reglamento ---
{reglamento}
--- Fin ---

Pregunta: {pregunta}
"""

    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}]
        )
        texto = response.choices[0].message.content
        return jsonify({"respuesta": texto})

    except Exception as e:
        return jsonify({"error": f"OpenAI error: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
