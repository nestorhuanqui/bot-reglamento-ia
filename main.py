<!DOCTYPE html>
<html lang="es">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <meta name="description" content="Asistente virtual del colegio para resolver dudas de padres de familia.">
  <title>Asistente del Colegio</title>
  <style>
    body {
      font-family: 'Segoe UI', sans-serif;
      background-color: #ece5dd;
      margin: 0;
      padding: 0;
    }

    h2 {
      text-align: center;
      background: #25D366;
      color: white;
      padding: 15px;
      margin: 0;
      font-size: 1.3em;
    }

    .chat-container {
      max-width: 500px;
      margin: auto;
      background: #fff;
      display: flex;
      flex-direction: column;
      height: 100vh;
    }

    .chat-box {
      flex: 1;
      padding: 10px;
      overflow-y: auto;
    }

    .mensaje {
      margin: 8px 0;
      padding: 10px;
      border-radius: 10px;
      max-width: 80%;
      line-height: 1.4;
      word-wrap: break-word;
    }

    .usuario {
      align-self: flex-end;
      background-color: #dcf8c6;
    }

    .bot {
      align-self: flex-start;
      background-color: #f1f0f0;
      border: 1px solid #ccc;
    }

    .entrada {
      display: flex;
      padding: 10px;
      border-top: 1px solid #ccc;
      background: #f0f0f0;
    }

    input[type="text"] {
      flex: 1;
      padding: 10px;
      border: none;
      border-radius: 20px;
      outline: none;
      font-size: 1em;
    }

    button {
      background: #25D366;
      border: none;
      color: white;
      padding: 10px 20px;
      margin-left: 10px;
      border-radius: 20px;
      cursor: pointer;
      font-size: 1em;
    }

    button:disabled {
      background: #b9e0c5;
      cursor: not-allowed;
    }
  </style>
</head>
<body>

<h2>Asistente Virtual del Colegio</h2>

<div class="chat-container">
  <div class="chat-box" id="chatBox"></div>
  <div class="entrada">
    <input type="text" id="pregunta" placeholder="Escribe tu consulta...">
    <button onclick="enviarConsulta()" id="btnEnviar">Enviar</button>
  </div>
</div>

<script>
  const URL_BACKEND = "https://bot.tecnoeducando.edu.pe/consulta";
  const TOKEN = "e398a7d3-dc9f-4ef9-bb29-07bff1672ef1";

  const LIMITE_CONSULTAS = 5;
  const TIEMPO_LIMITE = 60000; // 1 minuto
  let consultas = [];

  function puedeConsultar() {
    const ahora = Date.now();
    consultas = consultas.filter(ts => ahora - ts < TIEMPO_LIMITE);
    if (consultas.length >= LIMITE_CONSULTAS) {
      alert("Has alcanzado el límite de 5 consultas por minuto.");
      return false;
    }
    consultas.push(ahora);
    return true;
  }

  function enviarConsulta() {
    const input = document.getElementById("pregunta");
    const texto = input.value.trim();
    const btn = document.getElementById("btnEnviar");

    if (!texto || !puedeConsultar()) return;

    agregarMensaje(texto, 'usuario');
    input.value = "";
    btn.disabled = true;

    const idTemp = agregarMensaje("Pensando...", 'bot');

    fetch(URL_BACKEND, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "X-Token": TOKEN
      },
      body: JSON.stringify({ pregunta: texto })
    })
    .then(async res => {
      if (!res.ok) {
        const error = await res.text();
        throw new Error("Respuesta no válida: " + error);
      }
      return res.json();
    })
    .then(data => {
      actualizarMensaje(idTemp, data.respuesta || "El bot no pudo generar una respuesta.");
    })
    .catch(err => {
      console.error(err);
      actualizarMensaje(idTemp, "Error al consultar el servidor: " + err.message);
    })
    .finally(() => {
      btn.disabled = false;
    });
  }

  function agregarMensaje(texto, clase) {
    const chatBox = document.getElementById("chatBox");
    const msg = document.createElement("div");
    msg.className = `mensaje ${clase}`;
    msg.textContent = texto;
    const id = "msg-" + Date.now();
    msg.id = id;
    chatBox.appendChild(msg);
    chatBox.scrollTop = chatBox.scrollHeight;
    return id;
  }

  function actualizarMensaje(id, nuevoTexto) {
    const el = document.getElementById(id);
    if (el) el.textContent = nuevoTexto;
  }
</script>

</body>
</html>
