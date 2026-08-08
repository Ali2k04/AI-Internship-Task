"""
app.py
------
Flask web UI for the chatbot (bonus feature).

Run with:
    pip install flask
    python app.py
Then open http://127.0.0.1:5000 in your browser.

Endpoints:
    GET  /            -> chat page
    POST /chat         -> {"message": "...", "user_id": "..."} -> JSON reply
    GET  /history/<id> -> JSON conversation history for a user
    POST /voice        -> capture one utterance from the server mic (optional,
                           only works when running app.py on a machine with a
                           microphone and the voice.py deps installed)
"""

from flask import Flask, render_template, request, jsonify

from chatbot import Chatbot
import voice

app = Flask(__name__)
bot = Chatbot()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json(force=True)
    message = data.get("message", "")
    user_id = data.get("user_id", "web_user")
    if not message:
        return jsonify({"error": "message is required"}), 400
    result = bot.chat(message, user_id=user_id)
    return jsonify(result)


@app.route("/history/<user_id>", methods=["GET"])
def history(user_id):
    return jsonify(bot.history(user_id))


@app.route("/voice", methods=["POST"])
def voice_turn():
    """Record one spoken utterance on the server, run it through the
    chatbot, and speak the reply back. Requires SpeechRecognition +
    pyttsx3 + pyaudio and a working microphone on the server host."""
    if not voice.voice_available():
        return jsonify({"error": "Voice packages not installed on server."}), 501

    user_id = request.args.get("user_id", "voice_user")
    spoken_text = voice.listen()
    if not spoken_text:
        return jsonify({"error": "Could not understand audio."}), 400

    result = bot.chat(spoken_text, user_id=user_id)
    voice.speak(result["reply"])
    result["heard"] = spoken_text
    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True)
