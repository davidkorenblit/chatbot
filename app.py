import os
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# Sample intelligent responses / chatbot logic
BOT_RESPONSES = {
    "hello": "Hello! How can I assist you today?",
    "hi": "Hi there! What can I help you with?",
    "who are you": "I am an Azure-ready Python Chatbot service!",
    "status": "All systems operational on Azure Web App.",
    "help": "You can ask me questions, test API responses, or connect me to an LLM provider (Azure OpenAI, etc.)."
}

@app.route("/")
def home():
    """Renders the chatbot web interface."""
    return render_template("index.html")

@app.route("/healthz")
def healthz():
    """Health check endpoint for Azure App Service."""
    return jsonify({"status": "healthy", "service": "azure-python-chatbot"}), 200

@app.route("/api/chat", methods=["POST"])
def chat():
    """Chat API endpoint."""
    data = request.get_json(silent=True) or {}
    user_message = data.get("message", "").strip()

    if not user_message:
        return jsonify({"error": "Empty message received"}), 400

    # Match predefined answers or fallback to echo/default response
    lowered = user_message.lower()
    reply = BOT_RESPONSES.get(
        lowered,
        f"I received your message: '{user_message}'. Connect an Azure OpenAI or LLM backend to enable full conversational intelligence!"
    )

    return jsonify({
        "reply": reply,
        "sender": "bot"
    })

if __name__ == "__main__":
    # Azure Web App assigns the PORT environment variable (default 8000 for Azure App Service)
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port, debug=False)
