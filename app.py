import logging
import os
import re
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv

# Azure & OpenAI SDKs
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from azure.core.exceptions import ClientAuthenticationError, HttpResponseError
from openai import AzureOpenAI, APIConnectionError, APIError

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("azure-ai-chatbot")

# Load environment variables from .env if running locally
load_dotenv()

app = Flask(__name__)

# Global client cache
_azure_openai_client = None


def get_config():
    """
    Retrieves and cleans Azure OpenAI connection settings from environment variables.
    """
    raw_endpoint = (
        os.getenv("AZURE_OPENAI_ENDPOINT")
        or os.getenv("FOUNDRY_PROJECT_ENDPOINT")
        or "https://hub-ailab-dk001.cognitiveservices.azure.com/"
    ).strip()

    model_name = (
        os.getenv("AZURE_OPENAI_DEPLOYMENT")
        or os.getenv("AGENT_NAME")
        or "gpt-4.1-mini"
    ).strip()

    # Clean endpoint: remove 'endpoint: ' prefix and trailing slashes
    endpoint = re.sub(r'^[a-zA-Z_-]+:\s*', '', raw_endpoint).rstrip("/")
    if not endpoint.startswith("http"):
        endpoint = f"https://{endpoint}"

    # If endpoint has project paths, clean them to root domain
    if "/api/projects" in endpoint:
        endpoint = endpoint.split("/api/projects")[0]
    if "/agents/" in endpoint:
        endpoint = endpoint.split("/agents/")[0]

    return endpoint, model_name


def get_azure_openai_client(endpoint: str):
    """
    Initializes or returns cached AzureOpenAI client authenticated via Managed Identity.
    """
    global _azure_openai_client
    if _azure_openai_client is None:
        logger.info(f"Initializing AzureOpenAI client for endpoint: {endpoint}")
        credential = DefaultAzureCredential()
        token_provider = get_bearer_token_provider(
            credential,
            "https://cognitiveservices.azure.com/.default"
        )
        _azure_openai_client = AzureOpenAI(
            azure_endpoint=endpoint,
            azure_ad_token_provider=token_provider,
            api_version="2024-05-01-preview"
        )
        logger.info("AzureOpenAI client initialized successfully.")
    return _azure_openai_client


@app.route("/")
def home():
    """Renders the chatbot web interface."""
    return render_template("index.html")


@app.route("/healthz")
def healthz():
    """Health check probe endpoint for Azure App Service."""
    endpoint, model_name = get_config()
    configured = bool(endpoint and model_name)
    return jsonify({
        "status": "healthy" if configured else "unconfigured",
        "service": "azure-ai-chatbot",
        "endpoint": endpoint,
        "model": model_name
    }), 200 if configured else 503


@app.route("/api/chat", methods=["POST"])
def chat():
    """
    Server-side chat endpoint communicating directly with Azure OpenAI model.
    """
    endpoint, model_name = get_config()

    data = request.get_json(silent=True) or {}
    user_message = data.get("message", "").strip()
    conversation_history = data.get("history", [])

    if not user_message:
        return jsonify({
            "error": "ValidationError",
            "message": "The 'message' field is required and cannot be empty."
        }), 400

    try:
        client = get_azure_openai_client(endpoint)

        # Build messages list
        messages = [
            {"role": "system", "content": "You are a helpful, precise, and friendly AI assistant."}
        ]

        # Add any previous history if provided
        for msg in conversation_history:
            if isinstance(msg, dict) and msg.get("role") and msg.get("content"):
                messages.append({"role": msg["role"], "content": msg["content"]})

        # Add current user message
        messages.append({"role": "user", "content": user_message})

        logger.info(f"Sending chat request to model '{model_name}' at '{endpoint}'...")
        response = client.chat.completions.create(
            model=model_name,
            messages=messages
        )

        assistant_reply = ""
        if response.choices:
            assistant_reply = response.choices[0].message.content or ""

        return jsonify({
            "reply": assistant_reply,
            "citations": [],
            "thread_id": "conv-" + str(os.urandom(4).hex()),
            "sender": "bot"
        }), 200

    except APIConnectionError as conn_err:
        logger.error(f"Azure OpenAI Connection Error: {conn_err}")
        return jsonify({
            "error": "ConnectionError",
            "message": f"Failed to connect to Azure OpenAI at {endpoint}. Please verify network and endpoint.",
            "details": str(conn_err)
        }), 503

    except ClientAuthenticationError as auth_err:
        logger.error(f"Authentication failure with Managed Identity: {auth_err}")
        return jsonify({
            "error": "AuthenticationFailed",
            "message": "Authentication failed while acquiring token with Managed Identity.",
            "details": str(auth_err)
        }), 401

    except HttpResponseError as http_err:
        status_code = getattr(http_err, "status_code", 500)
        logger.error(f"Azure returned HTTP {status_code}: {http_err.message}")
        return jsonify({
            "error": "AzureServiceError",
            "message": f"Azure OpenAI returned HTTP {status_code}.",
            "details": http_err.message
        }), status_code

    except Exception as e:
        logger.exception(f"Unexpected error processing chat request: {e}")
        return jsonify({
            "error": "InternalServerError",
            "message": "An error occurred while generating response.",
            "details": str(e)
        }), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port, debug=False)
