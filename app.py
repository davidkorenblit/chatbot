import logging
import os
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv

# Azure AI Foundry Projects & Identity SDKs
from azure.identity import DefaultAzureCredential
from azure.core.exceptions import (
    ClientAuthenticationError,
    HttpResponseError,
    ResourceNotFoundError,
    ServiceRequestError,
)
from azure.ai.projects import AIProjectClient

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
_project_client = None


def get_config():
    """
    Retrieves required Azure AI connection settings exclusively from environment variables.
    """
    project_endpoint = os.getenv("FOUNDRY_PROJECT_ENDPOINT", "").strip()
    agent_name = os.getenv("AGENT_NAME", "").strip()
    agent_id = os.getenv("AZURE_AI_AGENT_ID", "").strip()
    return project_endpoint, agent_name, agent_id


def get_project_client(project_endpoint: str):
    """
    Initializes or returns cached AIProjectClient from azure.ai.projects.
    """
    global _project_client
    if _project_client is None:
        logger.info(f"Initializing AIProjectClient with endpoint: {project_endpoint}...")
        credential = DefaultAzureCredential()
        _project_client = AIProjectClient(
            endpoint=project_endpoint,
            credential=credential
        )
        logger.info("AIProjectClient initialized successfully.")
    return _project_client


@app.route("/")
def home():
    """Renders the chatbot web interface."""
    return render_template("index.html")


@app.route("/healthz")
def healthz():
    """Health check probe endpoint for Azure App Service."""
    project_endpoint, agent_name, agent_id = get_config()
    assistant_identifier = agent_name or agent_id
    configured = bool(project_endpoint and assistant_identifier)
    return jsonify({
        "status": "healthy" if configured else "unconfigured",
        "service": "azure-ai-agent-chatbot",
        "project_endpoint_set": bool(project_endpoint),
        "agent_identifier_set": bool(assistant_identifier)
    }), 200 if configured else 503


@app.route("/api/chat", methods=["POST"])
def chat():
    """
    Server-side chat endpoint communicating with your Azure AI Foundry Agent
    via the Prompt Agent conversations & responses API.
    """
    project_endpoint, agent_name, agent_id = get_config()

    # Resolve the assistant identifier: prefer AGENT_NAME (e.g., 'doc-assistant'), fall back to AZURE_AI_AGENT_ID
    assistant_identifier = agent_name or agent_id

    missing_vars = []
    if not project_endpoint:
        missing_vars.append("FOUNDRY_PROJECT_ENDPOINT")
    if not assistant_identifier:
        missing_vars.append("AGENT_NAME or AZURE_AI_AGENT_ID")

    if missing_vars:
        error_msg = f"Missing required environment variable(s): {', '.join(missing_vars)}."
        logger.error(f"Configuration Error: {error_msg}")
        return jsonify({
            "error": "ConfigurationError",
            "message": error_msg,
            "details": "Ensure FOUNDRY_PROJECT_ENDPOINT and AGENT_NAME (or AZURE_AI_AGENT_ID) are configured."
        }), 500

    data = request.get_json(silent=True) or {}
    user_message = data.get("message", "").strip()
    thread_id = data.get("thread_id", "").strip()

    if not user_message:
        return jsonify({
            "error": "ValidationError",
            "message": "The 'message' field is required and cannot be empty."
        }), 400

    try:
        project_client = get_project_client(project_endpoint)
        openai_client = project_client.get_openai_client(agent_name=assistant_identifier)

        # 1. Conversation creation if new session
        if not thread_id:
            logger.info(f"Creating new conversation for agent '{assistant_identifier}'...")
            conversation = openai_client.conversations.create()
            thread_id = conversation.id
            logger.info(f"Conversation created: {thread_id}")

        # 2. Synchronous response execution
        logger.info(f"Executing response for conversation '{thread_id}'...")
        response = openai_client.responses.create(
            conversation=thread_id,
            input=user_message
        )

        # 3. Extract output text
        assistant_reply = getattr(response, "output_text", "")
        if not assistant_reply and hasattr(response, "text"):
            assistant_reply = str(response.text)

        logger.info(f"Response received, length: {len(assistant_reply)} chars")

        # 4. Extract citations/annotations if present
        citations = []
        annotations = getattr(response, "annotations", []) or getattr(response, "citations", []) or []
        if annotations:
            for idx, annotation in enumerate(annotations):
                citation_info = {
                    "index": idx + 1,
                    "text": getattr(annotation, "text", f"[{idx+1}]"),
                    "type": getattr(annotation, "type", "citation"),
                    "source": getattr(annotation, "source", "Azure AI Grounding Source")
                }
                citations.append(citation_info)

        if not assistant_reply:
            logger.warning(f"No reply text found in response for conversation '{thread_id}'.")

        return jsonify({
            "reply": assistant_reply,
            "citations": citations,
            "thread_id": thread_id,
            "sender": "bot"
        }), 200

    except (ServiceRequestError, HttpResponseError) as req_err:
        logger.error(f"Azure AI Service Request Error: {req_err}")
        return jsonify({
            "error": "ServiceError",
            "message": "Failed to communicate with Azure AI Foundry Agent Service.",
            "details": str(req_err)
        }), 503

    except ClientAuthenticationError as auth_err:
        logger.error(f"Auth Error: {auth_err}")
        return jsonify({
            "error": "AuthenticationFailed",
            "message": "Authentication failed with Managed Identity.",
            "details": str(auth_err)
        }), 401

    except Exception as e:
        logger.exception(f"Unexpected error: {e}")
        return jsonify({
            "error": "InternalServerError",
            "message": "An error occurred while communicating with the Azure AI Foundry Agent.",
            "details": str(e)
        }), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port, debug=False)
