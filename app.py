import logging
import os
import time
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv

# Azure AI Projects & Identity SDKs
from azure.identity import DefaultAzureCredential
from azure.core.exceptions import (
    ClientAuthenticationError,
    HttpResponseError,
    ResourceNotFoundError,
    ServiceRequestError,
)
from azure.ai.projects import AIProjectClient
from openai import APIConnectionError, APIError

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
_openai_client = None


def get_config():
    """
    Retrieves required Azure AI connection settings exclusively from environment variables.
    """
    project_endpoint = os.getenv("FOUNDRY_PROJECT_ENDPOINT", "").strip()
    agent_name = os.getenv("AGENT_NAME", "").strip()
    agent_id = os.getenv("AZURE_AI_AGENT_ID", "").strip()
    return project_endpoint, agent_name, agent_id


def get_openai_client(project_endpoint: str):
    """
    Initializes or returns cached OpenAI client via AIProjectClient.get_openai_client().
    """
    global _project_client, _openai_client
    if _openai_client is None:
        logger.info(f"Initializing AIProjectClient with endpoint: {project_endpoint}...")
        credential = DefaultAzureCredential()
        _project_client = AIProjectClient(
            endpoint=project_endpoint,
            credential=credential
        )
        _openai_client = _project_client.get_openai_client()
        logger.info("AIProjectClient and OpenAI client initialized successfully.")
    return _openai_client


@app.route("/")
def home():
    """Renders the chatbot web interface."""
    return render_template("index.html")


@app.route("/healthz")
def healthz():
    """Health check probe endpoint for Azure App Service."""
    project_endpoint, agent_name, agent_id = get_config()
    assistant_identifier = agent_id or agent_name
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
    Server-side chat endpoint communicating with your Azure AI Foundry Agent.
    """
    project_endpoint, agent_name, agent_id = get_config()

    # Resolve the assistant identifier: prefer AZURE_AI_AGENT_ID (GUID), fall back to AGENT_NAME
    assistant_identifier = agent_id or agent_name

    missing_vars = []
    if not project_endpoint:
        missing_vars.append("FOUNDRY_PROJECT_ENDPOINT")
    if not assistant_identifier:
        missing_vars.append("AZURE_AI_AGENT_ID or AGENT_NAME")

    if missing_vars:
        error_msg = f"Missing required environment variable(s): {', '.join(missing_vars)}."
        logger.error(f"Configuration Error: {error_msg}")
        return jsonify({
            "error": "ConfigurationError",
            "message": error_msg,
            "details": "Ensure FOUNDRY_PROJECT_ENDPOINT and AZURE_AI_AGENT_ID (or AGENT_NAME) are configured."
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
        openai_client = get_openai_client(project_endpoint)

        # 1. Thread creation if new conversation
        if not thread_id:
            logger.info("Creating new thread on Azure AI Foundry...")
            thread = openai_client.beta.threads.create()
            thread_id = thread.id
            logger.info(f"Thread created: {thread_id}")

        # 2. Append message
        openai_client.beta.threads.messages.create(
            thread_id=thread_id,
            role="user",
            content=user_message
        )

        # 3. Create run
        logger.info(f"Creating run for agent '{assistant_identifier}' on thread '{thread_id}'...")
        run = openai_client.beta.threads.runs.create(
            thread_id=thread_id,
            assistant_id=assistant_identifier
        )

        # 4. Poll status
        poll_start = time.time()
        timeout_seconds = 60

        while run.status in ["queued", "in_progress", "requires_action"]:
            if time.time() - poll_start > timeout_seconds:
                logger.warning(f"Agent run timed out after {timeout_seconds}s.")
                return jsonify({
                    "error": "TimeoutError",
                    "message": "The Azure AI Foundry Agent response timed out. Please try again.",
                    "thread_id": thread_id
                }), 504

            time.sleep(1)
            run = openai_client.beta.threads.runs.retrieve(
                run_id=run.id,
                thread_id=thread_id
            )

        if run.status != "completed":
            logger.error(f"Agent run failed: {run.status}, details: {getattr(run, 'last_error', None)}")
            return jsonify({
                "error": "AgentExecutionError",
                "message": f"Agent run ended with status: '{run.status}'.",
                "details": str(getattr(run, "last_error", "Check Azure AI Studio logs.")),
                "thread_id": thread_id
            }), 502

        # 5. Extract assistant message
        messages = openai_client.beta.threads.messages.list(thread_id=thread_id)
        assistant_reply = ""
        citations = []

        for msg in messages.data:
            if msg.role == "assistant":
                for content_part in msg.content:
                    if content_part.type == "text":
                        assistant_reply = content_part.text.value
                        if hasattr(content_part.text, "annotations") and content_part.text.annotations:
                            for idx, annotation in enumerate(content_part.text.annotations):
                                citation_info = {
                                    "index": idx + 1,
                                    "text": getattr(annotation, "text", f"[{idx+1}]"),
                                    "type": getattr(annotation, "type", "citation"),
                                    "source": "Azure AI Grounding Source"
                                }
                                citations.append(citation_info)
                break

        return jsonify({
            "reply": assistant_reply,
            "citations": citations,
            "thread_id": thread_id,
            "sender": "bot"
        }), 200

    except APIConnectionError as conn_err:
        logger.error(f"Connection Error: {conn_err}")
        return jsonify({
            "error": "ConnectionError",
            "message": "Failed to connect to Azure AI Foundry. Please check project endpoint and network.",
            "details": str(conn_err)
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
