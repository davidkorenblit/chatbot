import logging
import os
import time
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv

# Azure SDKs for Managed Identity authentication and Project Client
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
    Never hardcode connection strings, keys, or agent IDs.
    """
    connection_string = os.getenv("AZURE_AI_PROJECT_CONNECTION_STRING", "").strip()
    agent_id = os.getenv("AZURE_AI_AGENT_ID", "").strip()
    return connection_string, agent_id


def get_ai_project_client(connection_string: str) -> AIProjectClient:
    """
    Initializes or returns cached AIProjectClient authenticated via
    Azure System-Assigned Managed Identity using DefaultAzureCredential.
    """
    global _project_client
    if _project_client is None:
        logger.info("Initializing AIProjectClient using DefaultAzureCredential (Managed Identity)...")
        credential = DefaultAzureCredential()
        _project_client = AIProjectClient.from_connection_string(
            conn_str=connection_string,
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
    connection_string, agent_id = get_config()
    configured = bool(connection_string and agent_id)
    return jsonify({
        "status": "healthy" if configured else "unconfigured",
        "service": "azure-ai-agent-chatbot",
        "connection_string_set": bool(connection_string),
        "agent_id_set": bool(agent_id)
    }), 200 if configured else 503


@app.route("/api/chat", methods=["POST"])
def chat():
    """
    Server-side chat endpoint proxying requests securely to Azure AI Foundry Agent.
    All token acquisition, agent execution, and API calls are handled strictly on the backend.
    """
    # 1. Configuration Validation
    connection_string, agent_id = get_config()
    missing_vars = []
    if not connection_string:
        missing_vars.append("AZURE_AI_PROJECT_CONNECTION_STRING")
    if not agent_id:
        missing_vars.append("AZURE_AI_AGENT_ID")

    if missing_vars:
        error_msg = f"Missing required environment variable(s): {', '.join(missing_vars)}."
        logger.error(f"Configuration Error: {error_msg}")
        return jsonify({
            "error": "ConfigurationError",
            "message": error_msg,
            "details": "Ensure AZURE_AI_PROJECT_CONNECTION_STRING and AZURE_AI_AGENT_ID are configured in App Service Settings / environment variables."
        }), 500

    # 2. Input Validation
    data = request.get_json(silent=True) or {}
    user_message = data.get("message", "").strip()
    thread_id = data.get("thread_id", "").strip()

    if not user_message:
        return jsonify({
            "error": "ValidationError",
            "message": "The 'message' field is required and cannot be empty."
        }), 400

    try:
        # 3. Client Initialization (System-Assigned Managed Identity via DefaultAzureCredential)
        client = get_ai_project_client(connection_string)

        # 4. Thread Lifecycle Management
        if not thread_id:
            logger.info("Creating new conversation thread on Azure AI Foundry...")
            thread = client.agents.create_thread()
            thread_id = thread.id
            logger.info(f"Thread created with ID: {thread_id}")

        # 5. Append User Message
        client.agents.create_message(
            thread_id=thread_id,
            role="user",
            content=user_message
        )

        # 6. Execute Agent Run
        logger.info(f"Triggering Agent run for thread '{thread_id}' with Agent ID '{agent_id}'...")
        run = client.agents.create_run(
            thread_id=thread_id,
            agent_id=agent_id
        )

        # 7. Poll Run Status with Timeout Protection
        poll_start = time.time()
        timeout_seconds = 45

        while run.status in ["queued", "in_progress", "requires_action"]:
            if time.time() - poll_start > timeout_seconds:
                logger.warning(f"Agent run '{run.id}' timed out after {timeout_seconds}s.")
                return jsonify({
                    "error": "TimeoutError",
                    "message": "The Azure AI Agent response timed out. Please try again.",
                    "thread_id": thread_id
                }), 504

            time.sleep(1)
            run = client.agents.get_run(thread_id=thread_id, run_id=run.id)

        if run.status != "completed":
            logger.error(f"Agent run failed with status: '{run.status}'. Run error details: {getattr(run, 'last_error', None)}")
            return jsonify({
                "error": "AgentExecutionError",
                "message": f"Azure AI Agent run completed with non-success status: '{run.status}'.",
                "details": str(getattr(run, "last_error", "Check Azure AI Studio logs for details.")),
                "thread_id": thread_id
            }), 502

        # 8. Retrieve Messages & Parse Annotations/Citations
        messages = client.agents.list_messages(thread_id=thread_id)
        assistant_reply = ""
        citations = []

        for msg in messages.data:
            if msg.role == "assistant":
                for content_part in msg.content:
                    if content_part.type == "text":
                        assistant_reply = content_part.text.value
                        
                        # Extract citations from annotations
                        if hasattr(content_part.text, "annotations") and content_part.text.annotations:
                            for idx, annotation in enumerate(content_part.text.annotations):
                                citation_info = {
                                    "index": idx + 1,
                                    "text": getattr(annotation, "text", f"[{idx+1}]"),
                                    "type": getattr(annotation, "type", "citation"),
                                }

                                if hasattr(annotation, "file_citation") and annotation.file_citation:
                                    citation_info["source"] = getattr(annotation.file_citation, "quote", "File reference")
                                elif hasattr(annotation, "url_citation") and annotation.url_citation:
                                    citation_info["source"] = getattr(annotation.url_citation, "url", "URL citation reference")
                                else:
                                    citation_info["source"] = "Azure AI Grounding Source"

                                citations.append(citation_info)
                break  # Process latest assistant message

        return jsonify({
            "reply": assistant_reply,
            "citations": citations,
            "thread_id": thread_id,
            "sender": "bot"
        }), 200

    # 9. Standard Error Handling for Authentication & Authorization Failures
    except ClientAuthenticationError as auth_err:
        logger.error(f"Authentication failure with Managed Identity / DefaultAzureCredential: {auth_err}")
        return jsonify({
            "error": "AuthenticationFailed",
            "message": "HTTP 401: Authentication failed while acquiring token with Managed Identity.",
            "details": "Ensure System-Assigned Managed Identity is enabled on your Azure App Service."
        }), 401

    except HttpResponseError as http_err:
        status_code = getattr(http_err, "status_code", 500)
        logger.error(f"Azure AI Foundry service returned HTTP {status_code}: {http_err.message}")

        if status_code == 401:
            return jsonify({
                "error": "Unauthorized",
                "message": "HTTP 401: Unauthorized access to Azure AI Foundry Agent endpoint.",
                "details": "Invalid or expired credentials provided by DefaultAzureCredential."
            }), 401
        elif status_code == 403:
            return jsonify({
                "error": "Forbidden",
                "message": "HTTP 403: Access forbidden. The Managed Identity lacks required RBAC permissions.",
                "details": "Verify that the App Service Managed Identity has been assigned the 'Azure AI Developer' or 'Cognitive Services OpenAI Contributor' role on the Azure AI Project."
            }), 403
        elif status_code == 404:
            return jsonify({
                "error": "NotFound",
                "message": "HTTP 404: Azure AI Foundry Project, Agent, or Thread was not found.",
                "details": "Check that AZURE_AI_AGENT_ID and connection string parameters match your Azure AI Foundry resource."
            }), 404
        else:
            return jsonify({
                "error": "AzureServiceError",
                "message": f"Azure AI Foundry API returned HTTP {status_code}.",
                "details": http_err.message
            }), status_code

    except ResourceNotFoundError as not_found_err:
        logger.error(f"Resource not found: {not_found_err}")
        return jsonify({
            "error": "ResourceNotFound",
            "message": "The requested Agent ID or conversation Thread ID was not found.",
            "details": str(not_found_err)
        }), 404

    except ServiceRequestError as req_err:
        logger.error(f"Network error connecting to Azure AI Foundry: {req_err}")
        return jsonify({
            "error": "ServiceUnavailable",
            "message": "Network error while connecting to Azure AI Foundry endpoint.",
            "details": str(req_err)
        }), 503

    except Exception as e:
        logger.exception(f"Unexpected error processing chat request: {e}")
        return jsonify({
            "error": "InternalServerError",
            "message": "An unexpected error occurred while communicating with the Azure AI Foundry Agent.",
            "details": str(e)
        }), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port, debug=False)
