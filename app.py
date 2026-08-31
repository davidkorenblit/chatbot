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
    Never hardcode endpoints, keys, or agent identifiers.
    """
    project_endpoint = os.getenv("FOUNDRY_PROJECT_ENDPOINT", "").strip()
    agent_name = os.getenv("AGENT_NAME", "").strip()
    return project_endpoint, agent_name


def get_ai_project_client(project_endpoint: str) -> AIProjectClient:
    """
    Initializes or returns cached AIProjectClient authenticated via
    Azure System-Assigned Managed Identity using DefaultAzureCredential with the Foundry Project Endpoint.
    """
    global _project_client
    if _project_client is None:
        logger.info("Initializing AIProjectClient using DefaultAzureCredential with Foundry Project Endpoint...")
        credential = DefaultAzureCredential()
        _project_client = AIProjectClient(
            endpoint=project_endpoint,
            credential=credential
        )
        logger.info("AIProjectClient initialized successfully.")
    return _project_client


# --- Cross-Version SDK Compatibility Helpers (Supports azure-ai-projects 1.x & 2.x+) ---
def get_agents_handler(client: AIProjectClient):
    """
    Returns an interface for threads, messages, and runs whether using
    azure-ai-projects 2.x+ (via get_openai_client().beta.threads) or 1.x (via client.agents).
    """
    if hasattr(client, "get_openai_client"):
        return "openai", client.get_openai_client()
    elif hasattr(client, "agents"):
        return "agents", client.agents
    raise AttributeError("AIProjectClient has neither 'get_openai_client' nor 'agents'.")


def create_agent_thread(client: AIProjectClient):
    mode, handler = get_agents_handler(client)
    if mode == "openai":
        return handler.beta.threads.create()
    else:
        if hasattr(handler, "create_thread"):
            return handler.create_thread()
        elif hasattr(handler, "threads") and hasattr(handler.threads, "create"):
            return handler.threads.create()
        raise AttributeError("Could not create thread with current SDK structure.")


def create_agent_message(client: AIProjectClient, thread_id: str, role: str, content: str):
    mode, handler = get_agents_handler(client)
    if mode == "openai":
        return handler.beta.threads.messages.create(thread_id=thread_id, role=role, content=content)
    else:
        if hasattr(handler, "create_message"):
            return handler.create_message(thread_id=thread_id, role=role, content=content)
        elif hasattr(handler, "messages") and hasattr(handler.messages, "create"):
            return handler.messages.create(thread_id=thread_id, role=role, content=content)
        raise AttributeError("Could not create message with current SDK structure.")


def create_agent_run(client: AIProjectClient, thread_id: str, agent_name: str):
    mode, handler = get_agents_handler(client)
    if mode == "openai":
        return handler.beta.threads.runs.create(thread_id=thread_id, assistant_id=agent_name)
    else:
        if hasattr(handler, "create_run"):
            try:
                return handler.create_run(thread_id=thread_id, assistant_id=agent_name)
            except TypeError:
                return handler.create_run(thread_id=thread_id, agent_id=agent_name)
        elif hasattr(handler, "runs") and hasattr(handler.runs, "create"):
            try:
                return handler.runs.create(thread_id=thread_id, assistant_id=agent_name)
            except TypeError:
                return handler.runs.create(thread_id=thread_id, agent_id=agent_name)
        raise AttributeError("Could not create run with current SDK structure.")


def get_agent_run(client: AIProjectClient, thread_id: str, run_id: str):
    mode, handler = get_agents_handler(client)
    if mode == "openai":
        return handler.beta.threads.runs.retrieve(thread_id=thread_id, run_id=run_id)
    else:
        if hasattr(handler, "get_run"):
            return handler.get_run(thread_id=thread_id, run_id=run_id)
        elif hasattr(handler, "runs") and hasattr(handler.runs, "get"):
            return handler.runs.get(thread_id=thread_id, run_id=run_id)
        elif hasattr(handler, "runs") and hasattr(handler.runs, "retrieve"):
            return handler.runs.retrieve(thread_id=thread_id, run_id=run_id)
        raise AttributeError("Could not get run status with current SDK structure.")


def list_agent_messages(client: AIProjectClient, thread_id: str):
    mode, handler = get_agents_handler(client)
    if mode == "openai":
        return handler.beta.threads.messages.list(thread_id=thread_id)
    else:
        if hasattr(handler, "list_messages"):
            return handler.list_messages(thread_id=thread_id)
        elif hasattr(handler, "messages") and hasattr(handler.messages, "list"):
            return handler.messages.list(thread_id=thread_id)
        raise AttributeError("Could not list messages with current SDK structure.")


@app.route("/")
def home():
    """Renders the chatbot web interface."""
    return render_template("index.html")


@app.route("/healthz")
def healthz():
    """Health check probe endpoint for Azure App Service."""
    project_endpoint, agent_name = get_config()
    configured = bool(project_endpoint and agent_name)
    return jsonify({
        "status": "healthy" if configured else "unconfigured",
        "service": "azure-ai-agent-chatbot",
        "project_endpoint_set": bool(project_endpoint),
        "agent_name_set": bool(agent_name)
    }), 200 if configured else 503


@app.route("/api/chat", methods=["POST"])
def chat():
    """
    Server-side chat endpoint proxying requests securely to Azure AI Foundry Agent.
    All token acquisition, agent execution, and API calls are handled strictly on the backend.
    """
    # 1. Configuration Validation
    project_endpoint, agent_name = get_config()
    missing_vars = []
    if not project_endpoint:
        missing_vars.append("FOUNDRY_PROJECT_ENDPOINT")
    if not agent_name:
        missing_vars.append("AGENT_NAME")

    if missing_vars:
        error_msg = f"Missing required environment variable(s): {', '.join(missing_vars)}."
        logger.error(f"Configuration Error: {error_msg}")
        return jsonify({
            "error": "ConfigurationError",
            "message": error_msg,
            "details": "Ensure FOUNDRY_PROJECT_ENDPOINT and AGENT_NAME are configured in App Service Settings / environment variables."
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
        client = get_ai_project_client(project_endpoint)

        # 4. Thread Lifecycle Management
        if not thread_id:
            logger.info("Creating new conversation thread on Azure AI Foundry...")
            thread = create_agent_thread(client)
            thread_id = thread.id
            logger.info(f"Thread created with ID: {thread_id}")

        # 5. Append User Message
        create_agent_message(
            client=client,
            thread_id=thread_id,
            role="user",
            content=user_message
        )

        # 6. Execute Agent Run
        logger.info(f"Triggering Agent run for thread '{thread_id}' with Agent '{agent_name}'...")
        run = create_agent_run(
            client=client,
            thread_id=thread_id,
            agent_name=agent_name
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
            run = get_agent_run(client=client, thread_id=thread_id, run_id=run.id)

        if run.status != "completed":
            logger.error(f"Agent run failed with status: '{run.status}'. Run error details: {getattr(run, 'last_error', None)}")
            return jsonify({
                "error": "AgentExecutionError",
                "message": f"Azure AI Agent run completed with non-success status: '{run.status}'.",
                "details": str(getattr(run, "last_error", "Check Azure AI Studio logs for details.")),
                "thread_id": thread_id
            }), 502

        # 8. Retrieve Messages & Parse Annotations/Citations
        messages = list_agent_messages(client=client, thread_id=thread_id)
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
                "details": "Check that AGENT_NAME and FOUNDRY_PROJECT_ENDPOINT match your Azure AI Foundry resource."
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
            "message": "The requested Agent Name/ID or conversation Thread ID was not found.",
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
