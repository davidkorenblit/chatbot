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

# Global client caches
_project_client = None
_openai_client = None


import re

def get_config():
    """
    Retrieves required Azure AI connection settings exclusively from environment variables.
    Automatically sanitizes any accidental 'endpoint: ' prefix or trailing subpaths.
    """
    raw_endpoint = os.getenv("FOUNDRY_PROJECT_ENDPOINT", "").strip()
    agent_name = os.getenv("AGENT_NAME", "").strip()

    # Clean up endpoint if it has 'endpoint: ' prefix or trailing protocol paths
    project_endpoint = raw_endpoint
    if project_endpoint:
        project_endpoint = re.sub(r'^[a-zA-Z_-]+:\s*', '', project_endpoint)
        if "/agents/" in project_endpoint:
            project_endpoint = project_endpoint.split("/agents/")[0]
        project_endpoint = project_endpoint.rstrip("/")

    return project_endpoint, agent_name



def get_openai_assistants_client(project_endpoint: str, agent_name: str = ""):
    """
    Initializes or returns cached OpenAI client via AIProjectClient.get_openai_client().
    Supports both project-level Assistants (threads/runs) and Agent endpoints (responses/chat).
    """
    global _project_client, _openai_client
    if _openai_client is None:
        logger.info("Initializing AIProjectClient with Foundry Project Endpoint (allow_preview=True)...")
        credential = DefaultAzureCredential()
        _project_client = AIProjectClient(
            endpoint=project_endpoint,
            credential=credential,
            allow_preview=True
        )
        try:
            if agent_name:
                _openai_client = _project_client.get_openai_client(agent_name=agent_name)
                logger.info(f"OpenAI client initialized for Agent '{agent_name}'.")
            else:
                _openai_client = _project_client.get_openai_client()
                logger.info("OpenAI client initialized for Project endpoint.")
        except Exception as e:
            logger.warning(f"Falling back to project-level OpenAI client: {e}")
            _openai_client = _project_client.get_openai_client()

    return _openai_client


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
    Supports Responses API, Assistants Threads/Runs API, and Chat Completions API.
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
        # 3. Client Initialization
        openai_client = get_openai_assistants_client(project_endpoint, agent_name)

        assistant_reply = ""
        citations = []

        # 4. Strategy A: Try Responses API (Azure AI Foundry Agent Endpoint protocol)
        if hasattr(openai_client, "responses") and callable(getattr(openai_client.responses, "create", None)):
            try:
                logger.info("Calling Azure AI Foundry Agent Responses API...")
                kwargs = {"input": user_message}
                if thread_id:
                    kwargs["conversation"] = thread_id
                
                resp = openai_client.responses.create(**kwargs)
                if hasattr(resp, "output_text") and resp.output_text:
                    assistant_reply = resp.output_text
                elif hasattr(resp, "output") and resp.output:
                    assistant_reply = str(resp.output)
                
                thread_id = getattr(resp, "conversation", None) or getattr(resp, "id", thread_id)
                logger.info("Responses API call succeeded.")
            except Exception as resp_err:
                logger.warning(f"Responses API call failed ({resp_err}), falling back to Assistants/Threads API...")

        # 5. Strategy B: Assistants API (Threads / Runs)
        if not assistant_reply and hasattr(openai_client, "beta") and hasattr(openai_client.beta, "threads"):
            if not thread_id:
                logger.info("Creating new conversation thread on Azure AI Foundry...")
                thread = openai_client.beta.threads.create()
                thread_id = thread.id

            openai_client.beta.threads.messages.create(
                thread_id=thread_id,
                role="user",
                content=user_message
            )

            logger.info(f"Triggering Agent run for thread '{thread_id}' with Agent '{agent_name}'...")
            run = openai_client.beta.threads.runs.create(
                thread_id=thread_id,
                assistant_id=agent_name
            )

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
                run = openai_client.beta.threads.runs.retrieve(
                    run_id=run.id,
                    thread_id=thread_id
                )

            if run.status != "completed":
                logger.error(f"Agent run failed with status: '{run.status}'. Run error details: {getattr(run, 'last_error', None)}")
                return jsonify({
                    "error": "AgentExecutionError",
                    "message": f"Azure AI Agent run completed with non-success status: '{run.status}'.",
                    "details": str(getattr(run, "last_error", "Check Azure AI Studio logs for details.")),
                    "thread_id": thread_id
                }), 502

            messages = openai_client.beta.threads.messages.list(thread_id=thread_id)
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
                                    }
                                    if hasattr(annotation, "file_citation") and annotation.file_citation:
                                        citation_info["source"] = getattr(annotation.file_citation, "quote", "File reference")
                                    elif hasattr(annotation, "url_citation") and annotation.url_citation:
                                        citation_info["source"] = getattr(annotation.url_citation, "url", "URL citation reference")
                                    else:
                                        citation_info["source"] = "Azure AI Grounding Source"
                                    citations.append(citation_info)
                    break

        # 6. Strategy C: Chat Completions API fallback
        if not assistant_reply and hasattr(openai_client, "chat") and hasattr(openai_client.chat, "completions"):
            logger.info("Calling Chat Completions API fallback...")
            chat_resp = openai_client.chat.completions.create(
                model=agent_name,
                messages=[{"role": "user", "content": user_message}]
            )
            if chat_resp.choices:
                assistant_reply = chat_resp.choices[0].message.content or ""

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
        # Check for openai APIConnectionError
        if type(e).__name__ == "APIConnectionError" or "APIConnectionError" in str(type(e)):
            logger.error(f"OpenAI APIConnectionError: {e}")
            return jsonify({
                "error": "ConnectionError",
                "message": "Failed to connect to Azure AI Foundry endpoint. Please verify FOUNDRY_PROJECT_ENDPOINT URL.",
                "details": str(e)
            }), 503

        logger.exception(f"Unexpected error processing chat request: {e}")
        return jsonify({
            "error": "InternalServerError",
            "message": "An unexpected error occurred while communicating with the Azure AI Foundry Agent.",
            "details": str(e)
        }), 500



if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port, debug=False)
