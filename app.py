import os
import time
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv

# Import official Azure SDKs for Managed Identity authentication and Project Client
try:
    from azure.identity import DefaultAzureCredential
    from azure.ai.projects import AIProjectClient
    AZURE_SDK_AVAILABLE = True
except ImportError:
    AZURE_SDK_AVAILABLE = False

# Load local environment variables from .env if present
load_dotenv()

app = Flask(__name__)

# Config options from application environment variables (App Settings)
CONNECTION_STRING = os.getenv("AZURE_AI_PROJECT_CONNECTION_STRING", "").strip()
AGENT_ID = os.getenv("AZURE_AI_AGENT_ID", "").strip()

project_client = None

# Initialize Project Client using Managed Identity (System-Assigned)
if AZURE_SDK_AVAILABLE and CONNECTION_STRING:
    try:
        # DefaultAzureCredential handles authentication using System-Assigned Managed Identity
        # When running on Azure App Service, it automatically retrieves MSI tokens.
        credential = DefaultAzureCredential()
        project_client = AIProjectClient.from_connection_string(
            conn_str=CONNECTION_STRING,
            credential=credential
        )
        print("Successfully initialized Azure AI Project Client using Managed Identity.")
    except Exception as e:
        print(f"Error initializing Azure AI Project Client: {e}")
else:
    if not AZURE_SDK_AVAILABLE:
        print("Warning: azure-ai-projects or azure-identity SDK is not installed.")
    if not CONNECTION_STRING:
        print("Warning: AZURE_AI_PROJECT_CONNECTION_STRING is not set in environment variables.")

@app.route("/")
def home():
    """Renders the chatbot web interface."""
    return render_template("index.html")

@app.route("/healthz")
def healthz():
    """Health check endpoint for Azure App Service probes."""
    status = "healthy"
    details = "Running with static/fallback chat mode"
    
    if project_client:
        details = "Azure AI Projects client initialized"
    
    return jsonify({
        "status": status,
        "service": "azure-ai-agent-chatbot",
        "details": details
    }), 200

@app.route("/api/chat", methods=["POST"])
def chat():
    """
    Chat API endpoint. Proxies user messages to the Azure AI Foundry Agent
    using System-Assigned Managed Identity, retrieves the response, parses
    citations, and returns them to the frontend.
    """
    data = request.get_json(silent=True) or {}
    user_message = data.get("message", "").strip()
    thread_id = data.get("thread_id", "").strip()

    if not user_message:
        return jsonify({"error": "Empty message received"}), 400

    # FALLBACK MOCK MODE: If project connection string or agent is not configured, run in mock mode
    if not project_client or not AGENT_ID:
        mock_reply = (
            f"Hello! I am currently running in **Local Mock Mode** because "
            f"Azure AI Project integration is not fully configured yet.\n\n"
            f"**Your Message:** \"{user_message}\"\n\n"
            f"**To activate live agent integration:**\n"
            f"1. Assign the 'Azure AI Developer' role to the App Service System-Assigned Managed Identity.\n"
            f"2. Add `AZURE_AI_PROJECT_CONNECTION_STRING` and `AZURE_AI_AGENT_ID` to your App Settings."
        )
        mock_citations = [
            {
                "index": 1,
                "text": "App Service Identity Settings",
                "type": "file_citation",
                "source": "Enable System-Assigned Managed Identity in your Web App's Identity blade."
            },
            {
                "index": 2,
                "text": "Azure AI Agent RBAC permissions",
                "type": "url_citation",
                "source": "https://learn.microsoft.com/en-us/azure/ai-studio/concepts/rbac"
            }
        ]
        return jsonify({
            "reply": mock_reply,
            "citations": mock_citations,
            "thread_id": "mock-thread-123",
            "sender": "bot"
        })

    try:
        # Step 1: Create a Thread if thread_id is not already provided
        if not thread_id:
            thread = project_client.agents.create_thread()
            thread_id = thread.id
            print(f"Created new Agent conversation thread: {thread_id}")

        # Step 2: Add user message to the active Thread
        project_client.agents.create_message(
            thread_id=thread_id,
            role="user",
            content=user_message
        )

        # Step 3: Run the Agent on the Thread
        run = project_client.agents.create_run(
            thread_id=thread_id,
            agent_id=AGENT_ID
        )

        # Step 4: Poll run status until completion
        start_time = time.time()
        while run.status in ["queued", "in_progress"]:
            # Timeout safety after 30 seconds
            if time.time() - start_time > 30:
                return jsonify({
                    "reply": "Request Timeout: The Azure AI Agent took too long to complete the run. Please try again.",
                    "sender": "bot",
                    "thread_id": thread_id
                }), 504
            
            time.sleep(1)
            run = project_client.agents.get_run(thread_id=thread_id, run_id=run.id)

        if run.status != "completed":
            return jsonify({
                "reply": f"Azure Agent Run completed with status: '{run.status}'. Please review Azure AI Studio logs.",
                "sender": "bot",
                "thread_id": thread_id
            }), 502

        # Step 5: Retrieve message logs for the thread
        messages = project_client.agents.list_messages(thread_id=thread_id)
        
        assistant_reply = ""
        citations = []

        # Find the latest response from the assistant
        for msg in messages.data:
            if msg.role == "assistant":
                for content_part in msg.content:
                    if content_part.type == "text":
                        assistant_reply = content_part.text.value
                        
                        # Extract and parse citations/annotations (e.g. from vector search or tools)
                        if hasattr(content_part.text, "annotations") and content_part.text.annotations:
                            for idx, annotation in enumerate(content_part.text.annotations):
                                citation_info = {
                                    "index": idx + 1,
                                    "text": annotation.text,
                                    "type": annotation.type,
                                }
                                
                                # Format based on annotation type (file search RAG vs URL search)
                                if hasattr(annotation, "file_citation") and annotation.file_citation:
                                    citation_info["source"] = getattr(annotation.file_citation, "quote", "File citation reference")
                                elif hasattr(annotation, "url_citation") and annotation.url_citation:
                                    citation_info["source"] = getattr(annotation.url_citation, "url", "External link citation")
                                else:
                                    citation_info["source"] = "Azure AI Internal Search"
                                    
                                citations.append(citation_info)
                break  # Stop at the latest assistant message

        return jsonify({
            "reply": assistant_reply,
            "citations": citations,
            "thread_id": thread_id,
            "sender": "bot"
        })

    except Exception as e:
        print(f"Error communicating with Azure AI Foundry Agent: {e}")
        return jsonify({
            "reply": f"An error occurred while connecting to the Azure AI Agent service: {str(e)}",
            "sender": "bot",
            "thread_id": thread_id
        }), 500

if __name__ == "__main__":
    # App Service configures the PORT environment variable dynamically
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port, debug=False)
