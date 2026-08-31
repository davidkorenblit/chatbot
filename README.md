# Secure Azure AI Foundry Agent RAG Web App

A production-ready Python Flask chatbot application optimized for deployment on **Azure App Service (Linux Web App)**. It integrates with **Azure AI Foundry Agents** using **System-Assigned Managed Identity** (passwordless architecture) to fetch answers and display citation references dynamically.

## Key Security Practices
- **Passwordless Auth:** Authenticates dynamically using `DefaultAzureCredential` via the Web App's System-Assigned Managed Identity.
- **No Hardcoded Secrets:** Connection strings and identifiers are retrieved entirely from environment variables.
- **Server-Side Proxying:** All model calls occur in the Python backend to keep tokens secure.

## Prerequisites & Azure Setup

### 1. Enable Managed Identity
1. Go to your **Azure App Service** in the Azure Portal.
2. Under **Settings** -> **Identity**, toggle the Status of **System assigned** to **On** and save.

### 2. RBAC Permissions (Azure AI Project)
Grant the Web App's identity permission to use the AI Hub:
1. Go to the Azure AI Foundry Project / Resource Group in the Azure Portal.
2. Select **Access Control (IAM)** -> **Add role assignment**.
3. Assign the **Azure AI Developer** (or **Contributor**) role to the App Service's Managed Identity.

### 3. Application Settings (App Settings)
Add these environment variables in your App Service's **Configuration** blade (or local `.env`):
- `FOUNDRY_PROJECT_ENDPOINT`: The Project Endpoint for your Azure AI Foundry Project.
- `AGENT_NAME`: The unique ID or Name of your Azure AI Agent.

---

## Local Development

1. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On Linux/macOS:
   source venv/bin/activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Setup local environment variables by creating a `.env` file:
   ```ini
   FOUNDRY_PROJECT_ENDPOINT="https://<your-hub-or-project>.services.ai.azure.com/api/projects/<project-name>/..."
   AGENT_NAME="your-agent-id"
   ```
   *Note:* `DefaultAzureCredential` will automatically pick up your signed-in credentials from Azure CLI (`az login`) or VS Code.

4. Run locally:
   ```bash
   python app.py
   ```

---

## Azure App Service Deployment

- **Runtime Stack**: Python 3.10 / 3.11+
- **Startup Command** (Automatically detected, or set manually):
  ```bash
  gunicorn --bind=0.0.0.0 --timeout 600 app:app
  ```

---

## Challenges Encountered & Troubleshooting Journey

During the architecture setup, integration, and deployment of Azure AI Foundry, Azure AI Search, and Storage accounts, several key security and configuration challenges were encountered and resolved:

### 1. Agent Guardrails & Overly Strict Content Filtering (PII / Name Blocking)
- **The Challenge:** The primary objective of the agent is to serve as an interactive Resume/CV querying assistant. However, strict default content safety filters treated personal first and last names as blocked PII/sensitive entities, preventing the agent from answering questions regarding candidate details.
- **The Solution:** Adjusted the Azure AI Content Safety / moderation thresholds to a minimal filtering level appropriate for resume and portfolio data processing, resolving false-positive content blocks while maintaining security.

### 2. Azure AI Foundry to Azure AI Search `403 Forbidden` (Managed Identity Delegation)
- **The Challenge:** Connecting the Azure AI Foundry Project to Azure AI Search resulted in a `403 Forbidden` authorization error. Assigning RBAC permissions to the developer's personal Azure user identity (`c5418ed3...`) did not resolve the issue, as backend service-to-service communication relies on the Foundry project's own Managed Identity rather than the client user.
- **The Solution:** Located the System-Assigned Managed Identity of the Foundry Project (`proj-ailab-dk001`, Object ID: `bb459e63-8a40-4a0e-8d3c-6cd7cb3b2da5`) and granted it the necessary RBAC roles directly on the Azure AI Search resource:
  - `Search Index Data Contributor`
  - `Search Service Contributor`
  Following these role assignments, Azure AI Foundry successfully queried the search indexes.

### 3. Azure Storage Account & Container Scope Permissions
- **The Challenge:** Initial attempts to ingest documents from Azure Blob Storage (`storagebotcv001`, container: `docs`) failed due to insufficient permissions. Assigning the `Storage Blob Data Contributor` role using email address failed because Azure AD could not resolve the user identity via email string in the CLI. Furthermore, granting permissions strictly at the sub-container level still triggered access errors during certain control-plane operations.
- **The Solution:**
  1. Retrieved the exact User Principal Object ID (`c5418ed3-...`) from Azure AD and assigned the `Storage Blob Data Contributor` role directly using the Object ID at the `docs` container level.
  2. To eliminate authorization boundaries across the lab environment, elevated the `Storage Blob Data Contributor` assignment to the root Storage Account scope (`storagebotcv001`), ensuring full access across all storage containers and blobs.

