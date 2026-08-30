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
- `AZURE_AI_PROJECT_CONNECTION_STRING`: The connection string for your Azure AI Foundry Project.
  - *Format:* `<region>.api.azure.com;subscriptionId=<sub-id>;resourceGroupName=<rg>;projectName=<project-name>`
- `AZURE_AI_AGENT_ID`: The unique ID of your Azure AI Agent.

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
   AZURE_AI_PROJECT_CONNECTION_STRING="your-connection-string"
   AZURE_AI_AGENT_ID="your-agent-id"
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
