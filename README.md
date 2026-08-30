# Azure Python Chatbot Web App

A production-ready Python Flask chatbot application optimized for deployment on **Azure App Service (Linux Web App)**.

## Features
- **Azure App Service Ready**: Uses dynamic `PORT` binding and Gunicorn WSGI.
- **Health Check Endpoint**: `/healthz` for Azure Health Probes.
- **REST Chat API**: `/api/chat` endpoint.
- **Modern Web UI**: Responsive glassmorphism interface.

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

3. Run locally:
   ```bash
   python app.py
   ```

## Azure App Service Deployment

- **Runtime Stack**: Python 3.10+ / 3.11+
- **Startup Command** (optional, Azure detects automatically with `app.py`):
  ```bash
  gunicorn --bind=0.0.0.0 --timeout 600 app:app
  ```
