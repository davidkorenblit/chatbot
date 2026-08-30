import os
import requests
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("AZURE_OPENAI_API_KEY")
endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")

print("Endpoint:", endpoint)
print("Key prefix:", api_key[:10] if api_key else "None")

# Try to list models
headers = {
    "api-key": api_key,
    "Authorization": f"Bearer {api_key}" # sometimes standard Bearer works
}

try:
    # Azure OpenAI v1 endpoint models list:
    url = f"{endpoint}/models"
    print("Requesting:", url)
    res = requests.get(url, headers=headers)
    print("Status code:", res.status_code)
    print("Response text:", res.text[:1000])
except Exception as e:
    print("Error:", e)
