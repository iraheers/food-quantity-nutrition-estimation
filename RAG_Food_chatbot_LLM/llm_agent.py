import requests
import base64
from history_logger import log_interaction, get_history

# Ollama HTTP API endpoint (run: ollama serve)
OLLAMA_API_URL = "http://localhost:11434/api/generate"

# Default prompt for images
DEFAULT_IMAGE_PROMPT = "Describe what food item is shown in this image."


def chat_with_image(image_path: str) -> str:
    return run_ollama_image(image_path, DEFAULT_IMAGE_PROMPT)


def chat_with_text(text: str) -> str:
    # Retrieve chat history
    history = get_history()
    history_text = "\n".join([f"User: {h['input']}\nBot: {h['response']}" for h in history])

    # Add new user input
    if history_text:
        prompt = f"{history_text}\nUser: {text}\nBot:"
    else:
        prompt = f"User: {text}\nBot:"

    # Make the API call to Ollama
    payload = {"model": "llava:latest", "prompt": prompt, "stream": False}
    res = requests.post(OLLAMA_API_URL, json=payload)
    data = res.json()
    response = data.get("response", f"Error: {data}")

    # Save interaction
    log_interaction(text, response)

    return response


def run_ollama_image(image_path: str, prompt: str) -> str:
    with open(image_path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode("utf-8")
    payload = {
        "model": "llava:latest",
        "prompt": prompt,
        "stream": False,
        "images": [img_b64]
    }
    res = requests.post(OLLAMA_API_URL, json=payload)
    data = res.json()
    return data.get("response", f"Error: {data}")




