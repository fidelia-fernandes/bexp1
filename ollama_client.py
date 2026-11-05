"""
ollama_client.py
================
Unified Ollama connectivity, model health check, and execution wrapper for ECommerceGPT.
Distinguishes between connection errors, missing model errors, and generation errors.
Never silently fabricates responses.
"""

import urllib.request
import json

try:
    import ollama
    OLLAMA_LIB_AVAILABLE = True
except ImportError:
    OLLAMA_LIB_AVAILABLE = False


def check_ollama_daemon(host="http://127.0.0.1:11434"):
    """
    Check if the Ollama daemon server is reachable.
    Returns: (is_online: bool, message: str)
    """
    try:
        req = urllib.request.Request(f"{host}/api/version", method="GET")
        with urllib.request.urlopen(req, timeout=2.0) as resp:
            if resp.status == 200:
                data = json.loads(resp.read().decode())
                return True, f"Ollama daemon online (v{data.get('version', 'unknown')})"
    except Exception as e:
        return False, f"Could not connect to Ollama daemon on {host}. Error: {e}"
    return False, f"Could not reach Ollama daemon on {host}."


def check_model_available(model_name="qwen2.5:0.5b"):
    """
    Check if a specific model has been pulled and is available locally.
    Returns: (is_available: bool, message: str)
    """
    online, msg = check_ollama_daemon()
    if not online:
        return False, msg

    try:
        models_data = ollama.list()
        # Handle dict or object representation in ollama-python
        models_list = []
        if isinstance(models_data, dict) and "models" in models_data:
            models_list = [m.get("model", m.get("name", "")) for m in models_data["models"]]
        elif hasattr(models_data, "models"):
            models_list = [getattr(m, "model", getattr(m, "name", "")) for m in models_data.models]

        # Match model name or tag
        matched = any(model_name in m or m in model_name for m in models_list)
        if matched:
            return True, f"Model '{model_name}' is available."
        return False, f"Model '{model_name}' not found locally. Available models: {models_list}. Run 'ollama pull {model_name}'."
    except Exception as e:
        return False, f"Error querying Ollama model list: {e}"


def safe_chat(model="qwen2.5:0.5b", prompt=None, messages=None, temperature=0.2):
    """
    Execute chat generation with explicit error reporting.
    Returns: (success: bool, content_or_error: str)
    """
    if not OLLAMA_LIB_AVAILABLE:
        return False, "[Error: 'ollama' python package is not installed. Run 'pip install ollama']"

    daemon_ok, daemon_msg = check_ollama_daemon()
    if not daemon_ok:
        return False, f"[Ollama Offline: {daemon_msg}. Ensure 'ollama serve' is running.]"

    if messages is None:
        if prompt is None:
            return False, "[Error: No prompt provided to safe_chat]"
        messages = [{"role": "user", "content": prompt}]

    try:
        resp = ollama.chat(
            model=model,
            messages=messages,
            options={"temperature": temperature}
        )
        content = resp["message"]["content"].strip()
        return True, content
    except Exception as e:
        err_str = str(e)
        if "connection" in err_str.lower() or "dial tcp" in err_str.lower():
            return False, f"[Ollama Connection Error: Failed to reach Ollama server. {err_str}]"
        elif "not found" in err_str.lower():
            return False, f"[Ollama Model Error: Model '{model}' not found. Run 'ollama pull {model}']"
        else:
            return False, f"[Ollama Generation Error: {err_str}]"


def safe_embeddings(model="nomic-embed-text", text=""):
    """
    Generate dense embedding vector with explicit error reporting.
    Returns: (success: bool, vector_or_error)
    """
    if not OLLAMA_LIB_AVAILABLE:
        return False, "[Error: 'ollama' python package is not installed]"

    daemon_ok, daemon_msg = check_ollama_daemon()
    if not daemon_ok:
        return False, f"[Ollama Offline: {daemon_msg}]"

    try:
        resp = ollama.embeddings(model=model, prompt=text)
        return True, resp["embedding"]
    except Exception as e:
        return False, f"[Ollama Embedding Error: {e}]"
