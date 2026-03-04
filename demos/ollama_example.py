import requests
import json

def send_ollama_request(prompt, model="llama3.2"):
    """
    Send a request to Ollama using the HTTP API
    """
    url = "http://localhost:11434/api/generate"
    
    data = {
        "model": model,
        "prompt": prompt,
        "stream": False
    }
    
    try:
        response = requests.post(url, json=data)
        response.raise_for_status()
        
        result = response.json()
        return result["response"]
    
    except requests.exceptions.RequestException as e:
        print(f"Error making request: {e}")
        return None

def stream_ollama_request(prompt, model="llama3.2"):
    """
    Send a streaming request to Ollama
    """
    url = "http://localhost:11434/api/generate"
    
    data = {
        "model": model,
        "prompt": prompt,
        "stream": True
    }
    
    try:
        response = requests.post(url, json=data, stream=True)
        response.raise_for_status()
        
        for line in response.iter_lines():
            if line:
                chunk = json.loads(line.decode('utf-8'))
                print(chunk["response"], end="", flush=True)
                
                if chunk.get("done", False):
                    break
        print()  # New line at the end
        
    except requests.exceptions.RequestException as e:
        print(f"Error making streaming request: {e}")

if __name__ == "__main__":
    # Example 1: Simple request
    print("=== Simple Request ===")
    response = send_ollama_request("What is the capital of Japan?")
    print(f"Response: {response}")
    
    print("\n=== Streaming Request ===")
    # Example 2: Streaming request
    stream_ollama_request("Write a short poem about coding") 