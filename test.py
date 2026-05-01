import os
from google import genai

# Make sure your API key is in the environment
os.environ["GEMINI_API_KEY"] = "AIzaSyD7s1amUzk43rfQyDuGQw1ZmB1jkG2vNNk"

client = genai.Client()

for model in client.models.list():
    print(f"Model Name: {model.name}")
    print(f"Supported Actions: {model.supported_actions}")
    print("-" * 40)