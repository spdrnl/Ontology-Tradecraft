import requests
from pathlib import Path
import pprint
import json

PROJECT_ROOT = Path(__file__).parent.parent
path = PROJECT_ROOT / Path("prompts/generate_candidates_prompts.md")
query = path.read_text()
url = "http://localhost:11434/api/generate"
data = {
    "model": "gemma3n",
    "prompt": query,
    "stream": False
}

response = requests.post(url, json=data)
response_dict = json.loads(response.text)
print(response_dict['response'])