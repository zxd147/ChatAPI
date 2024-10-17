import requests
import json

chat_url = 'http://192.168.0.246:7861/chat/knowledge_base_chat'
data = {
  "query": "你好",
  "knowledge_base_name": "samples",
  "top_k": 3,
  "score_threshold": 0.8,
  "history": [
    {
      "role": "user",
      "content": "我们来玩成语接龙，我先来，生龙活虎"
    },
    {
      "role": "assistant",
      "content": "虎头虎脑"
    }
  ],
  "stream": False,
  "model_name": "ChatGLM-6B",
  "temperature": 0.7,
  "max_tokens": 0,
  "prompt_name": "default"
}
headers = {
    'Content-Type': 'application/json'
}
response = requests.post(chat_url, headers=headers, data=json.dumps(data))
# 检查请求是否成功
if response.status_code == 200:
    result = response.json()
    print("Response:")
    print(response.json())
else:
    result = response.json()
    print(f"Request failed with status code {response.status_code}")
    print(response.text)


