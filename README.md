# 🧠 Mental Health Chatbot API

This project is a simple Flask API that serves a mental health chatbot built using the `DialoGPT-small` model from Hugging Face.

---

## 🚀 Endpoints

- `GET /load_model`  
  Loads the pre-trained model into memory.

- `POST /predict`  
  Send a JSON message like this:
  ```json
  {
    "message": "I feel anxious lately."
  }
