from flask import Flask, request, jsonify
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

app = Flask(__name__)

# Global model/tokenizer
tokenizer = None
model = None
model_name = "microsoft/DialoGPT-small"

@app.route("/")
def home():
    return jsonify({"message": "API is running. Use /load_model and /predict"})

@app.route("/load_model", methods=["GET"])
def load_model():
    global tokenizer, model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    return jsonify({"status": "Model loaded successfully"})

@app.route("/predict", methods=["POST"])
def predict():
    global tokenizer, model
    if tokenizer is None or model is None:
        return jsonify({"error": "Model not loaded. Call /load_model first."}), 400

    data = request.get_json()
    user_input = data.get("message", "")

    if not user_input:
        return jsonify({"error": "Missing 'message' in request body."}), 400

    input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors="pt")
    chat_history_ids = model.generate(input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)
    response = tokenizer.decode(chat_history_ids[:, input_ids.shape[-1]:][0], skip_special_tokens=True)

    return jsonify({"response": response})
