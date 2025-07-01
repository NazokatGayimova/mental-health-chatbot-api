from flask import Flask, request, jsonify
from retrieval import ResponseRetriever

app = Flask(__name__)
retriever = ResponseRetriever()

@app.route("/")
def home():
    return jsonify({"message": "API is running. Use /predict"})

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    user_input = data.get("message", "").strip()

    if not user_input:
        return jsonify({"error": "Missing 'message' in request body."}), 400

    response = retriever.get_best_response(user_input)
    return jsonify({"response": response})

if __name__ == "__main__":
    print("🚀 Starting Flask server...")
    app.run(debug=True)

