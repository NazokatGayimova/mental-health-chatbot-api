# 🧠 Mental Health Support Chatbot

This project is a simple and effective mental health support chatbot powered by Flask and Streamlit. It uses real-world empathetic dialogues from the EmpatheticDialogues dataset to provide meaningful, context-aware, and compassionate responses to user inputs.

---

## 🔧 Features

- Retrieval-based chatbot using semantic similarity
- Real empathetic responses from cleaned dataset
- Fallback logical responses when dataset matches are poor
- Flask API (`/predict`) with model loading
- Streamlit user interface for interactive chat
- Precomputed sentence embeddings for fast response

---

## 📁 Project Structure
mental-health-chatbot-api/
├── api_app.py                # Flask backend
├── streamlit_app.py          # Streamlit frontend
├── retrieval.py              # Retrieval logic
├── data.csv                  # Original dataset
├── data_cleaned.csv          # Cleaned dataset used for embedding
├── generate_embeddings.py    # Script to generate embeddings
├── precomputed_embeddings.pkl # Precomputed vector file
├── requirements.txt          # Python dependencies
├── Procfile                  # For deployment (Heroku-ready)
├── .gitignore
└── venv/                     # Local virtual environment
---

## 🚀 How to Run

### 1. Clone the repository and create virtual environment
```bash
git clone https://github.com/NazokatGayimova/mental-health-chatbot-api.git
cd mental-health-chatbot-api
python3 -m venv venv
source venv/bin/activate

2. Install dependencies
pip install -r requirements.txt

3. Start the Flask API
python api_app.py

4. Run the Streamlit app (in a new terminal)
streamlit run streamlit_app.py

curl -X POST http://127.0.0.1:5000/predict \
     -H "Content-Type: application/json" \
     -d '{"message": "I feel overwhelmed"}'

🙋 Dataset Source

EmpatheticDialogues dataset by Facebook AI (https://huggingface.co/datasets/empathetic_dialogues)

MIT License
