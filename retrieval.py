import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, util
import torch

class ResponseRetriever:
    def __init__(self):
        print("📄 Loading cleaned data and precomputed embeddings...")
        self.df = pd.read_csv("data_cleaned.csv")
        self.embeddings = np.load("situation_embeddings.npy")
        self.embeddings = torch.tensor(self.embeddings, dtype=torch.float32).to("cpu")

        self.model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
        print("✅ Model and data ready.")

    def keyword_fallback(self, message):
        message = message.lower()
        if any(word in message for word in ["sad", "unhappy", "depressed", "lost", "cry"]):
            return "I'm sorry you're feeling that way. You're not alone — would you like to talk about it?"
        elif any(word in message for word in ["happy", "joy", "excited", "grateful", "great"]):
            return "That's wonderful to hear! What made you feel this way?"
        elif any(word in message for word in ["angry", "mad", "frustrated", "upset"]):
            return "That sounds tough. Do you want to share what made you feel this way?"
        elif any(word in message for word in ["anxious", "worried", "nervous", "scared"]):
            return "Anxiety can be really hard. I'm here to listen if you'd like to talk."
        elif any(word in message for word in ["love", "relationship", "crush", "heart"]):
            return "Relationships can be very emotional. I'm here if you need someone to talk to."
        elif any(word in message for word in ["dog", "cat", "pet", "animal", "puppy", "kitten"]):
            return "That sounds like a special moment! Pets bring us so much joy."
        else:
            return "I'm here for you. Please tell me more."

    def get_best_response(self, user_input):
        user_input_lower = user_input.lower()
        query_embedding = self.model.encode(user_input_lower, convert_to_tensor=True).to("cpu")

        similarities = util.cos_sim(query_embedding, self.embeddings)[0]
        best_index = similarities.argmax().item()
        best_score = similarities[best_index].item()

        print(f"🔍 Input: {user_input_lower}")
        print(f"📊 Best similarity score: {best_score:.4f}")
        print(f"📌 Best match index: {best_index}")

        try:
            matched_situation = self.df.iloc[best_index]["Situation"]
            best_response = self.df.iloc[best_index]["empathetic_dialogues"]
            print(f"📌 Matched situation: {matched_situation}")
            print(f"🧠 Raw response: {best_response}")
        except Exception as e:
            print(f"❌ Error accessing matched row: {e}")
            return self.keyword_fallback(user_input_lower)

        if best_score < 0.3 or pd.isna(best_response) or "Agent :" not in str(best_response):
            return self.keyword_fallback(user_input_lower)

        parts = str(best_response).split("Agent :")
        final = parts[1].strip() if len(parts) > 1 and parts[1].strip() else self.keyword_fallback(user_input_lower)
        return final

