import pandas as pd
import pickle
from sentence_transformers import SentenceTransformer

print("📦 Loading data and model...")
df = pd.read_csv("data.csv")
df.dropna(subset=["Situation", "empathetic_dialogues"], inplace=True)
df = df[df["Situation"].str.strip() != ""]

model = SentenceTransformer("all-MiniLM-L6-v2")

print("🔍 Encoding situations...")
embeddings = model.encode(df["Situation"].tolist(), convert_to_tensor=True, show_progress_bar=True)

print("💾 Saving embeddings as precomputed_embeddings.pkl...")
with open("precomputed_embeddings.pkl", "wb") as f:
    pickle.dump(embeddings, f)

print("💾 Overwriting cleaned data to data.csv...")
df.to_csv("data.csv", index=False)

print("✅ Embeddings and cleaned data saved successfully.")

