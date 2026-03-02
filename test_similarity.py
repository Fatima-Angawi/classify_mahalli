from app.data.loader import load_dataset
from app.embeddings.embedder import Embedder


df = load_dataset("data/combined_data.csv")
sample_texts = df["text"].head(40).astype(str).tolist()

embedder = Embedder()
sim_df = embedder.get_similarity_matrix(sample_texts)

print(sim_df.iloc[:5, :5]) 

