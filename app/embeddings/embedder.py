import pandas as pd
from sentence_transformers import SentenceTransformer, util
from app.config import MODEL_NAME

class Embedder:
    def __init__(self):
        self.model = SentenceTransformer(MODEL_NAME)

    def encode(self, texts, batch_size=32):
        return self.model.encode(texts, batch_size=batch_size, show_progress_bar=True)
    
    def get_similarity_matrix(self, texts_list):

        embeddings = self.encode(texts_list)

        sim_matrix = util.cos_sim(embeddings, embeddings)
        
        short_labels = [str(t)[:30] + "..." for t in texts_list]
        df_sim = pd.DataFrame(sim_matrix.cpu().numpy(), index=short_labels, columns=short_labels)
        
        return df_sim

