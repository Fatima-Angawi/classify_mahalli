from sentence_transformers import SentenceTransformer
import numpy as np

DEFAULT_MODEL = "all-MiniLM-L6-v2"

class Embedder:
    def __init__(self, model_name: str = DEFAULT_MODEL):
        self.model = SentenceTransformer(model_name)

    def encode(self, texts: list[str]):
        if isinstance(texts, str):
            texts = [texts]
        embeddings = self.model.encode(
            texts,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return np.array(embeddings)
