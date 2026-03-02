from xgboost import XGBClassifier
from app.config import MODEL_PATH

class Predictor:
    def __init__(self):
        self.model = XGBClassifier()
        self.model.load_model(MODEL_PATH)

    def predict(self, embeddings):
        return self.model.predict(embeddings)