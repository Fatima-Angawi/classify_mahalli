import numpy as np
from app.embeddings.embedder import Embedder
from app.models.predictor import Predictor

text = ["اشتراك نتفلكس مدى الحياة 20 ريال"]

embedder = Embedder()
predictor = Predictor()

embedding = embedder.encode(text)
prediction = predictor.predict(np.array(embedding))

print(prediction)