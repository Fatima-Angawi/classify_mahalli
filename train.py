import numpy as np
from app.data.loader import load_dataset
from app.embeddings.embedder import Embedder
from app.models.trainer import Trainer
from app.models.evaluator import evaluate

# تحميل البيانات
df = load_dataset("data/combined_data.csv")

# توليد embeddings
embedder = Embedder()
embeddings = embedder.encode(df["text"].astype(str).tolist())

X = np.array(embeddings)
y = df["label"].values

# تدريب
trainer = Trainer()
X_test, y_test = trainer.train(X, y)

# تقييم
evaluate(trainer.model, X_test, y_test)

# حفظ
trainer.save()