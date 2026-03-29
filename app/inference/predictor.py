import torch
import numpy as np

class predictor:
    def __init__(self, model, tokenizer, threshold):
        self.model     = model
        self.tokenizer = tokenizer
        self.threshold = threshold

    def predict_with_tier(self, texts: list[str]):
        probs = self.predict(texts)  # تستخدم predict الموجود
        preds = (probs >= self.threshold).astype(int)
        tiers = [self._tier(p) for p in probs]
        return preds, probs, tiers

    def _tier(self, prob: float) -> str:
        if prob >= self.threshold:
            return "AUTO_REMOVE"
        elif prob >= 0.5:
            return "HUMAN_REVIEW"
        else:
            return "CLEAR"
    
    def predict(self, texts: list[str]) -> np.ndarray:
        self.model.eval()

        all_lengths = [len(self.tokenizer.encode(t)) for t in texts]
        max_len = min(int(np.mean(all_lengths) + 2*np.std(all_lengths)), 512)

        inputs = self.tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_len,
        return_tensors="pt",
     ).to(self.model.device)

        with torch.no_grad():
           outputs = self.model(**inputs)
           probs   = torch.softmax(outputs.logits, dim=-1)[:, 1].cpu().numpy()

        return probs