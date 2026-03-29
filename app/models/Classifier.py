from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    DataCollatorWithPadding
)
import os
os.environ["PYTHONUTF8"] = "1"
from arabert import ArabertPreprocessor
from datasets import Dataset
from sklearn.metrics import f1_score
import numpy as np
import torch
from config import MODEL_NAME
import torch.nn as nn
import os
class TextClassifier:

    def __init__(self):
        self.prep      =  ArabertPreprocessor(model_name=MODEL_NAME, apply_farasa_segmentation=False)
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.model     = None   # يتحمل بعد fine_tune()

    def preprocess(self, text: str) -> str:
        return self.prep.preprocess(text)
    

    # ── 2. Fine-tune (end-to-end) ─────────────
    def fine_tune(self, train_df, val_df, class_weights_tensor):
        self.model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME,
            num_labels=2,
            ignore_mismatched_sizes=True,
        )

        def tokenize_fn(batch):
            all_lengths = [len(self.tokenizer.encode(t)) for t in batch["text"]]
            # Set max_len to mean + 2*std, capped at 512 to avoid OOM
            max_len = 512

            return self.tokenizer(
                batch["text"],
                #I didnt set padding to max len because it causes OOM with long texts. Instead, we will pad to the longest in the batch using DataCollatorWithPadding.
                padding="longest",
                truncation=True,
                max_length=max_len,
    )

        def to_ds(df):
            ds = Dataset.from_pandas(df[["text", "label"]].reset_index(drop=True))
            ds = ds.map(tokenize_fn, batched=True)
            ds = ds.rename_column("label", "labels")
            ds.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
            return ds

        train_ds = to_ds(train_df)
        val_ds   = to_ds(val_df)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        weights = class_weights_tensor.to(device)
        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)

        class WeightedTrainer(Trainer):
            def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
                labels  = inputs.pop("labels")
                outputs = model(**inputs)
                #handle class imbalance with weighted cross-entropy
                loss    = nn.CrossEntropyLoss(weight=weights)(outputs.logits, labels)
                return (loss, outputs) if return_outputs else loss

        def compute_metrics(eval_pred):
            preds  = np.argmax(eval_pred.predictions, axis=-1)
            labels = eval_pred.label_ids
            return {"f1_scam": f1_score(labels, preds, pos_label=1)}

        args = TrainingArguments(
            output_dir="artifacts/finetune",
            num_train_epochs=3,
            #internal data loader
            
            per_device_train_batch_size=16,
            per_device_eval_batch_size=32,
            learning_rate=2e-5,
            warmup_ratio=0.1,
            weight_decay=0.01,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="f1_scam",
            greater_is_better=True,
            fp16=torch.cuda.is_available(),
            seed=42,
        )

        WeightedTrainer(
            model=self.model,
            args=args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
                   
            data_collator=data_collator,
            compute_metrics=compute_metrics,
            #early stopping if no improvement in f1_scam for 2 consecutive evals
            callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
        ).train()

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
    
    def predict(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
    self.model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    self.model.to(device)
    
    all_probs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        inputs = self.tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        ).to(device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)[:, 1].cpu().numpy()
        
        all_probs.append(probs)
        torch.cuda.empty_cache()  # ← important
    
    return np.concatenate(all_probs)




    # ── 4. Save / Load ────────────────────────────────────
    def save(self, path: str = "artifacts/AraBERT"):
        os.makedirs(path, exist_ok=True)
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

    def load(self, path: str = "artifacts/AraBERT"):
        os.makedirs(path, exist_ok=True)
        self.model     = AutoModelForSequenceClassification.from_pretrained(path)
        self.tokenizer = AutoTokenizer.from_pretrained(path)
