import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from app.config import TEST_SIZE, RANDOM_STATE, MODEL_PATH
import joblib

class Trainer:
    def __init__(self):
        self.model = XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            eval_metric="logloss",
            random_state=RANDOM_STATE
        )

    def train(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            stratify=y
        )

        self.model.fit(X_train, y_train)
        return X_test, y_test

    def save(self):
        self.model.save_model(MODEL_PATH)