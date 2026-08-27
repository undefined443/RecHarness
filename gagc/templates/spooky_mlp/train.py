"""Spooky Author Identification — TF-IDF + MLP baseline.

3-class text classification (EAP/HPL/MWS). Minimize multi-class log loss.

Key hyper-parameters (all tunable by RecHarness):
  MAX_FEATURES : TF-IDF vocabulary cap
  NGRAM_MIN/MAX: TF-IDF n-gram range
  MIN_DF/MAX_DF: TF-IDF document-frequency cutoffs
  HIDDEN_DIM   : MLP hidden width
  DROPOUT      : MLP dropout rate
  LR           : AdamW learning rate
  WEIGHT_DECAY : AdamW weight decay
  BATCH_SIZE   : training batch size
  EPOCHS       : total training epochs
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import log_loss as sklearn_log_loss
from torch import optim
from torch.utils.data import DataLoader, TensorDataset

# ── Hyper-parameters ──────────────────────────────────────────────────
MAX_FEATURES = int(os.environ.get("SPOOKY_MAX_FEATURES", "10000"))
NGRAM_MIN    = int(os.environ.get("SPOOKY_NGRAM_MIN",     "1"))
NGRAM_MAX    = int(os.environ.get("SPOOKY_NGRAM_MAX",     "2"))
MIN_DF       = int(os.environ.get("SPOOKY_MIN_DF",        "2"))
MAX_DF       = float(os.environ.get("SPOOKY_MAX_DF",      "0.95"))
HIDDEN_DIM   = int(os.environ.get("SPOOKY_HIDDEN_DIM",    "256"))
DROPOUT      = float(os.environ.get("SPOOKY_DROPOUT",     "0.3"))
LR           = float(os.environ.get("SPOOKY_LR",          "1e-3"))
WEIGHT_DECAY = float(os.environ.get("SPOOKY_WEIGHT_DECAY","1e-4"))
BATCH_SIZE   = int(os.environ.get("SPOOKY_BATCH_SIZE",    "256"))
EPOCHS       = int(os.environ.get("SPOOKY_EPOCHS",        "20"))

CLASSES = ["EAP", "HPL", "MWS"]

TRAIN_DATA = os.environ.get("SPOOKY_TRAIN_DATA", "./input/spooky_author/train.csv")
VAL_DATA   = os.environ.get("SPOOKY_VAL_DATA",   "./input/spooky_author/val.csv")
MODEL_OUT  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoint.pt")


class SpookyClassifier(nn.Module):
    """TF-IDF features -> 2-layer MLP -> (N, 3) logits."""

    def __init__(self, input_dim: int, num_classes: int = 3,
                 hidden_dim: int = 256, dropout: float = 0.3):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)  # logits (N, 3)


def create_model(input_dim: int, device: torch.device) -> SpookyClassifier:
    return SpookyClassifier(
        input_dim=input_dim, num_classes=len(CLASSES),
        hidden_dim=HIDDEN_DIM, dropout=DROPOUT,
    ).to(device)


def _evaluate(model, vectorizer, texts, labels, device) -> float:
    model.eval()
    with torch.no_grad():
        X = vectorizer.transform(texts)
        X_tensor = torch.from_numpy(X.toarray()).float().to(device)
        logits = model(X_tensor)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
    probs = probs / probs.sum(axis=1, keepdims=True)  # guard against float32 rounding
    y_true = [CLASSES.index(a) for a in labels]
    return float(sklearn_log_loss(y_true, probs, labels=list(range(len(CLASSES)))))


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_df = pd.read_csv(TRAIN_DATA)
    texts = train_df["text"].values
    labels = train_df["author"].values
    print(f"Train data: {len(train_df)} samples")
    print(f"Author distribution: {dict(zip(*np.unique(labels, return_counts=True)))}")

    vectorizer = TfidfVectorizer(
        max_features=MAX_FEATURES,
        ngram_range=(NGRAM_MIN, NGRAM_MAX),
        min_df=MIN_DF,
        max_df=MAX_DF,
        sublinear_tf=True,
    )
    X = vectorizer.fit_transform(texts)
    input_dim = X.shape[1]
    print(f"TF-IDF dimension: {input_dim}")

    y = np.array([CLASSES.index(a) for a in labels])
    X_tensor = torch.from_numpy(X.toarray()).float()
    y_tensor = torch.from_numpy(y).long()
    dataloader = DataLoader(
        TensorDataset(X_tensor, y_tensor), batch_size=BATCH_SIZE, shuffle=True,
    )

    model = create_model(input_dim, device)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion = nn.CrossEntropyLoss()

    val_df = pd.read_csv(VAL_DATA) if os.path.isfile(VAL_DATA) else None

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        total = 0
        for batch_X, batch_y in dataloader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            logits = model(batch_X)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch_X.size(0)
            total += batch_X.size(0)
        scheduler.step()
        avg_loss = total_loss / total
        print(f"Epoch {epoch + 1}/{EPOCHS} - train_loss: {avg_loss:.4f} "
              f"lr: {optimizer.param_groups[0]['lr']:.6f}")

        if val_df is not None:
            val_log_loss = _evaluate(model, vectorizer, val_df["text"].values,
                                      val_df["author"].values, device)
            val_score = max(0.0, 1.0 - val_log_loss / __import__("math").log(3))
            print(f"  val_log_loss: {val_log_loss:.4f}  val_score: {val_score:.4f}")

    torch.save(
        {"model_state_dict": model.state_dict(), "vectorizer": vectorizer, "input_dim": input_dim},
        MODEL_OUT,
    )
    print(f"Checkpoint saved to: {MODEL_OUT}")

    if val_df is not None:
        final_val_log_loss = _evaluate(model, vectorizer, val_df["text"].values,
                                        val_df["author"].values, device)
        print(f"LOGLOSS={final_val_log_loss:.6f}")


if __name__ == "__main__":
    main()
