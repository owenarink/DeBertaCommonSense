import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from models.bert import BertTransformer
from processing import preprocess
from tokenizer_bbpe import encode_grouped_bbpe, load_bbpe


LABEL2IDX = {"A": 0, "B": 1, "C": 2}


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_model(hparams, meta):
    return BertTransformer(
        vocab_size=int(hparams.get("vocab_size", meta["vocab_size"])),
        num_classes=1,
        pad_idx=int(hparams.get("pad_idx", meta.get("pad_id", 0))),
        model_dim=int(hparams["model_dim"]),
        num_heads=int(hparams["num_heads"]),
        num_layers=int(hparams["num_layers"]),
        ff_mult=int(hparams["ff_mult"]),
        dropout=float(hparams["dropout"]),
        max_len=int(hparams["max_len"]),
        pooling=str(hparams["pooling"]),
    )


@torch.no_grad()
def evaluate(model, x_val, y_val, device, batch_size=128):
    model.eval()
    criterion = nn.CrossEntropyLoss(label_smoothing=0.10)
    loader = DataLoader(
        TensorDataset(torch.tensor(x_val, dtype=torch.long), torch.tensor(y_val, dtype=torch.long)),
        batch_size=batch_size,
        shuffle=False,
    )

    correct = 0
    total = 0
    total_loss = 0.0
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        batch, choices, length = xb.shape
        scores = model(xb.view(batch * choices, length)).view(batch, choices)
        loss = criterion(scores, yb)
        total_loss += float(loss.item()) * batch
        correct += (scores.argmax(dim=1) == yb).sum().item()
        total += batch

    return correct / max(1, total), total_loss / max(1, total), total


def main():
    parser = argparse.ArgumentParser(description="Evaluate DeBertaCommonSense on the validation split.")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--checkpoint", default="checkpoints/deberta_commonsense.pt")
    parser.add_argument("--hparams", default="checkpoints/deberta_commonsense_hparams.json")
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()

    with open(args.hparams, "r") as f:
        hparams = json.load(f)

    train_raw = pd.read_csv(os.path.join(args.data_dir, "train_data.csv"))
    test_raw = pd.read_csv(os.path.join(args.data_dir, "test_data.csv"))
    answers_raw = pd.read_csv(os.path.join(args.data_dir, "train_answers.csv"))
    train_df, _ = preprocess(train_raw, test_raw, answers_raw)

    seed = int(hparams.get("seed", 40))
    _, val_rows = train_test_split(
        train_df,
        test_size=0.2,
        random_state=seed,
        stratify=train_df["label"].astype(str),
    )

    tokenizer, meta = load_bbpe(hparams["tokenizer_path"], hparams["tokenizer_meta_path"])
    pad_id = int(hparams.get("pad_idx", meta.get("pad_id", 0)))
    x_val = encode_grouped_bbpe(
        val_rows,
        tokenizer,
        max_len=int(hparams["grouped_max_len"]),
        pad_id=pad_id,
    )
    y_val = np.array([LABEL2IDX[x] for x in val_rows["label"].astype(str).str.strip()], dtype=np.int64)

    device = get_device()
    model = build_model(hparams, meta).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))

    acc, loss, total = evaluate(model, x_val, y_val, device=device, batch_size=args.batch_size)
    print(f"val_acc={acc:.4f} val_loss={loss:.4f} n={total}")


if __name__ == "__main__":
    main()
