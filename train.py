import argparse
import json
import math
import os
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from models.bert import BertTransformer
from processing import preprocess
from tokenizer_bbpe import encode_grouped_bbpe, ensure_bbpe, load_bbpe


IDX2LABEL = {0: "A", 1: "B", 2: "C"}
LABEL2IDX = {"A": 0, "B": 1, "C": 2}


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_warmup_cosine_scheduler(optimizer, warmup_steps, total_steps, min_lr_ratio=0.05):
    def lr_lambda(step: int):
        step = min(step, total_steps)
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def token_dropout(x_ids: torch.Tensor, pad_id: int, unk_id: int, p: float, protected_ids=()):
    if p <= 0:
        return x_ids
    x = x_ids.clone()
    mask = x != pad_id
    for protected_id in protected_ids:
        mask &= x != int(protected_id)
    mask &= torch.rand_like(x.float()) < p
    x[mask] = unk_id
    return x


def grouped_pairwise_hinge_loss(scores: torch.Tensor, targets: torch.Tensor, margin: float = 0.20):
    gold = scores.gather(1, targets.unsqueeze(1))
    margins = margin - gold + scores
    non_gold = torch.ones_like(scores, dtype=torch.bool)
    non_gold.scatter_(1, targets.unsqueeze(1), False)
    return torch.relu(margins[non_gold]).mean()


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

    return correct / max(1, total), total_loss / max(1, total)


def build_model(vocab_size: int, pad_id: int, args):
    return BertTransformer(
        vocab_size=vocab_size,
        num_classes=1,
        pad_idx=pad_id,
        model_dim=args.model_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        ff_mult=args.ff_mult,
        dropout=args.dropout,
        max_len=args.model_max_len,
        pooling=args.pooling,
    )


def main():
    parser = argparse.ArgumentParser(description="Train DeBertaCommonSense.")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    parser.add_argument("--seed", type=int, default=40)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-len", type=int, default=128)
    parser.add_argument("--model-max-len", type=int, default=512)
    parser.add_argument("--vocab-size", type=int, default=12000)
    parser.add_argument("--min-freq", type=int, default=2)
    parser.add_argument("--model-dim", type=int, default=256)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--ff-mult", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.30)
    parser.add_argument("--pooling", choices=["mean", "cls"], default="mean")
    parser.add_argument("--lr", type=float, default=6e-5)
    parser.add_argument("--weight-decay", type=float, default=5e-2)
    parser.add_argument("--token-dropout", type=float, default=0.10)
    parser.add_argument("--hinge-weight", type=float, default=0.35)
    parser.add_argument("--hinge-margin", type=float, default=0.20)
    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    train_raw = pd.read_csv(os.path.join(args.data_dir, "train_data.csv"))
    test_raw = pd.read_csv(os.path.join(args.data_dir, "test_data.csv"))
    answers_raw = pd.read_csv(os.path.join(args.data_dir, "train_answers.csv"))
    train_df, _ = preprocess(train_raw, test_raw, answers_raw)

    train_rows, val_rows = train_test_split(
        train_df,
        test_size=0.2,
        random_state=args.seed,
        stratify=train_df["label"].astype(str),
    )

    tokenizer_path = os.path.join(args.checkpoint_dir, "deberta_commonsense_tokenizer.json")
    tokenizer_meta_path = os.path.join(args.checkpoint_dir, "deberta_commonsense_tokenizer_meta.json")
    ensure_bbpe(
        train_df=train_df,
        tokenizer_path=tokenizer_path,
        meta_path=tokenizer_meta_path,
        vocab_size=args.vocab_size,
        min_freq=args.min_freq,
    )
    tokenizer, meta = load_bbpe(tokenizer_path, tokenizer_meta_path)
    pad_id = int(meta["pad_id"])
    unk_id = int(meta["unk_id"])

    x_train = encode_grouped_bbpe(train_rows, tokenizer, max_len=args.max_len, pad_id=pad_id)
    x_val = encode_grouped_bbpe(val_rows, tokenizer, max_len=args.max_len, pad_id=pad_id)
    y_train = np.array([LABEL2IDX[x] for x in train_rows["label"].astype(str).str.strip()], dtype=np.int64)
    y_val = np.array([LABEL2IDX[x] for x in val_rows["label"].astype(str).str.strip()], dtype=np.int64)

    device = get_device()
    model = build_model(tokenizer.get_vocab_size(), pad_id, args).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.10)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    train_loader = DataLoader(
        TensorDataset(torch.tensor(x_train).long(), torch.tensor(y_train).long()),
        batch_size=args.batch_size,
        shuffle=True,
    )
    schedule_epochs = min(40, args.epochs)
    total_steps = max(1, schedule_epochs * len(train_loader))
    scheduler = make_warmup_cosine_scheduler(optimizer, int(0.05 * total_steps), total_steps)

    hparams = {
        "seed": args.seed,
        "pad_idx": pad_id,
        "model_dim": args.model_dim,
        "num_heads": args.num_heads,
        "num_layers": args.num_layers,
        "ff_mult": args.ff_mult,
        "dropout": args.dropout,
        "max_len": args.model_max_len,
        "pooling": args.pooling,
        "grouped_max_len": args.max_len,
        "bbpe_vocab_size": args.vocab_size,
        "bbpe_min_freq": args.min_freq,
        "tokenizer_path": tokenizer_path,
        "tokenizer_meta_path": tokenizer_meta_path,
        "vocab_size": tokenizer.get_vocab_size(),
        "hinge_weight": args.hinge_weight,
        "hinge_margin": args.hinge_margin,
    }
    hparams_path = os.path.join(args.checkpoint_dir, "deberta_commonsense_hparams.json")
    with open(hparams_path, "w") as f:
        json.dump(hparams, f, indent=2)

    best_acc = -1.0
    best_loss = float("inf")
    bad_epochs = 0
    best_path = os.path.join(args.checkpoint_dir, "deberta_commonsense.pt")
    best_acc_path = os.path.join(args.checkpoint_dir, "deberta_commonsense_best_acc.pt")
    best_loss_path = os.path.join(args.checkpoint_dir, "deberta_commonsense_best_loss.pt")

    print(f"Using device: {device}")
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        train_correct = 0
        train_total = 0

        progress = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", leave=False)
        for xb, yb in progress:
            xb = xb.to(device)
            yb = yb.to(device)
            xb = token_dropout(xb, pad_id=pad_id, unk_id=unk_id, p=args.token_dropout)

            batch, choices, length = xb.shape
            scores = model(xb.view(batch * choices, length)).view(batch, choices)
            ce_loss = criterion(scores, yb)
            hinge_loss = grouped_pairwise_hinge_loss(scores, yb, margin=args.hinge_margin)
            loss = ce_loss + args.hinge_weight * hinge_loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            running_loss += float(loss.item()) * batch
            train_correct += (scores.argmax(dim=1) == yb).sum().item()
            train_total += batch
            progress.set_postfix(loss=f"{loss.item():.4f}", lr=f"{scheduler.get_last_lr()[0]:.2e}")

        val_acc, val_loss = evaluate(model, x_val, y_val, device)
        train_loss = running_loss / max(1, train_total)
        train_acc = train_correct / max(1, train_total)
        print(
            f"epoch={epoch} train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

        improved = False
        if val_acc > best_acc + 1e-6:
            best_acc = val_acc
            torch.save(model.state_dict(), best_acc_path)
            torch.save(model.state_dict(), best_path)
            improved = True
        if val_loss < best_loss - 1e-4:
            best_loss = val_loss
            torch.save(model.state_dict(), best_loss_path)
            improved = True

        bad_epochs = 0 if improved else bad_epochs + 1
        if bad_epochs >= args.patience:
            print(f"Early stopping after {epoch} epochs. best_acc={best_acc:.4f} best_loss={best_loss:.4f}")
            break


if __name__ == "__main__":
    main()
