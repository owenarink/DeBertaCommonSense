import argparse
import json
import os

import pandas as pd
import torch

from models.bert import BertTransformer
from processing import preprocess
from tokenizer_bbpe import encode_grouped_bbpe, load_bbpe


OPTIONS = ["A", "B", "C"]


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
def predict(model, x, device, batch_size=128):
    model.eval()
    preds = []
    for start in range(0, len(x), batch_size):
        end = min(start + batch_size, len(x))
        xb = torch.tensor(x[start:end], dtype=torch.long, device=device)
        batch, choices, length = xb.shape
        scores = model(xb.view(batch * choices, length)).view(batch, choices)
        preds.extend(scores.argmax(dim=1).cpu().numpy().tolist())
    return [OPTIONS[i] for i in preds]


def main():
    parser = argparse.ArgumentParser(description="Create a DeBertaCommonSense submission.")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--checkpoint", default="checkpoints/deberta_commonsense.pt")
    parser.add_argument("--hparams", default="checkpoints/deberta_commonsense_hparams.json")
    parser.add_argument("--output", default="submits/submission_deberta_commonsense.csv")
    args = parser.parse_args()

    with open(args.hparams, "r") as f:
        hparams = json.load(f)

    train_raw = pd.read_csv(os.path.join(args.data_dir, "train_data.csv"))
    test_raw = pd.read_csv(os.path.join(args.data_dir, "test_data.csv"))
    answers_raw = pd.read_csv(os.path.join(args.data_dir, "train_answers.csv"))
    _, test_df = preprocess(train_raw, test_raw, answers_raw)

    tokenizer_path = hparams["tokenizer_path"]
    tokenizer_meta_path = hparams["tokenizer_meta_path"]
    tokenizer, meta = load_bbpe(tokenizer_path, tokenizer_meta_path)
    x_test = encode_grouped_bbpe(
        test_df,
        tokenizer,
        max_len=int(hparams["grouped_max_len"]),
        pad_id=int(hparams.get("pad_idx", meta.get("pad_id", 0))),
    )

    device = get_device()
    model = build_model(hparams, meta).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    answers = predict(model, x_test, device=device)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    pd.DataFrame({"id": test_df["id"].astype(str).tolist(), "answer": answers}).to_csv(args.output, index=False)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
