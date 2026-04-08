import json
import os

import numpy as np
import pandas as pd
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.processors import TemplateProcessing
from tokenizers.trainers import BpeTrainer


DEFAULT_TOK_PATH = "checkpoints/deberta_commonsense_tokenizer.json"
DEFAULT_META_PATH = "checkpoints/deberta_commonsense_tokenizer_meta.json"


def train_bbpe_tokenizer_from_train_df(
    train_df: pd.DataFrame,
    out_path: str = DEFAULT_TOK_PATH,
    meta_path: str = DEFAULT_META_PATH,
    vocab_size: int = 12000,
    min_freq: int = 2,
):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    os.makedirs(os.path.dirname(meta_path), exist_ok=True)

    texts = []
    for _, row in train_df.iterrows():
        false_sentence = str(row["FalseSent"])
        for option in ["A", "B", "C"]:
            texts.append(f"{false_sentence} {str(row[f'Option{option}'])}")

    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=True)

    trainer = BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_freq,
        special_tokens=["<pad>", "<unk>", "<sep>", "<cls>", "<eos>"],
    )
    tokenizer.train_from_iterator(texts, trainer=trainer)

    cls_id = tokenizer.token_to_id("<cls>")
    sep_id = tokenizer.token_to_id("<sep>")
    eos_id = tokenizer.token_to_id("<eos>")
    tokenizer.post_processor = TemplateProcessing(
        single="<cls> $A <eos>",
        pair="<cls> $A <sep> $B <eos>",
        special_tokens=[("<cls>", cls_id), ("<sep>", sep_id), ("<eos>", eos_id)],
    )
    tokenizer.save(out_path)

    meta = {
        "vocab_size": int(tokenizer.get_vocab_size()),
        "requested_vocab_size": int(vocab_size),
        "min_freq": int(min_freq),
        "special_tokens": ["<pad>", "<unk>", "<sep>", "<cls>", "<eos>"],
        "pad_id": int(tokenizer.token_to_id("<pad>")),
        "unk_id": int(tokenizer.token_to_id("<unk>")),
        "cls_id": int(cls_id),
        "sep_id": int(sep_id),
        "eos_id": int(eos_id),
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)


def load_bbpe(tokenizer_path: str = DEFAULT_TOK_PATH, meta_path: str = DEFAULT_META_PATH):
    tokenizer = Tokenizer.from_file(tokenizer_path)
    with open(meta_path, "r") as f:
        meta = json.load(f)
    return tokenizer, meta


def ensure_bbpe(
    train_df: pd.DataFrame,
    tokenizer_path: str = DEFAULT_TOK_PATH,
    meta_path: str = DEFAULT_META_PATH,
    vocab_size: int = 12000,
    min_freq: int = 2,
):
    if not os.path.exists(tokenizer_path) or not os.path.exists(meta_path):
        train_bbpe_tokenizer_from_train_df(
            train_df=train_df,
            out_path=tokenizer_path,
            meta_path=meta_path,
            vocab_size=vocab_size,
            min_freq=min_freq,
        )


def encode_grouped_bbpe(df: pd.DataFrame, tok: Tokenizer, max_len: int, pad_id: int):
    tok.enable_truncation(max_length=max_len)
    tok.enable_padding(length=max_len, pad_id=pad_id, pad_token="<pad>")

    pairs = []
    for _, row in df.iterrows():
        false_sentence = str(row["FalseSent"])
        pairs.append((false_sentence, str(row["OptionA"])))
        pairs.append((false_sentence, str(row["OptionB"])))
        pairs.append((false_sentence, str(row["OptionC"])))

    encodings = tok.encode_batch(pairs)
    x = np.full((len(df), 3, max_len), pad_id, dtype=np.int64)
    for i, encoding in enumerate(encodings):
        x[i // 3, i % 3, :] = np.asarray(encoding.ids, dtype=np.int64)
    return x
