# DeBertaCommonSense

Official implementation of **DeBertaCommonSense**, a grouped BBPE commonsense reasoning model with DeBERTa-inspired disentangled attention.

Hugging Face: https://huggingface.co/owenarink/attentiontypes-commonsense

## Installation

```bash
conda create -n deberta-commonsense python=3.11
conda activate deberta-commonsense
pip install -r requirements.txt
```

## Training

```bash
python train.py
```

## Prediction

```bash
python predict.py
```

The default prediction file is written to `submits/submission_deberta_commonsense.csv`.

## Citation

```bibtex
@misc{arink2026debertacommonsense,
  title={DeBertaCommonSense},
  author={Arink, Owen},
  year={2026}
}
```
