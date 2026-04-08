# DeBertaCommonSense

Official implementation of **DeBertaCommonSense**, a grouped BBPE commonsense reasoning model with DeBERTa-inspired disentangled attention.

Hugging Face: https://huggingface.co/owenarink/attentiontypes-commonsense

DOI: https://doi.org/10.5281/zenodo.19471738

Model path: `owenarink/attentiontypes-commonsense`

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

## Evaluation

```bash
python eval/evaluate.py
```

## Citation

```bibtex
@misc{arink2026debertacommonsense,
  title={DeBertaCommonSense},
  author={Arink, Owen André},
  year={2026},
  doi={10.5281/zenodo.19471738},
  url={https://doi.org/10.5281/zenodo.19471738}
}
```
