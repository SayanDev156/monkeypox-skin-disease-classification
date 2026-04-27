# Monkeypox Skin Disease Classification

End-to-end TensorFlow/Keras project for multi-class skin-image classification with a Vision Transformer backbone and a Streamlit inference app.

## Overview

This project classifies skin images into five categories:

- chickenpox
- measles
- monkeypox
- normal
- other

It includes:

- a training pipeline for real datasets or demo runs
- a reusable Vision Transformer architecture
- a Streamlit web interface for single-image inference
- saved model artifacts and training logs

## Structure

- `app.py` - Streamlit inference app
- `train_model.py` - training pipeline
- `model_architecture.py` - Vision Transformer model definitions
- `data_preprocessing.py` - dataset loading, preprocessing, synthetic dataset utility
- `requirements.txt` - dependencies
- `saved_model/` - trained model artifacts
- `logs/` - training logs

## Install

```bash
pip install -r requirements.txt
```

## Train with real dataset

Dataset layout:

```text
dataset/
  train/<class_name>/
  val/<class_name>/
  test/<class_name>/
```

Run:

```bash
python train_model.py --data-dir dataset --initial-epochs 10 --fine-tune-epochs 10
```

## Demo training

```bash
python train_model.py --demo --samples-per-class 220 --batch-size 16 --initial-epochs 8 --fine-tune-epochs 4 --lightweight-vit --no-augment
```

## Run web app

```bash
streamlit run app.py
```

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## Disclaimer

This is an educational AI project and must not be used as a substitute for professional medical diagnosis.
