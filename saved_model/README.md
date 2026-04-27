# Saved Models Directory

This directory contains the trained model files.

The current exported model is `monkeypox_vit.keras` with metadata stored in `metadata.json`.

## How to Save Models

After training, the model will be saved here as:
- `monkeypox_vit.keras` - Complete model in Keras format

## Loading the Model

```python
import tensorflow as tf
model = tf.keras.models.load_model('saved_model/monkeypox_vit.keras')
```

## Model Files

- `monkeypox_vit.keras` - Final trained model
- `metadata.json` - Saved class names, image size, and training history summary

