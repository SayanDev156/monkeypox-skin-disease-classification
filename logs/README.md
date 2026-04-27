# Training Logs Directory

This directory contains training logs, metrics, and visualizations.

## Contents

- `training_log_*.csv` - CSV logs of training metrics
- `confusion_matrix.png` - Confusion matrix visualization
- `training_history.png` - Training/validation curves
- TensorBoard logs in subdirectories

## TensorBoard

To view training logs in TensorBoard:

```bash
tensorboard --logdir=logs
```

Then open http://localhost:6006 in your browser.

## Metrics Tracked

- Loss
- Accuracy
- Precision
- Recall
- AUC
- Learning Rate

