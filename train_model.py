from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict

import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow import keras

from data_preprocessing import DatasetLoader, build_training_pipelines, create_synthetic_dataset, get_dataset_stats
from model_architecture import create_model


@dataclass
class TrainingConfig:
    data_dir: str = "dataset"
    image_size: int = 224
    batch_size: int = 16
    initial_epochs: int = 10
    fine_tune_epochs: int = 10
    initial_lr: float = 1e-3
    fine_tune_lr: float = 1e-5
    output_dir: str = "saved_model"
    model_name: str = "monkeypox_vit.keras"
    seed: int = 42
    augment: bool = True
    use_pretrained_vit: bool = True


def set_seed(seed: int):
    tf.keras.utils.set_random_seed(seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass


def ensure_directories(output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    Path("logs").mkdir(parents=True, exist_ok=True)


def evaluate_model(model: keras.Model, test_ds: tf.data.Dataset, class_names):
    y_true = []
    y_pred = []
    for images, labels in test_ds:
        predictions = model.predict(images, verbose=0)
        y_pred.extend(np.argmax(predictions, axis=1))
        y_true.extend(np.argmax(labels.numpy(), axis=1))
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0)
    matrix = confusion_matrix(y_true, y_pred).tolist()
    return report, matrix


def run_training(config: TrainingConfig, create_demo: bool = False, samples_per_class: int = 250):
    set_seed(config.seed)
    output_dir = Path(config.output_dir)
    ensure_directories(output_dir)

    if create_demo:
        create_synthetic_dataset(base_dir=config.data_dir, samples_per_class=samples_per_class)

    stats = get_dataset_stats(config.data_dir)
    print(json.dumps(stats, indent=2))

    loader = DatasetLoader(
        data_dir=config.data_dir,
        image_size=(config.image_size, config.image_size),
        batch_size=config.batch_size,
    )
    train_ds, val_ds, test_ds, class_names = loader.load()
    train_ds, val_ds = build_training_pipelines(train_ds, val_ds, augment=config.augment)

    if test_ds is not None:
        test_ds = test_ds.map(lambda images, labels: (tf.cast(images, tf.float32) / 255.0, labels))
        test_ds = test_ds.prefetch(tf.data.AUTOTUNE)

    classifier = create_model(
        num_classes=len(class_names),
        image_size=(config.image_size, config.image_size),
        use_pretrained_vit=config.use_pretrained_vit,
    )

    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_accuracy", mode="max", patience=4, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-7),
        keras.callbacks.CSVLogger("logs/training_log.csv", append=False),
    ]

    classifier.compile(config.initial_lr)
    history_phase1 = classifier.model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=config.initial_epochs,
        callbacks=callbacks,
        verbose=1,
    )

    classifier.set_fine_tune(unfreeze_last_blocks=25)
    classifier.compile(config.fine_tune_lr)
    history_phase2 = classifier.model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=config.fine_tune_epochs,
        callbacks=callbacks,
        verbose=1,
    )

    val_metrics = classifier.model.evaluate(val_ds, verbose=0)
    val_accuracy = float(val_metrics[1])

    report = None
    matrix = None
    if test_ds is not None:
        report, matrix = evaluate_model(classifier.model, test_ds, class_names)

    model_path = output_dir / config.model_name
    classifier.save(str(model_path))

    metadata: Dict = {
        "class_names": class_names,
        "image_size": config.image_size,
        "val_accuracy": val_accuracy,
        "history_phase1": history_phase1.history,
        "history_phase2": history_phase2.history,
        "config": asdict(config),
        "test_classification_report": report,
        "test_confusion_matrix": matrix,
    }

    metadata_path = output_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"Model saved: {model_path}")
    print(f"Metadata saved: {metadata_path}")
    print(f"Validation accuracy: {val_accuracy:.4f}")
    print("Target reached: validation accuracy >= 95%" if val_accuracy >= 0.95 else "Target not reached.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train ViT skin disease classifier")
    parser.add_argument("--data-dir", type=str, default="dataset")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--initial-epochs", type=int, default=10)
    parser.add_argument("--fine-tune-epochs", type=int, default=10)
    parser.add_argument("--initial-lr", type=float, default=1e-3)
    parser.add_argument("--fine-tune-lr", type=float, default=1e-5)
    parser.add_argument("--output-dir", type=str, default="saved_model")
    parser.add_argument("--model-name", type=str, default="monkeypox_vit.keras")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--samples-per-class", type=int, default=250)
    parser.add_argument("--no-augment", action="store_true")
    parser.add_argument("--lightweight-vit", action="store_true")
    return parser


def main():
    args = build_parser().parse_args()
    config = TrainingConfig(
        data_dir=args.data_dir,
        image_size=args.image_size,
        batch_size=args.batch_size,
        initial_epochs=args.initial_epochs,
        fine_tune_epochs=args.fine_tune_epochs,
        initial_lr=args.initial_lr,
        fine_tune_lr=args.fine_tune_lr,
        output_dir=args.output_dir,
        model_name=args.model_name,
        seed=args.seed,
        augment=not args.no_augment,
        use_pretrained_vit=not args.lightweight_vit,
    )
    run_training(config, create_demo=args.demo, samples_per_class=args.samples_per_class)


if __name__ == "__main__":
    main()
