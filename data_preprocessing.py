from __future__ import annotations

import random
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import tensorflow as tf
from PIL import Image, ImageDraw

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class DataPreprocessor:
    def __init__(self, image_size: Tuple[int, int] = (224, 224)):
        self.image_size = image_size

    def preprocess_for_model(self, image: Image.Image) -> np.ndarray:
        image = image.convert("RGB").resize(self.image_size)
        array = tf.keras.utils.img_to_array(image)
        return array / 255.0


class DatasetLoader:
    def __init__(self, data_dir: str = "dataset", image_size: Tuple[int, int] = (224, 224), batch_size: int = 32):
        self.data_dir = Path(data_dir)
        self.image_size = image_size
        self.batch_size = batch_size

    def _dataset_has_images(self, directory: Path) -> bool:
        if not directory.exists():
            return False
        return any(path.suffix.lower() in SUPPORTED_EXTENSIONS for path in directory.rglob("*"))

    def _build_dataset(self, directory: Path, shuffle: bool, class_names: List[str] | None = None):
        if not directory.exists():
            return None
        return tf.keras.utils.image_dataset_from_directory(
            directory,
            labels="inferred",
            label_mode="categorical",
            class_names=class_names,
            image_size=self.image_size,
            batch_size=self.batch_size,
            shuffle=shuffle,
            seed=42,
        )

    def load(self):
        train_dir = self.data_dir / "train"
        val_dir = self.data_dir / "val"
        test_dir = self.data_dir / "test"

        if not self._dataset_has_images(train_dir):
            raise FileNotFoundError("Expected training images under dataset/train/<class_name>/")

        train_ds = self._build_dataset(train_dir, shuffle=True)
        class_names = list(train_ds.class_names)
        val_ds = self._build_dataset(val_dir, shuffle=False, class_names=class_names)
        test_ds = self._build_dataset(test_dir, shuffle=False, class_names=class_names)

        if val_ds is None:
            cardinality = int(tf.data.experimental.cardinality(train_ds).numpy())
            split_size = max(1, int(cardinality * 0.2))
            val_ds = train_ds.take(split_size)
            train_ds = train_ds.skip(split_size)

        return train_ds, val_ds, test_ds, class_names


def build_training_pipelines(train_ds: tf.data.Dataset, val_ds: tf.data.Dataset, augment: bool = True):
    augmentation = tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.08),
            tf.keras.layers.RandomZoom(0.12),
            tf.keras.layers.RandomContrast(0.1),
        ],
        name="augmentation",
    )

    def normalize_only(images, labels):
        return tf.cast(images, tf.float32) / 255.0, labels

    def normalize_with_augment(images, labels):
        images = tf.cast(images, tf.float32) / 255.0
        if augment:
            images = augmentation(images, training=True)
        return images, labels

    train_ds = train_ds.map(normalize_with_augment, num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.map(normalize_only, num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)
    return train_ds, val_ds


def create_synthetic_dataset(base_dir: str = "dataset", samples_per_class: int = 200):
    classes = ["monkeypox", "chickenpox", "measles", "normal", "other"]
    image_size = (224, 224)
    base_path = Path(base_dir)

    if base_path.exists():
        shutil.rmtree(base_path)

    for split, multiplier in (("train", 1.0), ("val", 0.25), ("test", 0.25)):
        for class_name in classes:
            target = base_path / split / class_name
            target.mkdir(parents=True, exist_ok=True)
            count = max(10, int(samples_per_class * multiplier))

            for index in range(count):
                base_colors = {
                    "monkeypox": (178, 128, 118),
                    "chickenpox": (188, 152, 132),
                    "measles": (206, 158, 136),
                    "normal": (214, 181, 154),
                    "other": (168, 146, 132),
                }
                background = base_colors[class_name]
                jittered = tuple(max(0, min(255, c + random.randint(-8, 8))) for c in background)
                image = Image.new("RGB", image_size, color=jittered)
                draw = ImageDraw.Draw(image)

                for _ in range(12):
                    x0 = random.randint(0, image_size[0] - 20)
                    y0 = random.randint(0, image_size[1] - 20)
                    color = tuple(max(0, min(255, c + random.randint(-16, 16))) for c in jittered)
                    draw.rectangle([x0, y0, x0 + random.randint(6, 18), y0 + random.randint(6, 18)], fill=color)

                if class_name == "monkeypox":
                    for _ in range(10):
                        x = random.randint(40, 180)
                        y = random.randint(40, 180)
                        draw.ellipse([x, y, x + 18, y + 18], fill=(120, 50, 50))
                        draw.ellipse([x + 4, y + 4, x + 14, y + 14], fill=(230, 200, 180))
                    draw.rectangle([10, 10, 70, 26], fill=(200, 60, 60))
                elif class_name == "chickenpox":
                    for _ in range(8):
                        x = random.randint(30, 190)
                        y = random.randint(30, 190)
                        draw.ellipse([x, y, x + 14, y + 14], fill=(220, 70, 70))
                    draw.rectangle([154, 10, 214, 26], fill=(220, 110, 40))
                elif class_name == "measles":
                    for _ in range(22):
                        x = random.randint(20, 200)
                        y = random.randint(20, 200)
                        draw.ellipse([x, y, x + 8, y + 8], fill=(160, 40, 40))
                    draw.rectangle([10, 198, 70, 214], fill=(170, 40, 40))
                elif class_name == "normal":
                    for _ in range(4):
                        x = random.randint(30, 180)
                        y = random.randint(30, 180)
                        draw.ellipse([x, y, x + 10, y + 10], fill=(205, 178, 156))
                    draw.rectangle([154, 198, 214, 214], fill=(120, 180, 120))
                else:
                    for _ in range(10):
                        x = random.randint(25, 190)
                        y = random.randint(25, 190)
                        draw.rectangle([x, y, x + 14, y + 10], fill=(100, 90, 150))
                    draw.rectangle([82, 10, 142, 26], fill=(110, 90, 170))

                image.save(target / f"{class_name}_{index:05d}.jpg", "JPEG")


def get_dataset_stats(data_dir: str = "dataset") -> Dict[str, Dict[str, int]]:
    base = Path(data_dir)
    stats: Dict[str, Dict[str, int]] = {}
    for split in ("train", "val", "test"):
        split_dir = base / split
        if not split_dir.exists():
            continue
        class_counts: Dict[str, int] = {}
        for class_dir in sorted(split_dir.iterdir()):
            if class_dir.is_dir():
                class_counts[class_dir.name] = sum(1 for p in class_dir.iterdir() if p.suffix.lower() in SUPPORTED_EXTENSIONS)
        stats[split] = class_counts
    return stats
