#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train EfficientNetB0 classifier on HAM10000 splits.")
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--output-dir", type=Path, default=Path("models/binary"))
    parser.add_argument("--task", choices=["binary", "multiclass"], default="binary")
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--fine-tune-lr", type=float, default=1e-4)
    parser.add_argument("--fine-tune-at", type=int, default=220)
    return parser.parse_args()


def load_splits(processed_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_df = pd.read_csv(processed_dir / "train.csv")
    val_df = pd.read_csv(processed_dir / "val.csv")
    test_df = pd.read_csv(processed_dir / "test.csv")
    return train_df, val_df, test_df


def build_generators(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    task: str,
    img_size: int,
    batch_size: int,
):
    if task == "binary":
        y_col = "binary_label"
        class_mode = "binary"
        train_df[y_col] = train_df[y_col].astype(str)
        val_df[y_col] = val_df[y_col].astype(str)
        test_df[y_col] = test_df[y_col].astype(str)
    else:
        y_col = "dx"
        class_mode = "categorical"

    train_aug = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        vertical_flip=True,
    )
    eval_aug = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1.0 / 255)

    train_gen = train_aug.flow_from_dataframe(
        train_df,
        x_col="image_path",
        y_col=y_col,
        target_size=(img_size, img_size),
        class_mode=class_mode,
        batch_size=batch_size,
        shuffle=True,
    )

    val_gen = eval_aug.flow_from_dataframe(
        val_df,
        x_col="image_path",
        y_col=y_col,
        target_size=(img_size, img_size),
        class_mode=class_mode,
        batch_size=batch_size,
        shuffle=False,
    )

    test_gen = eval_aug.flow_from_dataframe(
        test_df,
        x_col="image_path",
        y_col=y_col,
        target_size=(img_size, img_size),
        class_mode=class_mode,
        batch_size=batch_size,
        shuffle=False,
    )

    return train_gen, val_gen, test_gen


def build_model(task: str, img_size: int, learning_rate: float) -> tf.keras.Model:
    base_model = tf.keras.applications.EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_shape=(img_size, img_size, 3),
    )
    base_model.trainable = False

    inputs = tf.keras.Input(shape=(img_size, img_size, 3))
    x = base_model(inputs, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.3)(x)

    if task == "binary":
        outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)
        loss = "binary_crossentropy"
        metrics = ["accuracy", tf.keras.metrics.AUC(name="auc")]
    else:
        outputs = tf.keras.layers.Dense(7, activation="softmax")(x)
        loss = "categorical_crossentropy"
        metrics = ["accuracy"]

    model = tf.keras.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=loss,
        metrics=metrics,
    )
    return model


def evaluate_model(model: tf.keras.Model, test_gen, task: str) -> Dict:
    probs = model.predict(test_gen)
    y_true = test_gen.classes

    if task == "binary":
        y_score = probs.ravel()
        y_pred = (y_score >= 0.5).astype(int)
        report = classification_report(y_true, y_pred, output_dict=True)
        cm = confusion_matrix(y_true, y_pred)
        auc = roc_auc_score(y_true, y_score)
        return {"classification_report": report, "confusion_matrix": cm.tolist(), "auc": float(auc)}

    y_pred = np.argmax(probs, axis=1)
    report = classification_report(y_true, y_pred, output_dict=True)
    cm = confusion_matrix(y_true, y_pred)
    return {"classification_report": report, "confusion_matrix": cm.tolist()}


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_df, val_df, test_df = load_splits(args.processed_dir)
    train_gen, val_gen, test_gen = build_generators(
        train_df, val_df, test_df, args.task, args.img_size, args.batch_size
    )

    model = build_model(args.task, args.img_size, args.learning_rate)

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(args.output_dir / "best_model.keras"),
            monitor="val_loss",
            save_best_only=True,
        ),
    ]

    history = model.fit(train_gen, validation_data=val_gen, epochs=args.epochs, callbacks=callbacks)

    if hasattr(model.layers[1], "trainable"):
        model.layers[1].trainable = True
        for layer in model.layers[1].layers[: args.fine_tune_at]:
            layer.trainable = False

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=args.fine_tune_lr),
            loss=model.loss,
            metrics=model.metrics,
        )

        model.fit(train_gen, validation_data=val_gen, epochs=max(2, args.epochs // 2), callbacks=callbacks)

    eval_output = evaluate_model(model, test_gen, args.task)

    label_map = {int(v): k for k, v in test_gen.class_indices.items()}
    with open(args.output_dir / "label_map.json", "w", encoding="utf-8") as f:
        json.dump(label_map, f, indent=2)

    with open(args.output_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "history": {k: [float(x) for x in v] for k, v in history.history.items()},
                **{k: v for k, v in eval_output.items() if k != "confusion_matrix"},
            },
            f,
            indent=2,
        )

    pd.DataFrame(eval_output["confusion_matrix"]).to_csv(args.output_dir / "confusion_matrix.csv", index=False)
    with open(args.output_dir / "classification_report.json", "w", encoding="utf-8") as f:
        json.dump(eval_output["classification_report"], f, indent=2)

    print(f"Training complete. Artifacts saved in {args.output_dir}")


if __name__ == "__main__":
    main()
