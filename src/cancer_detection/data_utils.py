from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

MALIGNANT_DX = {"mel", "bcc", "akiec"}


def build_image_path_map(data_dir: Path) -> Dict[str, str]:
    """Map image_id to absolute image path across both HAM10000 image folders."""
    image_paths = {}
    for folder in ["HAM10000_images_part_1", "HAM10000_images_part_2"]:
        directory = data_dir / folder
        if not directory.exists():
            continue
        for image_file in directory.glob("*.jpg"):
            image_paths[image_file.stem] = str(image_file.resolve())
    return image_paths


def add_labels_and_paths(metadata: pd.DataFrame, data_dir: Path) -> pd.DataFrame:
    """Add binary target and image_path columns to metadata."""
    image_map = build_image_path_map(data_dir)
    df = metadata.copy()
    df["binary_label"] = df["dx"].apply(lambda x: 1 if x in MALIGNANT_DX else 0)
    df["image_path"] = df["image_id"].map(image_map)
    df = df.dropna(subset=["image_path"])
    return df


def stratified_splits(
    df: pd.DataFrame,
    target_col: str,
    test_size: float,
    val_size: float,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create train/val/test splits with stratification."""
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        stratify=df[target_col],
        random_state=random_state,
    )

    adjusted_val = val_size / (1.0 - test_size)
    train_df, val_df = train_test_split(
        train_df,
        test_size=adjusted_val,
        stratify=train_df[target_col],
        random_state=random_state,
    )

    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )
