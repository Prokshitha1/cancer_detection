#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.cancer_detection.data_utils import add_labels_and_paths, stratified_splits


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare HAM10000 metadata and split files.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--test-size", type=float, default=0.15)
    parser.add_argument("--val-size", type=float, default=0.15)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    metadata_path = args.data_dir / "HAM10000_metadata.csv"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing metadata file: {metadata_path}")

    metadata = pd.read_csv(metadata_path)
    prepared = add_labels_and_paths(metadata, args.data_dir)

    train_df, val_df, test_df = stratified_splits(
        prepared,
        target_col="dx",
        test_size=args.test_size,
        val_size=args.val_size,
        random_state=args.random_state,
    )

    prepared.to_csv(args.output_dir / "metadata_with_paths.csv", index=False)
    train_df.to_csv(args.output_dir / "train.csv", index=False)
    val_df.to_csv(args.output_dir / "val.csv", index=False)
    test_df.to_csv(args.output_dir / "test.csv", index=False)

    print(f"Saved processed metadata and splits to: {args.output_dir}")
    print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")


if __name__ == "__main__":
    main()
