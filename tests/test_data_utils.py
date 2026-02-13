from pathlib import Path

import pandas as pd

from src.cancer_detection.data_utils import MALIGNANT_DX, add_labels_and_paths


def test_malignant_mapping():
    assert "mel" in MALIGNANT_DX


def test_add_labels_and_paths(tmp_path: Path):
    part1 = tmp_path / "HAM10000_images_part_1"
    part1.mkdir(parents=True)
    img = part1 / "abc.jpg"
    img.write_bytes(b"x")

    df = pd.DataFrame({"image_id": ["abc"], "dx": ["mel"]})
    out = add_labels_and_paths(df, tmp_path)

    assert len(out) == 1
    assert out.iloc[0]["binary_label"] == 1
    assert out.iloc[0]["image_path"].endswith("abc.jpg")
