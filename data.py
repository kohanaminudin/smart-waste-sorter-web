import shutil
import numpy as np
import tensorflow as tf
from datasets import load_dataset
from sklearn.model_selection import train_test_split

from config import *


def load_raw():
    ds = load_dataset(REPO_ID, split="cleaned")
    df = ds.to_pandas().reset_index(drop=True)
    df["sample_idx"] = np.arange(len(df))
    df["target"] = df["class"].map(CLASS_MAPPING)

    if df["target"].isna().any():
        print(df.loc[df["target"].isna(), "class"].value_counts())
        raise ValueError("Some original classes have no target mapping.")
    return ds, df


def split_data(df):
    train_df, temp_df = train_test_split(
        df, test_size=0.20, random_state=SEED, stratify=df["target"]
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.50, random_state=SEED, stratify=temp_df["target"]
    )
    return train_df, val_df, test_df


def export_images(ds, splits: dict):
    if DATA_DIR.exists():
        shutil.rmtree(DATA_DIR)

    for split_name, split_df in splits.items():
        for class_name in CLASS_NAMES:
            (DATA_DIR / split_name / class_name).mkdir(parents=True, exist_ok=True)

        for _, row in split_df.iterrows():
            image = ds[int(row["sample_idx"])]["image"].convert("RGB")
            out = DATA_DIR / split_name / row["target"] / f"{int(row['sample_idx'])}.jpg"
            image.save(out, format="JPEG", quality=95)
    print("Image export completed!")


def make_datasets():
    def _load(split, shuffle):
        return tf.keras.utils.image_dataset_from_directory(
            DATA_DIR / split,
            class_names=CLASS_NAMES,
            image_size=IMAGE_SIZE,
            batch_size=BATCH_SIZE,
            label_mode="int",
            shuffle=shuffle,
            seed=SEED if shuffle else None,
        )

    auto = tf.data.AUTOTUNE
    train_ds = _load("train", True).prefetch(auto)
    val_ds = _load("val", False).prefetch(auto)
    test_ds = _load("test", False).prefetch(auto)
    return train_ds, val_ds, test_ds