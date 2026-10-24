import json
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

from config import *
from data import load_raw, split_data, export_images, make_datasets
from model import build_model, build_finetune_model
from evaluate import evaluate_model


def set_seed():
    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)


def main():
    set_seed()
    print("TensorFlow:", tf.__version__)
    print("GPU:", len(tf.config.list_physical_devices("GPU")) > 0)

    # 1. Data (export sekali saja, skip kalau sudah ada)
    if not DATA_DIR.exists():
        ds, df = load_raw()
        train_df, val_df, test_df = split_data(df)
        export_images(ds, {"train": train_df, "val": val_df, "test": test_df})
    train_ds, val_ds, test_ds = make_datasets()

    # 2. Baseline training
    model = build_model()
    model.fit(
        train_ds, validation_data=val_ds, epochs=EPOCHS,
        callbacks=[
            ModelCheckpoint(CHECKPOINT_PATH, monitor="val_loss",
                            save_best_only=True, mode="min", verbose=1),
            EarlyStopping(monitor="val_loss", patience=5,
                          restore_best_weights=True, verbose=1),
            ReduceLROnPlateau(monitor="val_loss", factor=0.2,
                              patience=2, min_lr=1e-6, verbose=1),
        ],
    )
    best_model = tf.keras.models.load_model(CHECKPOINT_PATH)
    evaluate_model(best_model, test_ds, "Baseline", "cm_baseline.png")

    # 3. Fine-tuning
    fine_model = build_finetune_model()
    fine_model.fit(
        train_ds, validation_data=val_ds, epochs=FINE_TUNE_EPOCHS,
        callbacks=[EarlyStopping(monitor="val_loss", patience=3,
                                 restore_best_weights=True)],
    )

    # 4. Pilih model terbaik berdasarkan val loss
    base_loss, base_acc = best_model.evaluate(val_ds, verbose=0)
    fine_loss, fine_acc = fine_model.evaluate(val_ds, verbose=0)
    print(f"Baseline   -> val_loss {base_loss:.4f}, val_acc {base_acc:.4f}")
    print(f"Fine-tuned -> val_loss {fine_loss:.4f}, val_acc {fine_acc:.4f}")

    if fine_loss < base_loss:
        selected, name = fine_model, "Fine-tuned MobileNetV2"
    else:
        selected, name = best_model, "Baseline MobileNetV2"
    print("Selected:", name)

    evaluate_model(selected, test_ds, name, "cm_final.png")

    # 5. Simpan
    selected.save(FINAL_MODEL_PATH)
    with open(LABEL_PATH, "w") as f:
        json.dump(CLASS_NAMES, f)
    print("Saved:", FINAL_MODEL_PATH)


if __name__ == "__main__":
    main()