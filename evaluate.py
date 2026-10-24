import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

from config import *


def evaluate_model(model, test_ds, title, save_name):
    loss, acc = model.evaluate(test_ds, verbose=1)
    print(f"{title} | Test loss: {loss:.4f} | Test accuracy: {acc:.4f}")

    y_true = np.concatenate([labels.numpy() for _, labels in test_ds])
    y_pred = np.argmax(model.predict(test_ds, verbose=0), axis=1)
    labels_idx = list(range(len(CLASS_NAMES)))

    print(classification_report(
        y_true, y_pred, labels=labels_idx,
        target_names=CLASS_NAMES, digits=4, zero_division=0,
    ))

    cm = confusion_matrix(y_true, y_pred, labels=labels_idx)
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.xlabel("Predicted class")
    plt.ylabel("Actual class")
    plt.title(f"Confusion Matrix - {title}")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / save_name, dpi=150)
    plt.close()