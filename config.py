from pathlib import Path

SEED = 42
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 15
FINE_TUNE_EPOCHS = 8
CLASS_NAMES = ["b3", "inorganic", "organic"]
REPO_ID = "huaweilin/waste-classification"

CLASS_MAPPING = {
    "Organic": "organic",
    "Recyclable": "inorganic",
    "Non-Recyclable": "inorganic",
    "Hazardous": "b3",
}

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "waste_dataset"
OUTPUT_DIR = ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

CHECKPOINT_PATH = OUTPUT_DIR / "best_waste_model.keras"
FINAL_MODEL_PATH = OUTPUT_DIR / "smart_waste_final.keras"
LABEL_PATH = OUTPUT_DIR / "smart_waste_labels.json"