"""
Central configuration for the GIS-ML solar PV site selection study.
Reference: Ashraf et al., Processes 2025, 13, 981.
"""
from pathlib import Path

# ---------- Paths ----------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
OUTPUTS = PROJECT_ROOT / "outputs"
MODELS_DIR = OUTPUTS / "models"
MAPS_DIR = OUTPUTS / "maps"
FIGURES_DIR = OUTPUTS / "figures"

for d in (MODELS_DIR, MAPS_DIR, FIGURES_DIR, DATA_PROCESSED):
    d.mkdir(parents=True, exist_ok=True)

# ---------- Data files ----------
POINTS_CSV = DATA_RAW / "Export_4.csv"       # 355 field-survey points
RASTER_DIR = DATA_RAW                        # 14 conditioning-factor GeoTIFFs

# ---------- Experimental settings (paper Section 4) ----------
RANDOM_STATE = 123
TEST_SIZE = 0.30          # 70/30 split
TARGET_COLUMN = "target"
COORD_COLUMNS = ["X", "Y"]

# ---------- Model hyperparameters (paper Sections 4.1-4.3) ----------
RF_PARAMS = dict(
    n_estimators=500,
    random_state=RANDOM_STATE,
    class_weight="balanced",
)

XGB_PARAMS = dict(
    n_estimators=500,
    random_state=RANDOM_STATE,
    scale_pos_weight=3,
    eval_metric="logloss",
)

MLP_PARAMS = dict(
    hidden_layer_sizes=(256, 128, 64),   # three hidden layers
    activation="relu",
    solver="adam",
    alpha=1000,                          # L2 regularization (paper Sect. 4.1)
    learning_rate="adaptive",
    max_iter=2000,
    random_state=42,
    early_stopping=True,
    validation_fraction=0.10,
)

# ---------- Raster processing ----------
NODATA_THRESHOLD = -1e20
RESAMPLING_METHOD = "bilinear"
OUTPUT_DTYPE = "float32"