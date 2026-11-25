"""
Data preprocessing: raster stacking, resampling, and point-value extraction.
Implements Section 3.2 of the paper (1 km grid, bilinear resampling,
invalid-value handling, Extract-Values-to-Points equivalent).
"""
import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling
from pathlib import Path
from sklearn.model_selection import train_test_split

from src.config import (
    POINTS_CSV, RASTER_DIR, RANDOM_STATE, TEST_SIZE,
    TARGET_COLUMN, COORD_COLUMNS, NODATA_THRESHOLD,
)


def load_points(csv_path=POINTS_CSV):
    """Load the 355 field-survey points (PV / non-PV)."""
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]
    if "target " in df.columns:
        df.rename(columns={"target ": TARGET_COLUMN}, inplace=True)
    return df


def split_points(df):
    """70/30 train-test split with fixed seed."""
    train, test = train_test_split(
        df, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    return train, test


def build_raster_stack(raster_dir=RASTER_DIR):
    """
    Resample all GeoTIFFs to the grid of the first raster using bilinear
    interpolation, replace invalid values (<= -1e20) with 0, clip to
    float32 range, and stack into a 3-D array (H, W, N_features).

    Returns
    -------
    stack : np.ndarray (H, W, F)
    meta  : dict with 'transform', 'width', 'height', 'crs'
    """
    tif_files = sorted(Path(raster_dir).glob("*.tif"))
    if not tif_files:
        raise FileNotFoundError(f"No .tif files found in {raster_dir}")

    with rasterio.open(tif_files[0]) as tmpl:
        transform = tmpl.transform
        width, height = tmpl.width, tmpl.height
        crs = tmpl.crs

    f32_max = np.finfo(np.float32).max
    f32_min = np.finfo(np.float32).min
    bands = []

    for f in tif_files:
        with rasterio.open(f) as src:
            arr = src.read(
                1,
                out_shape=(height, width),
                resampling=Resampling.bilinear,
            ).astype(np.float64)
            arr[arr <= NODATA_THRESHOLD] = 0.0
            arr = np.nan_to_num(arr, nan=0.0, posinf=f32_max, neginf=f32_min)
            arr = np.clip(arr, f32_min, f32_max).astype(np.float32)
            bands.append(arr)
            print(f"  [ok] {f.name}")

    stack = np.stack(bands, axis=-1)
    meta = dict(transform=transform, width=width, height=height, crs=crs)
    print(f"Raster stack: {stack.shape} (H, W, F)")
    return stack, meta


def extract_at_points(stack, coords, transform):
    """
    Sample raster-stack values at (lon, lat) coordinates.
    Out-of-bounds points get a zero-vector (consistent with paper).
    """
    values = np.zeros((len(coords), stack.shape[2]), dtype=np.float32)
    for i, (lon, lat) in enumerate(coords):
        col, row = ~transform * (lon, lat)
        row, col = int(row), int(col)
        if 0 <= row < stack.shape[0] and 0 <= col < stack.shape[1]:
            values[i] = stack[row, col, :]
    return values


def prepare_train_test(stack, meta):
    """Full pipeline: load points -> split -> extract -> return arrays."""
    df = load_points()
    train_df, test_df = split_points(df)

    X_train = extract_at_points(
        stack, train_df[COORD_COLUMNS].values, meta["transform"]
    )
    X_test = extract_at_points(
        stack, test_df[COORD_COLUMNS].values, meta["transform"]
    )
    y_train = train_df[TARGET_COLUMN].values
    y_test = test_df[TARGET_COLUMN].values

    print(f"Train: X={X_train.shape}, y={y_train.shape}")
    print(f"Test : X={X_test.shape},  y={y_test.shape}")
    return X_train, X_test, y_train, y_test, train_df, test_df