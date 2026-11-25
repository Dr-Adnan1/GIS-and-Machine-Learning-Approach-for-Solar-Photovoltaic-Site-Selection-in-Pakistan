"""
Full-raster suitability prediction and GeoTIFF export.
"""
import numpy as np
import rasterio

from src.config import MAPS_DIR


def predict_raster(model, scaler, stack, meta, tag):
    flat = stack.reshape(-1, stack.shape[2])
    X = scaler.transform(flat) if scaler is not None else flat
    proba = model.predict_proba(X)[:, 1]
    return proba.reshape(stack.shape[0], stack.shape[1]).astype(np.float32)


def save_geotiff(array, meta, tag):
    out = MAPS_DIR / f"{tag}_PV_suitability.tif"
    with rasterio.open(
        out, "w", driver="GTiff",
        height=meta["height"], width=meta["width"], count=1,
        dtype="float32", crs=meta["crs"], transform=meta["transform"],
    ) as dst:
        dst.write(array, 1)
    print(f"Saved {out}")
    return out