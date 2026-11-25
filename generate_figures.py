"""
Reproduce paper Figure 2 (conditioning-factor maps) and Figure 1 (study area).
Requires a shapefile of Cholistan boundary (not distributed).
"""
import rasterio
import matplotlib.pyplot as plt
from pathlib import Path

from src.config import RASTER_DIR, FIGURES_DIR

def plot_conditioning_factors():
    tifs = sorted(Path(RASTER_DIR).glob("*.tif"))
    n = len(tifs)
    fig, axes = plt.subplots(3, (n + 2) // 3, figsize=(15, 12))
    axes = axes.flatten()
    for ax, tif in zip(axes, tifs):
        with rasterio.open(tif) as src:
            arr = src.read(1)
        im = ax.imshow(arr, cmap="viridis")
        ax.set_title(tif.stem, fontsize=9)
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.04)
    for ax in axes[n:]:
        ax.axis("off")
    fig.savefig(FIGURES_DIR / "conditioning_factors.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

if __name__ == "__main__":
    plot_conditioning_factors()