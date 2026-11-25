# GIS–ML Framework for Solar PV Site Selection in the Cholistan Desert, Pakistan

Official code repository for:

> **Ashraf, H.A.; Li, J.; Li, Z.; Sohail, A.; Ahmed, R.; Butt, M.H.; Ullah, H.**
> *Geographic Information System and Machine Learning Approach for Solar
> Photovoltaic Site Selection: A Case Study in Pakistan.*
> **Processes 2025, 13, 981.** https://doi.org/10.3390/pr13040981

## Overview

This repository implements an integrated **GIS–Machine-Learning** pipeline
that identifies suitable sites for utility-scale solar photovoltaic (PV)
installations in the **Cholistan Desert, Punjab, Pakistan**.

Three classifiers are compared:

| Model | Key settings |
|-------|--------------|
| **Random Forest** | 500 trees, `class_weight='balanced'` |
| **XGBoost** | 500 rounds, `scale_pos_weight=3` |
| **MLP (Neural Net)** | 3 hidden layers (256-128-64), ReLU, Adam, L2 α=1000, early stopping |

The **Random Forest** model achieves the best performance (AUC = 0.92) and is
used for the final suitability map. Feature importance is interpreted with
**SHAP**.

## Dataset

- **355 field-survey points** (PV and non-PV locations), split 70/30.
- **14 conditioning factors** at 1 km resolution:
  - *Geographical:* LULC, aspect, slope, elevation, air temperature, PM10, NDVI, wind speed
  - *Socio-economic:* carbon emission, distance to residential area, population density, distance to road network
  - *Resource:* solar irradiation, sunshine duration

Raw data are **not included** in this repository due to licensing. Place your
files in `data/raw/`: