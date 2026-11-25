"""
Machine-learning model definitions (paper Sections 4.1-4.3).
"""
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from src.config import RF_PARAMS, XGB_PARAMS, MLP_PARAMS


def build_rf():
    """Random Forest — 500 trees, balanced class weights (paper Sect. 4.2)."""
    return RandomForestClassifier(**RF_PARAMS)


def build_xgb():
    """XGBoost — 500 boosting rounds, scale_pos_weight (paper Sect. 4.3)."""
    return XGBClassifier(**XGB_PARAMS)


def build_mlp():
    """MLP — 3 hidden layers 256-128-64, ReLU, Adam, L2 (paper Sect. 4.1)."""
    return MLPClassifier(**MLP_PARAMS)


def train_all(X_train, y_train, X_train_norm):
    """Train RF, XGBoost, and MLP. Returns fitted models + scaler."""
    scaler = StandardScaler().fit(X_train)

    print("Training Random Forest...")
    rf = build_rf().fit(X_train, y_train)

    print("Training XGBoost...")
    xgb = build_xgb().fit(X_train, y_train)

    print("Training MLP...")
    mlp = build_mlp().fit(X_train_norm, y_train)

    return {"Random Forest": rf, "XGBoost": xgb, "MLP": mlp}, scaler