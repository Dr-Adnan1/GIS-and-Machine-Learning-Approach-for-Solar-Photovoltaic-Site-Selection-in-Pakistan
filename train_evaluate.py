"""
Model evaluation: ROC curves, AUC, and accuracy (paper Section 4.5).
"""
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, accuracy_score, roc_curve

from src.config import FIGURES_DIR


def evaluate(model_name, model, X_test, y_test, scaler=None):
    if scaler is not None:
        X_test = scaler.transform(X_test)
    proba = model.predict_proba(X_test)[:, 1]
    preds = model.predict(X_test)
    auc = roc_auc_score(y_test, proba)
    acc = accuracy_score(y_test, preds)
    return auc, acc, proba


def plot_roc_curves(models, X_test, y_test, scaler):
    """Reproduce the ROC-curve figure from paper Section 4.5."""
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = {"Random Forest": "blue", "XGBoost": "green", "MLP": "red"}
    results = {}

    for name, model in models.items():
        s = scaler if name == "MLP" else None
        auc, acc, proba = evaluate(name, model, X_test, y_test, scaler=s)
        fpr, tpr, _ = roc_curve(y_test, proba)
        ax.plot(
            fpr, tpr, color=colors[name],
            label=f"{name} (AUC={auc:.2f}, Acc={acc:.2f})",
        )
        results[name] = {"AUC": auc, "Accuracy": acc}
        print(f"{name:14s}  AUC={auc:.4f}  Acc={acc:.4f}")

    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — Solar PV Site Selection")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)

    out = FIGURES_DIR / "roc_curves_pv_site_selection.png"
    fig.savefig(out, dpi=400, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved ROC figure -> {out}")
    return results