"""
=============================================================================
CTU-13 Scénario 8 — Implémentation KGB + Pipeline ML
=============================================================================
Implémente les deux variantes du détecteur KGB (Pevný et al., 2012) :
  - KGBf   : anomalie détectée dans les composantes à HAUTE variance (PCA)
  - KGBfog : anomalie détectée dans les composantes à BASSE  variance (PCA)

Pipeline :
  1. Chargement des features (entropy_features.csv produit par l'EDA)
  2. Prétraitement : normalisation, gestion du déséquilibre
  3. KGBf  : PCA → résidu sur PC haute variance → score d'anomalie
  4. KGBfog: PCA → résidu sur PC basse  variance → score d'anomalie
  5. Comparaison avec classifieurs ML supervisés (RF, SVM, LR)
  6. Évaluation complète : métriques du papier CTU-13

Usage :
  python 02_kgb_pipeline.py
  python 02_kgb_pipeline.py --features eda_output/entropy_features.csv
=============================================================================
"""

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import (
    confusion_matrix, classification_report,
    roc_auc_score, roc_curve, precision_recall_curve,
    f1_score, precision_score, recall_score
)
from sklearn.utils.class_weight import compute_class_weight

warnings.filterwarnings("ignore")

OUTPUT_DIR = Path("kgb_output")
OUTPUT_DIR.mkdir(exist_ok=True)

LABEL_BOTNET     = "Botnet"
LABEL_NORMAL     = "Normal"
LABEL_BACKGROUND = "Background"

# Features utilisées par KGB (Lakhina Entropy)
KGB_FEATURES = ["H_dst_ip", "H_dst_port", "H_src_port"]

# Features étendues pour les classifieurs ML supervisés
ALL_FEATURES = [
    "H_dst_ip", "H_dst_port", "H_src_port",
    "mean_bytes", "std_bytes", "mean_pkts", "std_pkts",
    "mean_dur", "n_flows",
]


# =============================================================================
# 1. Chargement & préparation
# =============================================================================

def load_features(path: str) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """
    Charge entropy_features.csv et prépare X, y.
    Retourne (df, X_kgb, y_binary) où y_binary = 1 si Botnet, 0 sinon.
    """
    print(f"\n[1/6] Chargement des features : {path}")
    df = pd.read_csv(path)

    print(f"      IPs totales    : {len(df):,}")
    for lbl in [LABEL_BOTNET, LABEL_NORMAL, LABEL_BACKGROUND]:
        n = (df["label"] == lbl).sum()
        print(f"      IPs {lbl:<12}: {n:,}")

    # Imputation des NaN (std=NaN si une seule observation)
    for col in ALL_FEATURES:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    # Label binaire : Botnet=1, reste=0
    df["y_binary"] = (df["label"] == LABEL_BOTNET).astype(int)

    # Label ternaire pour affichage
    df["y_ternary"] = df["label"].map({
        LABEL_BOTNET: 2, LABEL_NORMAL: 1, LABEL_BACKGROUND: 0
    }).fillna(0).astype(int)

    return df


def get_X(df: pd.DataFrame, features: list) -> np.ndarray:
    cols = [c for c in features if c in df.columns]
    missing = [c for c in features if c not in df.columns]
    if missing:
        print(f"      ⚠️  Features manquantes (ignorées) : {missing}")
    return df[cols].values


# =============================================================================
# 2. Implémentation KGB
# =============================================================================

class KGBDetector:
    """
    Implémentation du détecteur KGB (Pevný et al., 2012).

    KGB est basé sur l'analyse PCA des vecteurs d'entropie par IP source.
    Deux variantes :
      - KGBf   (fog=False) : score = résidu dans l'espace des PC à HAUTE variance
                             → détecte les anomalies "flagrantes" (comportement très différent)
      - KGBfog (fog=True)  : score = résidu dans l'espace des PC à BASSE  variance
                             → détecte les anomalies "discrètes" cachées dans le bruit

    Algorithme :
      1. Centrer les données (mean=0)
      2. PCA sur les données d'entraînement
      3. Sélectionner les k premières PC (KGBf) ou les dernières (KGBfog)
      4. Pour chaque IP : projeter sur les PC sélectionnées, mesurer le résidu
      5. Score d'anomalie ∈ [0,1] = résidu normalisé
    """

    def __init__(self, n_components: int = None, fog: bool = False, variance_threshold: float = 0.9):
        """
        Args:
            n_components       : nombre de PC à utiliser (None = auto via variance_threshold)
            fog                : False → KGBf (haute variance), True → KGBfog (basse variance)
            variance_threshold : seuil de variance expliquée pour sélection auto des PC (KGBf)
        """
        self.fog               = fog
        self.n_components      = n_components
        self.variance_threshold = variance_threshold
        self.pca               = None
        self.scaler            = None
        self.n_pc_selected     = None
        self.threshold         = None    # seuil de classification (fixé par fit)

    def fit(self, X_train: np.ndarray) -> "KGBDetector":
        """Entraîne le modèle PCA sur les données (supposées majoritairement normales)."""
        # Normalisation
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X_train)

        # PCA complète
        n_feat = X_train.shape[1]
        self.pca = PCA(n_components=n_feat)
        self.pca.fit(X_scaled)

        # Sélection des composantes
        if self.n_components is not None:
            self.n_pc_selected = self.n_components
        else:
            # KGBf : PC qui expliquent variance_threshold de la variance
            cumvar = np.cumsum(self.pca.explained_variance_ratio_)
            self.n_pc_selected = int(np.searchsorted(cumvar, self.variance_threshold) + 1)
            self.n_pc_selected = min(self.n_pc_selected, n_feat - 1)
            self.n_pc_selected = max(self.n_pc_selected, 1)

        name = "KGBfog" if self.fog else "KGBf"
        mode = "basse variance" if self.fog else "haute variance"
        print(f"      {name} : {self.n_pc_selected}/{n_feat} PC sélectionnées ({mode})")
        print(f"      Variance expliquée par PC : "
              f"{self.pca.explained_variance_ratio_[:min(5,n_feat)].round(3)}")

        # Seuil par défaut : percentile 95 des scores sur données d'entraînement
        scores_train = self._compute_scores(X_scaled)
        self.threshold = float(np.percentile(scores_train, 95))
        print(f"      Seuil (percentile 95 train) : {self.threshold:.4f}")

        return self

    def _compute_scores(self, X_scaled: np.ndarray) -> np.ndarray:
        """Calcule le score d'anomalie pour chaque observation."""
        # Projection dans l'espace PCA
        Z = self.pca.transform(X_scaled)     # (n_samples, n_features)

        if not self.fog:
            # KGBf : résidu = norme dans l'espace des PC sélectionnées (haute variance)
            # Le score mesure l'éloignement par rapport aux PC dominantes
            Z_selected = Z[:, :self.n_pc_selected]
            # Reconstruction depuis les PC sélectionnées
            X_recon = Z_selected @ self.pca.components_[:self.n_pc_selected]
            residual = X_scaled - X_recon
        else:
            # KGBfog : résidu dans l'espace des PC à basse variance (le "bruit")
            # Une anomalie discrète génère une perturbation anormale dans ce sous-espace
            Z_noise = Z[:, self.n_pc_selected:]
            X_recon = Z_noise @ self.pca.components_[self.n_pc_selected:]
            residual = X_scaled - X_recon

        # Score = norme L2 du résidu, normalisée en [0,1]
        scores = np.linalg.norm(residual, axis=1)
        if scores.max() > 0:
            scores = scores / scores.max()
        return scores

    def score(self, X: np.ndarray) -> np.ndarray:
        """Retourne le score d'anomalie ∈ [0,1] pour chaque IP."""
        X_scaled = self.scaler.transform(X)
        return self._compute_scores(X_scaled)

    def predict(self, X: np.ndarray, threshold: float = None) -> np.ndarray:
        """Prédit 1 (Botnet) ou 0 (non-Botnet) selon le seuil."""
        thr = threshold if threshold is not None else self.threshold
        return (self.score(X) >= thr).astype(int)

    def find_best_threshold(self, X: np.ndarray, y: np.ndarray) -> float:
        """Trouve le seuil qui maximise le F1-score."""
        scores = self.score(X)
        best_f1, best_thr = 0.0, 0.5
        for thr in np.linspace(0.01, 0.99, 200):
            preds = (scores >= thr).astype(int)
            f1 = f1_score(y, preds, zero_division=0)
            if f1 > best_f1:
                best_f1, best_thr = f1, thr
        self.threshold = best_thr
        return best_thr


# =============================================================================
# 3. Évaluation
# =============================================================================

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                    y_score: np.ndarray = None, name: str = "") -> dict:
    """
    Calcule les métriques du papier CTU-13 (Table 10) :
      TPR, TNR, FPR, FNR, Precision, F1, AUC-ROC
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0,1]).ravel()
    total = tn + fp + fn + tp

    tpr  = tp / (tp + fn) if (tp + fn) > 0 else 0   # Recall / Sensitivity
    tnr  = tn / (tn + fp) if (tn + fp) > 0 else 0   # Specificity
    fpr  = fp / (fp + tn) if (fp + tn) > 0 else 0
    fnr  = fn / (fn + tp) if (fn + tp) > 0 else 0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    acc  = (tp + tn) / total if total > 0 else 0
    f1   = 2 * prec * tpr / (prec + tpr) if (prec + tpr) > 0 else 0

    auc = float(roc_auc_score(y_true, y_score)) if y_score is not None else float("nan")

    metrics = {
        "Method": name,
        "TP": int(tp), "TN": int(tn), "FP": int(fp), "FN": int(fn),
        "TPR": round(tpr,  4),
        "TNR": round(tnr,  4),
        "FPR": round(fpr,  4),
        "FNR": round(fnr,  4),
        "Precision": round(prec, 4),
        "Accuracy":  round(acc,  4),
        "F1":        round(f1,   4),
        "AUC-ROC":   round(auc,  4),
    }
    return metrics


def print_metrics_table(all_metrics: list[dict]) -> None:
    """Affiche un tableau de métriques similaire à la Table 10 du papier."""
    print(f"\n{'Method':<15} {'TPR':>6} {'TNR':>6} {'FPR':>6} {'FNR':>6} "
          f"{'Prec':>6} {'F1':>6} {'AUC':>6}")
    print("-" * 65)
    for m in all_metrics:
        print(f"{m['Method']:<15} {m['TPR']:>6.3f} {m['TNR']:>6.3f} {m['FPR']:>6.3f} "
              f"{m['FNR']:>6.3f} {m['Precision']:>6.3f} {m['F1']:>6.3f} {m['AUC-ROC']:>6.3f}")


# =============================================================================
# 4. Pipeline principal
# =============================================================================

def run_pipeline(df: pd.DataFrame) -> None:

    y = df["y_binary"].values
    n_botnet = y.sum()
    print(f"\n      Botnet IPs : {n_botnet} / {len(y)} ({n_botnet/len(y)*100:.2f}%)")

    # -------------------------------------------------------------------------
    # Préparation des features KGB
    # -------------------------------------------------------------------------
    X_kgb = get_X(df, KGB_FEATURES)
    X_all = get_X(df, ALL_FEATURES)
    X_all = np.nan_to_num(X_all, nan=0.0)

    # -------------------------------------------------------------------------
    # [2/6] KGBf — composantes haute variance
    # -------------------------------------------------------------------------
    print("\n[2/6] KGBf (composantes haute variance)...")

    kgbf = KGBDetector(fog=False, variance_threshold=0.90)
    kgbf.fit(X_kgb)
    kgbf.find_best_threshold(X_kgb, y)

    scores_f  = kgbf.score(X_kgb)
    preds_f   = kgbf.predict(X_kgb)
    metrics_f = compute_metrics(y, preds_f, scores_f, name="KGBf")
    print(f"      Seuil optimal : {kgbf.threshold:.4f}")
    print(f"      F1={metrics_f['F1']:.3f}  TPR={metrics_f['TPR']:.3f}  "
          f"FPR={metrics_f['FPR']:.3f}  AUC={metrics_f['AUC-ROC']:.3f}")

    # -------------------------------------------------------------------------
    # [3/6] KGBfog — composantes basse variance
    # -------------------------------------------------------------------------
    print("\n[3/6] KGBfog (composantes basse variance)...")

    kgbfog = KGBDetector(fog=True, variance_threshold=0.90)
    kgbfog.fit(X_kgb)
    kgbfog.find_best_threshold(X_kgb, y)

    scores_fog  = kgbfog.score(X_kgb)
    preds_fog   = kgbfog.predict(X_kgb)
    metrics_fog = compute_metrics(y, preds_fog, scores_fog, name="KGBfog")
    print(f"      Seuil optimal : {kgbfog.threshold:.4f}")
    print(f"      F1={metrics_fog['F1']:.3f}  TPR={metrics_fog['TPR']:.3f}  "
          f"FPR={metrics_fog['FPR']:.3f}  AUC={metrics_fog['AUC-ROC']:.3f}")

    # -------------------------------------------------------------------------
    # [4/6] Classifieurs ML supervisés (baseline de comparaison)
    # -------------------------------------------------------------------------
    print("\n[4/6] Classifieurs ML supervisés (cross-validation 5-fold)...")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scaler_ml = StandardScaler()
    X_scaled  = scaler_ml.fit_transform(X_all)

    cw = compute_class_weight("balanced", classes=np.array([0,1]), y=y)
    class_weights = {0: cw[0], 1: cw[1]}

    classifiers = {
        "RandomForest": RandomForestClassifier(
            n_estimators=200, class_weight="balanced",
            max_depth=10, random_state=42, n_jobs=-1
        ),
        "SVM (RBF)": SVC(
            kernel="rbf", class_weight="balanced",
            probability=True, random_state=42, C=10
        ),
        "LogisticReg": LogisticRegression(
            class_weight="balanced", max_iter=1000, random_state=42
        ),
        "IsolationForest": None,   # traitement séparé (non supervisé)
    }

    all_metrics = [metrics_f, metrics_fog]

    for name, clf in classifiers.items():
        if name == "IsolationForest":
            # Isolation Forest : non supervisé, entraîné sur tout
            iso = IsolationForest(contamination=n_botnet/len(y),
                                  random_state=42, n_jobs=-1)
            iso.fit(X_scaled)
            preds_iso  = (iso.predict(X_scaled) == -1).astype(int)
            scores_iso = -iso.score_samples(X_scaled)
            scores_iso = (scores_iso - scores_iso.min()) / (scores_iso.max() - scores_iso.min())
            m = compute_metrics(y, preds_iso, scores_iso, name="IsoForest")
        else:
            preds_cv  = cross_val_predict(clf, X_scaled, y, cv=cv, method="predict")
            scores_cv = cross_val_predict(clf, X_scaled, y, cv=cv, method="predict_proba")[:,1]
            m = compute_metrics(y, preds_cv, scores_cv, name=name)

        all_metrics.append(m)
        print(f"      {name:<16} F1={m['F1']:.3f}  TPR={m['TPR']:.3f}  "
              f"FPR={m['FPR']:.3f}  AUC={m['AUC-ROC']:.3f}")

    # -------------------------------------------------------------------------
    # [5/6] Tableau de comparaison final
    # -------------------------------------------------------------------------
    print("\n[5/6] Tableau de comparaison (style Table 10 du papier CTU-13)")
    print_metrics_table(all_metrics)

    results_df = pd.DataFrame(all_metrics)
    results_df.to_csv(OUTPUT_DIR/"metrics_comparison.csv", index=False)
    print(f"\n      💾 metrics_comparison.csv")

    # -------------------------------------------------------------------------
    # [6/6] Visualisations
    # -------------------------------------------------------------------------
    print("\n[6/6] Génération des graphiques...")

    _plot_roc_curves(df, X_kgb, X_scaled, y, kgbf, kgbfog, classifiers)
    _plot_score_distributions(scores_f, scores_fog, y)
    _plot_pca_analysis(X_kgb, y, kgbf)
    _plot_metrics_comparison(all_metrics)
    _write_final_report(all_metrics, kgbf, kgbfog, n_botnet, len(y))


def _plot_roc_curves(df, X_kgb, X_scaled, y, kgbf, kgbfog, classifiers):
    """Courbes ROC pour tous les modèles."""
    fig, ax = plt.subplots(figsize=(9, 7))

    # KGB
    for detector, name, color in [(kgbf, "KGBf", "#d62728"),
                                   (kgbfog, "KGBfog", "#ff7f0e")]:
        scores = detector.score(X_kgb)
        fpr, tpr, _ = roc_curve(y, scores)
        auc = roc_auc_score(y, scores)
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})", lw=2, color=color)

    # Classifieurs supervisés
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    palette = ["#2ca02c", "#1f77b4", "#9467bd", "#8c564b"]
    for (name, clf), color in zip(
        [(n,c) for n,c in classifiers.items() if c is not None], palette
    ):
        scores_cv = cross_val_predict(clf, X_scaled, y, cv=cv, method="predict_proba")[:,1]
        fpr, tpr, _ = roc_curve(y, scores_cv)
        auc = roc_auc_score(y, scores_cv)
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})", lw=1.5, color=color)

    ax.plot([0,1],[0,1], "k--", lw=0.8, label="Random")
    ax.set_xlabel("False Positive Rate (FPR)")
    ax.set_ylabel("True Positive Rate (TPR / Recall)")
    ax.set_title("Courbes ROC — Détection Botnet Scénario 8 CTU-13")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR/"04_roc_curves.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"      📊 04_roc_curves.png")


def _plot_score_distributions(scores_f, scores_fog, y):
    """Distribution des scores KGB pour Botnet vs non-Botnet."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, scores, title in [
        (axes[0], scores_f,   "KGBf (haute variance)"),
        (axes[1], scores_fog, "KGBfog (basse variance)"),
    ]:
        ax.hist(scores[y==0], bins=50, alpha=0.6, label="Non-Botnet",
                color="#1f77b4", density=True)
        ax.hist(scores[y==1], bins=20, alpha=0.8, label="Botnet",
                color="#d62728", density=True)
        ax.set_title(f"Score d'anomalie — {title}")
        ax.set_xlabel("Score KGB [0,1]")
        ax.set_ylabel("Densité")
        ax.legend()

    plt.suptitle("Distribution des scores KGB par classe", fontsize=13)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR/"05_kgb_score_distributions.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"      📊 05_kgb_score_distributions.png")


def _plot_pca_analysis(X_kgb, y, kgbf):
    """Visualise la variance expliquée et la projection PCA 2D."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Variance expliquée
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X_kgb)
    pca_full = PCA()
    pca_full.fit(X_s)

    axes[0].bar(range(1, len(pca_full.explained_variance_ratio_)+1),
                pca_full.explained_variance_ratio_,
                color="#1f77b4", alpha=0.8)
    axes[0].plot(range(1, len(pca_full.explained_variance_ratio_)+1),
                 np.cumsum(pca_full.explained_variance_ratio_),
                 "r-o", markersize=5, label="Variance cumulée")
    axes[0].axhline(0.90, color="gray", ls="--", label="Seuil 90%")
    axes[0].set_title("Variance expliquée par composante PCA")
    axes[0].set_xlabel("Composante principale")
    axes[0].set_ylabel("Variance expliquée")
    axes[0].legend()
    axes[0].set_xticks(range(1, len(pca_full.explained_variance_ratio_)+1))

    # Projection 2D
    pca_2d = PCA(n_components=2)
    Z_2d = pca_2d.fit_transform(X_s)
    colors = {0: "#aec7e8", 1: "#d62728"}
    labels_plot = {0: "Non-Botnet", 1: "Botnet"}
    for cls in [0, 1]:
        mask = y == cls
        axes[1].scatter(Z_2d[mask, 0], Z_2d[mask, 1],
                        c=colors[cls], label=f"{labels_plot[cls]} (n={mask.sum()})",
                        alpha=0.6 if cls==0 else 1.0,
                        s=10 if cls==0 else 80,
                        marker="." if cls==0 else "*")
    axes[1].set_title("Projection PCA 2D des features KGB")
    axes[1].set_xlabel(f"PC1 ({pca_2d.explained_variance_ratio_[0]*100:.1f}%)")
    axes[1].set_ylabel(f"PC2 ({pca_2d.explained_variance_ratio_[1]*100:.1f}%)")
    axes[1].legend()

    plt.suptitle("Analyse PCA — Features KGB (Scénario 8)", fontsize=13)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR/"06_pca_analysis.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"      📊 06_pca_analysis.png")


def _plot_metrics_comparison(all_metrics: list[dict]) -> None:
    """Barplot comparatif des métriques clés."""
    names   = [m["Method"]    for m in all_metrics]
    f1s     = [m["F1"]        for m in all_metrics]
    tprs    = [m["TPR"]       for m in all_metrics]
    fprs    = [m["FPR"]       for m in all_metrics]
    aucs    = [m["AUC-ROC"]   for m in all_metrics]

    x    = np.arange(len(names))
    w    = 0.2
    fig, ax = plt.subplots(figsize=(12, 6))

    ax.bar(x - 1.5*w, f1s,  w, label="F1-score",  color="#2ca02c", alpha=0.85)
    ax.bar(x - 0.5*w, tprs, w, label="TPR",        color="#1f77b4", alpha=0.85)
    ax.bar(x + 0.5*w, fprs, w, label="FPR",        color="#d62728", alpha=0.85)
    ax.bar(x + 1.5*w, aucs, w, label="AUC-ROC",    color="#ff7f0e", alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15, ha="right")
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Score")
    ax.set_title("Comparaison des méthodes — Détection Botnet Scénario 8 CTU-13\n"
                 "(style Table 10 du papier García et al. 2014)")
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    ax.axhline(1.0, color="gray", ls="--", lw=0.5)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR/"07_metrics_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"      📊 07_metrics_comparison.png")


def _write_final_report(all_metrics, kgbf, kgbfog, n_botnet, n_total):
    """Rapport texte final."""
    rp = OUTPUT_DIR / "kgb_report.txt"
    with open(rp, "w") as f:
        f.write("="*70 + "\n")
        f.write("RAPPORT KGB — CTU-13 Scénario 8 (Botnet-49 / Murlo)\n")
        f.write("Pevný et al. (2012) + Comparaison ML\n")
        f.write("="*70 + "\n\n")

        f.write("PARAMÈTRES KGB\n" + "-"*40 + "\n")
        f.write(f"Features : {KGB_FEATURES}\n")
        f.write(f"KGBf   — PC haute variance, seuil optimal : {kgbf.threshold:.4f}\n")
        f.write(f"KGBfog — PC basse  variance, seuil optimal : {kgbfog.threshold:.4f}\n")
        f.write(f"IPs Botnet : {n_botnet} / {n_total} ({n_botnet/n_total*100:.3f}%)\n\n")

        f.write("MÉTRIQUES DE COMPARAISON\n" + "-"*40 + "\n")
        header = (f"{'Method':<16} {'TPR':>6} {'TNR':>6} {'FPR':>6} {'FNR':>6} "
                  f"{'Prec':>6} {'F1':>6} {'AUC':>6}\n")
        f.write(header)
        f.write("-"*65 + "\n")
        for m in all_metrics:
            f.write(f"{m['Method']:<16} {m['TPR']:>6.3f} {m['TNR']:>6.3f} "
                    f"{m['FPR']:>6.3f} {m['FNR']:>6.3f} {m['Precision']:>6.3f} "
                    f"{m['F1']:>6.3f} {m['AUC-ROC']:>6.3f}\n")

        best = max(all_metrics, key=lambda m: m["F1"])
        f.write(f"\nMeilleure méthode (F1) : {best['Method']} (F1={best['F1']:.3f})\n")

        f.write("\nINTERPRÉTATION\n" + "-"*40 + "\n")
        f.write("KGBf  : détecte les anomalies dans l'espace des PC dominantes.\n")
        f.write("        Efficace si le botnet génère un comportement très différent\n")
        f.write("        du trafic normal dans les directions de plus grande variance.\n\n")
        f.write("KGBfog: détecte les anomalies dans le 'bruit' résiduel (PC mineures).\n")
        f.write("        Efficace pour les botnets discrets qui se fondent dans le trafic\n")
        f.write("        normal sur les features dominantes.\n\n")
        f.write("Scénario 8 (Murlo) : botnet avec C&C propriétaire, scans DCERPC.\n")
        f.write("Entropies faibles → comportement répétitif et peu diversifié.\n")

    print(f"\n      📄 kgb_report.txt")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="KGB Detector — CTU-13 Scénario 8")
    parser.add_argument("--features", "-f",
                        default="eda_output/entropy_features.csv",
                        help="Chemin vers entropy_features.csv")
    args = parser.parse_args()

    print("\n" + "="*60)
    print("  KGB Detector — CTU-13 Scénario 8")
    print("="*60)

    df = load_features(args.features)
    run_pipeline(df)

    print("\n" + "="*60)
    print(f"  ✅ Pipeline terminé ! Résultats dans : {OUTPUT_DIR}/")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()