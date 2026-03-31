"""
=============================================================================
02_kgb_pipeline.py — Implémentation KGB + Pipeline ML
CTU-13 Scénario 8 (Botnet-49 / Murlo)
=============================================================================
Implémente les deux variantes du détecteur KGB (Pevný et al., 2012) :
  - KGBf   : anomalie dans les composantes PCA à HAUTE variance
  - KGBfog : anomalie dans les composantes PCA à BASSE  variance

Lit les features produites par 01_eda_ctu13_s8.py (entropy_features.csv).

Usage :
  python 02_kgb_pipeline.py
  python 02_kgb_pipeline.py --features eda_output/entropy_features.csv

Sorties (dans kgb_output/) :
  metrics_comparison.csv, kgb_report.txt
  04_roc_curves.png, 05_kgb_score_distributions.png,
  06_pca_analysis.png, 07_metrics_comparison.png
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
    confusion_matrix, roc_auc_score, roc_curve,
    precision_recall_curve, f1_score, average_precision_score
)
from sklearn.utils.class_weight import compute_class_weight

# ── Configuration centrale
from config import (
    LABEL_BOTNET, LABEL_NORMAL, LABEL_BACKGROUND,
    KGB_FEATURES, ALL_FEATURES, KGB_OUTPUT, EDA_OUTPUT
)

warnings.filterwarnings("ignore")
KGB_OUTPUT.mkdir(exist_ok=True)


# =============================================================================
# Détecteur KGB
# =============================================================================

class KGBDetector:
    """
    Détecteur d'anomalies KGB (Pevný et al., 2012).

    Algorithme :
      1. Normalisation StandardScaler sur données Background
      2. PCA complète sur les données normalisées
      3. Sélection des k composantes (haute ou basse variance)
      4. Résidu = X_normalisé - reconstruction depuis les k PC
      5. Score = ‖résidu‖₂ normalisé ∈ [0,1]

    Paramètres :
      fog               : False → KGBf (PC haute variance)
                          True  → KGBfog (PC basse variance)
      variance_threshold: seuil de variance cumulée pour sélection auto des PC
    """

    def __init__(self, fog: bool = False, variance_threshold: float = 0.90):
        self.fog               = fog
        self.variance_threshold = variance_threshold
        self.pca     = None
        self.scaler  = None
        self.n_pc    = None
        self.threshold = 0.5

    def fit(self, X_train_normal: np.ndarray) -> "KGBDetector":
        """Entraîne sur du trafic supposé normal (non supervisé)."""
        self.scaler = StandardScaler()
        Xs = self.scaler.fit_transform(X_train_normal)
        n  = Xs.shape[1]
        self.pca = PCA(n_components=n).fit(Xs)
        cumvar = np.cumsum(self.pca.explained_variance_ratio_)
        k = int(np.searchsorted(cumvar, self.variance_threshold) + 1)
        self.n_pc = max(1, min(k, n - 1))
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        """Score d'anomalie ∈ [0,1] pour chaque IP."""
        Xs = self.scaler.transform(X)
        Z  = self.pca.transform(Xs)
        if not self.fog:
            recon = Z[:, :self.n_pc] @ self.pca.components_[:self.n_pc]
        else:
            recon = Z[:, self.n_pc:] @ self.pca.components_[self.n_pc:]
        s = np.linalg.norm(Xs - recon, axis=1)
        return s / (s.max() + 1e-10)

    def find_best_threshold(self, X: np.ndarray,
                             y: np.ndarray) -> tuple[float, float]:
        """Trouve le seuil maximisant le F1-score."""
        scores = self.score(X)
        best_f1, best_t = 0.0, 0.5
        for t in np.linspace(0.01, 0.99, 300):
            f1 = f1_score(y, (scores >= t).astype(int), zero_division=0)
            if f1 > best_f1:
                best_f1, best_t = f1, t
        self.threshold = best_t
        return best_t, best_f1

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.score(X) >= self.threshold).astype(int)


# =============================================================================
# Métriques
# =============================================================================

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                    y_score: np.ndarray, name: str) -> dict:
    """Calcule les métriques du papier CTU-13 (Table 10)."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0,1]).ravel()
    eps  = 1e-10
    tpr  = tp / (tp + fn + eps)
    tnr  = tn / (tn + fp + eps)
    fpr  = fp / (fp + tn + eps)
    fnr  = fn / (fn + tp + eps)
    prec = tp / (tp + fp + eps)
    f1   = 2 * prec * tpr / (prec + tpr + eps)
    auc  = float(roc_auc_score(y_true, y_score)) \
           if len(np.unique(y_true)) > 1 else 0.5
    ap   = float(average_precision_score(y_true, y_score)) \
           if len(np.unique(y_true)) > 1 else 0.0
    return dict(
        Method=name, TP=int(tp), TN=int(tn), FP=int(fp), FN=int(fn),
        TPR=round(tpr,4), TNR=round(tnr,4), FPR=round(fpr,4),
        FNR=round(fnr,4), Precision=round(prec,4),
        F1=round(f1,4), AUC=round(auc,4), AP=round(ap,4)
    )


def print_metrics_table(all_metrics: list[dict]) -> None:
    print(f"\n{'Method':<16} {'F1':>6} {'TPR':>6} {'TNR':>6} "
          f"{'FPR':>6} {'Prec':>6} {'AUC':>6} {'AP':>6}")
    print("-"*65)
    for m in all_metrics:
        print(f"{m['Method']:<16} {m['F1']:>6.3f} {m['TPR']:>6.3f} "
              f"{m['TNR']:>6.3f} {m['FPR']:>6.3f} {m['Precision']:>6.3f} "
              f"{m['AUC']:>6.3f} {m['AP']:>6.3f}")


# =============================================================================
# Pipeline principal
# =============================================================================

def run_pipeline(df: pd.DataFrame) -> None:
    # Préparation features
    for col in ALL_FEATURES:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    X_kgb = df[[c for c in KGB_FEATURES if c in df.columns]].values
    X_all = df[[c for c in ALL_FEATURES  if c in df.columns]].fillna(0).values
    y     = (df["label"] == LABEL_BOTNET).astype(int).values

    n_botnet = y.sum()
    print(f"\n      IPs Botnet : {n_botnet} / {len(y)} ({n_botnet/len(y)*100:.2f}%)")

    # Entraînement non supervisé : Background uniquement (20%)
    bg_idx  = np.where(df["label"] == LABEL_BACKGROUND)[0]
    n_train = max(int(len(bg_idx) * 0.20), 50)
    X_train = X_kgb[bg_idx[:n_train]]

    all_metrics = []

    # ── KGBf et KGBfog
    print("\n[2/6] KGBf + KGBfog...")
    for fog, name in [(False, "KGBf"), (True, "KGBfog")]:
        det = KGBDetector(fog=fog)
        det.fit(X_train)
        best_t, best_f1 = det.find_best_threshold(X_kgb, y)
        s = det.score(X_kgb)
        p = det.predict(X_kgb)
        m = compute_metrics(y, p, s, name)
        all_metrics.append(m)
        print(f"      {name:<10} F1={m['F1']:.3f}  TPR={m['TPR']:.3f}  "
              f"FPR={m['FPR']:.3f}  AUC={m['AUC']:.3f}  AP={m['AP']:.3f}")

    # ── Classifieurs ML supervisés
    print("\n[3/6] Classifieurs ML supervisés (5-fold CV)...")
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X_all)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    classifiers = {
        "RandomForest": RandomForestClassifier(
            n_estimators=200, class_weight="balanced",
            max_depth=10, random_state=42, n_jobs=-1),
        "SVM (RBF)": SVC(
            kernel="rbf", class_weight="balanced",
            probability=True, random_state=42, C=10),
        "LogisticReg": LogisticRegression(
            class_weight="balanced", max_iter=1000, random_state=42),
    }
    for cname, clf in classifiers.items():
        preds  = cross_val_predict(clf, Xs, y, cv=cv, method="predict")
        scores = cross_val_predict(clf, Xs, y, cv=cv,
                                   method="predict_proba")[:, 1]
        m = compute_metrics(y, preds, scores, cname)
        all_metrics.append(m)
        print(f"      {cname:<16} F1={m['F1']:.3f}  TPR={m['TPR']:.3f}  "
              f"FPR={m['FPR']:.3f}  AUC={m['AUC']:.3f}")

    # ── IsolationForest
    iso = IsolationForest(contamination=max(n_botnet/len(y), 0.001),
                          random_state=42, n_jobs=-1)
    iso.fit(Xs)
    p_iso = (iso.predict(Xs) == -1).astype(int)
    s_iso = -iso.score_samples(Xs)
    s_iso = (s_iso - s_iso.min()) / (s_iso.max() - s_iso.min() + 1e-10)
    m_iso = compute_metrics(y, p_iso, s_iso, "IsoForest")
    all_metrics.append(m_iso)
    print(f"      {'IsoForest':<16} F1={m_iso['F1']:.3f}  "
          f"AUC={m_iso['AUC']:.3f}")

    # ── Tableau final
    print("\n[4/6] Tableau de comparaison (style Table 10 du papier CTU-13)")
    print_metrics_table(all_metrics)

    pd.DataFrame(all_metrics).to_csv(
        KGB_OUTPUT / "metrics_comparison.csv", index=False)

    # ── Graphiques
    print("\n[5/6] Génération des graphiques...")
    _plot_roc(df, X_kgb, Xs, y, all_metrics, classifiers, cv, scaler)
    _plot_score_distributions(
        KGBDetector(fog=False).fit(X_train).score(X_kgb),
        KGBDetector(fog=True ).fit(X_train).score(X_kgb),
        y)
    _plot_pca_analysis(X_kgb, y)
    _plot_metrics_comparison(all_metrics)

    _write_report(all_metrics, n_botnet, len(y))


def _plot_roc(df, X_kgb, Xs, y, all_metrics, classifiers, cv, scaler):
    fig, ax = plt.subplots(figsize=(9, 7))
    colors  = {"KGBf":"#d62728","KGBfog":"#ff7f0e",
               "RandomForest":"#2ca02c","SVM (RBF)":"#1f77b4",
               "LogisticReg":"#9467bd"}
    bg_idx  = np.where(df["label"] == LABEL_BACKGROUND)[0]
    X_train = X_kgb[bg_idx[:max(int(len(bg_idx)*0.20),50)]]

    for fog, name in [(False,"KGBf"),(True,"KGBfog")]:
        det = KGBDetector(fog=fog).fit(X_train)
        s   = det.score(X_kgb)
        fp_, tp_, _ = roc_curve(y, s)
        ax.plot(fp_, tp_,
                label=f"{name} (AUC={roc_auc_score(y,s):.3f})",
                color=colors[name], lw=2)

    for cname, clf in classifiers.items():
        from sklearn.model_selection import cross_val_predict as cvp
        s = cvp(clf, Xs, y, cv=cv, method="predict_proba")[:,1]
        fp_, tp_, _ = roc_curve(y, s)
        ax.plot(fp_, tp_,
                label=f"{cname} (AUC={roc_auc_score(y,s):.3f})",
                color=colors.get(cname,"gray"), lw=1.5)

    ax.plot([0,1],[0,1],"k--",lw=0.8,label="Random")
    ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
    ax.set_title("Courbes ROC — Scénario 8 CTU-13")
    ax.legend(loc="lower right",fontsize=9)
    ax.grid(True,alpha=0.3)
    plt.tight_layout()
    plt.savefig(KGB_OUTPUT/"04_roc_curves.png",dpi=150,bbox_inches="tight")
    plt.close()
    print(f"      📊 04_roc_curves.png")


def _plot_score_distributions(scores_f, scores_fog, y):
    fig, axes = plt.subplots(1,2,figsize=(12,5))
    for ax, sc, title in [
        (axes[0],scores_f,  "KGBf (haute variance)"),
        (axes[1],scores_fog,"KGBfog (basse variance)"),
    ]:
        ax.hist(sc[y==0],bins=50,alpha=0.6,label="Non-Botnet",
                color="#1f77b4",density=True)
        ax.hist(sc[y==1],bins=20,alpha=0.8,label="Botnet",
                color="#d62728",density=True)
        ax.set_title(f"Score — {title}")
        ax.set_xlabel("Score KGB [0,1]"); ax.set_ylabel("Densité")
        ax.legend()
    plt.suptitle("Distribution des scores KGB",fontsize=13)
    plt.tight_layout()
    plt.savefig(KGB_OUTPUT/"05_kgb_score_distributions.png",
                dpi=150,bbox_inches="tight")
    plt.close()
    print(f"      📊 05_kgb_score_distributions.png")


def _plot_pca_analysis(X_kgb, y):
    fig, axes = plt.subplots(1,2,figsize=(12,5))
    sc = StandardScaler()
    Xs = sc.fit_transform(X_kgb)
    pca = PCA().fit(Xs)
    axes[0].bar(range(1,len(pca.explained_variance_ratio_)+1),
                pca.explained_variance_ratio_,color="#1f77b4",alpha=0.8)
    axes[0].plot(range(1,len(pca.explained_variance_ratio_)+1),
                 np.cumsum(pca.explained_variance_ratio_),
                 "r-o",markersize=5,label="Variance cumulée")
    axes[0].axhline(0.90,color="gray",ls="--",label="Seuil 90%")
    axes[0].set_title("Variance expliquée par PC")
    axes[0].legend()

    pca2 = PCA(n_components=2)
    Z = pca2.fit_transform(Xs)
    for cls, col, lbl, mk, sz in [
        (0,"#aec7e8","Non-Botnet",".","5"),
        (1,"#d62728","Botnet","*","80"),
    ]:
        m = y == cls
        axes[1].scatter(Z[m,0],Z[m,1],c=col,
                        label=f"{lbl} (n={m.sum()})",
                        alpha=0.6,s=int(sz),
                        marker=mk)
    axes[1].set_title("Projection PCA 2D")
    axes[1].set_xlabel(f"PC1 ({pca2.explained_variance_ratio_[0]*100:.1f}%)")
    axes[1].set_ylabel(f"PC2 ({pca2.explained_variance_ratio_[1]*100:.1f}%)")
    axes[1].legend()
    plt.suptitle("Analyse PCA — Features KGB",fontsize=13)
    plt.tight_layout()
    plt.savefig(KGB_OUTPUT/"06_pca_analysis.png",
                dpi=150,bbox_inches="tight")
    plt.close()
    print(f"      📊 06_pca_analysis.png")


def _plot_metrics_comparison(all_metrics):
    names = [m["Method"] for m in all_metrics]
    f1s   = [m["F1"]     for m in all_metrics]
    tprs  = [m["TPR"]    for m in all_metrics]
    fprs  = [m["FPR"]    for m in all_metrics]
    aucs  = [m["AUC"]    for m in all_metrics]
    x = np.arange(len(names)); w = 0.2
    fig, ax = plt.subplots(figsize=(12,6))
    ax.bar(x-1.5*w,f1s, w,label="F1",    color="#2ca02c",alpha=0.85)
    ax.bar(x-0.5*w,tprs,w,label="TPR",   color="#1f77b4",alpha=0.85)
    ax.bar(x+0.5*w,fprs,w,label="FPR",   color="#d62728",alpha=0.85)
    ax.bar(x+1.5*w,aucs,w,label="AUC",   color="#ff7f0e",alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(names,rotation=15,ha="right")
    ax.set_ylim(0,1.15); ax.set_ylabel("Score")
    ax.set_title("Comparaison méthodes — Scénario 8 CTU-13\n"
                 "(style Table 10, García et al. 2014)")
    ax.legend(); ax.grid(axis="y",alpha=0.3)
    plt.tight_layout()
    plt.savefig(KGB_OUTPUT/"07_metrics_comparison.png",
                dpi=150,bbox_inches="tight")
    plt.close()
    print(f"      📊 07_metrics_comparison.png")


def _write_report(all_metrics, n_botnet, n_total):
    rp = KGB_OUTPUT / "kgb_results.txt"
    with open(rp, "w") as f:
        f.write(f"IPs Botnet : {n_botnet} / {n_total} ({n_botnet/n_total*100:.3f}%)\n")
        f.write(f"{'Method':<16} {'F1':>6} {'TPR':>6} {'TNR':>6} {'FPR':>6} {'Prec':>6} {'AUC':>6} {'AP':>6}\n")
        f.write("-"*65 + "\n")
        for m in all_metrics:
            f.write(f"{m['Method']:<16} {m['F1']:>6.3f} {m['TPR']:>6.3f} "
                    f"{m['TNR']:>6.3f} {m['FPR']:>6.3f} {m['Precision']:>6.3f} "
                    f"{m['AUC']:>6.3f} {m['AP']:>6.3f}\n")
        best = max(all_metrics, key=lambda m: m["F1"])
        f.write(f"\nBest (F1) : {best['Method']} — F1={best['F1']:.3f}  AUC={best['AUC']:.3f}\n")
    print(f"      📄 kgb_results.txt")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="KGB Detector — CTU-13 Scénario 8")
    parser.add_argument(
        "--features", "-f",
        default=str(EDA_OUTPUT / "entropy_features.csv"),
        help="Chemin vers entropy_features.csv "
             "(produit par 01_eda_ctu13_s8.py)")
    args = parser.parse_args()

    feat_path = Path(args.features)
    if not feat_path.exists():
        print(f"❌ Fichier introuvable : {feat_path}")
        print("   Lancez d'abord : python 01_eda_ctu13_s8.py")
        raise SystemExit(1)

    print("\n" + "="*60)
    print("  KGB Detector — CTU-13 Scénario 8")
    print("="*60)

    df = pd.read_csv(feat_path)
    print(f"\n[1/6] Features chargées : {len(df):,} IPs "
          f"({(df['label']==LABEL_BOTNET).sum()} Botnet)")

    run_pipeline(df)

    print("\n" + "="*60)
    print(f"  ✅ Pipeline terminé ! Résultats dans : {KGB_OUTPUT}/")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
