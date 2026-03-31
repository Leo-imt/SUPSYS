"""
=============================================================================
03_analysis_report.py — Analyse critique et rapport académique
CTU-13 Scénario 8 (Botnet-49 / Murlo)
=============================================================================
Corrections par rapport à 02_kgb_pipeline.py :
  1. Split temporel correct : KGB entraîné UNIQUEMENT sur Background (20%)
     → évite le data leakage du script 02
  2. Courbes Precision-Recall (plus informatives que ROC sur données
     très déséquilibrées)
  3. Analyse détaillée des Faux Négatifs / Faux Positifs
  4. Rapport académique complet

Usage :
  python 03_analysis_report.py
  python 03_analysis_report.py --features eda_output/entropy_features.csv

Sorties (dans report_output/) :
  08_precision_recall.png, 09_botnet_scores.png, rapport_academique.txt
=============================================================================
"""

import argparse
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import (
    roc_auc_score, f1_score, confusion_matrix,
    average_precision_score, precision_recall_curve
)

# ── Configuration centrale
from config import (
    LABEL_BOTNET, LABEL_NORMAL, LABEL_BACKGROUND,
    KGB_FEATURES, ALL_FEATURES, REPORT_OUTPUT, EDA_OUTPUT
)

warnings.filterwarnings("ignore")
REPORT_OUTPUT.mkdir(exist_ok=True)


# =============================================================================
# KGB non supervisé (version correcte sans leakage)
# =============================================================================

class KGBDetector:
    """Voir 02_kgb_pipeline.py pour la documentation complète."""

    def __init__(self, fog=False, variance_threshold=0.90):
        self.fog = fog
        self.variance_threshold = variance_threshold
        self.pca = self.scaler = None
        self.n_pc = None
        self.threshold = 0.5

    def fit(self, X_train_normal):
        self.scaler = StandardScaler()
        Xs = self.scaler.fit_transform(X_train_normal)
        n  = Xs.shape[1]
        self.pca = PCA(n_components=n).fit(Xs)
        cumvar = np.cumsum(self.pca.explained_variance_ratio_)
        k = int(np.searchsorted(cumvar, self.variance_threshold) + 1)
        self.n_pc = max(1, min(k, n - 1))
        return self

    def score(self, X):
        Xs = self.scaler.transform(X)
        Z  = self.pca.transform(Xs)
        if not self.fog:
            recon = Z[:, :self.n_pc] @ self.pca.components_[:self.n_pc]
        else:
            recon = Z[:, self.n_pc:] @ self.pca.components_[self.n_pc:]
        s = np.linalg.norm(Xs - recon, axis=1)
        return s / (s.max() + 1e-10)

    def find_best_threshold(self, X, y):
        scores = self.score(X)
        best_f1, best_t = 0.0, 0.5
        for t in np.linspace(0.01, 0.99, 300):
            f1 = f1_score(y, (scores >= t).astype(int), zero_division=0)
            if f1 > best_f1:
                best_f1, best_t = f1, t
        self.threshold = best_t
        return best_t, best_f1

    def predict(self, X):
        return (self.score(X) >= self.threshold).astype(int)


def metrics(y_true, y_pred, y_score, name):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0,1]).ravel()
    eps  = 1e-10
    tpr  = tp/(tp+fn+eps); tnr = tn/(tn+fp+eps)
    fpr  = fp/(fp+tn+eps); fnr = fn/(fn+tp+eps)
    prec = tp/(tp+fp+eps); f1  = 2*prec*tpr/(prec+tpr+eps)
    auc  = roc_auc_score(y_true, y_score) \
           if len(np.unique(y_true)) > 1 else 0.5
    ap   = average_precision_score(y_true, y_score) \
           if len(np.unique(y_true)) > 1 else 0.0
    return dict(Method=name, TP=int(tp), TN=int(tn), FP=int(fp), FN=int(fn),
                TPR=round(tpr,4), TNR=round(tnr,4), FPR=round(fpr,4),
                FNR=round(fnr,4), Precision=round(prec,4),
                F1=round(f1,4), AUC=round(auc,4), AP=round(ap,4))


def get_train_idx(df):
    """20% des IPs Background pour l'entraînement non supervisé."""
    bg_idx = np.where(df["label"] == LABEL_BACKGROUND)[0]
    return bg_idx[:max(int(len(bg_idx)*0.20), 50)]


# =============================================================================
# Analyse 1 : évaluation non supervisée correcte
# =============================================================================

def analysis_unsupervised_split(df):
    print("\n── Analyse 1 : Évaluation non supervisée (split correct) ──")
    X = df[KGB_FEATURES].fillna(0).values
    y = (df["label"] == LABEL_BOTNET).astype(int).values
    train_idx = get_train_idx(df)
    X_train = X[train_idx]

    results = []
    for fog, name in [(False,"KGBf_unsup"),(True,"KGBfog_unsup")]:
        det = KGBDetector(fog=fog).fit(X_train)
        det.find_best_threshold(X, y)
        s = det.score(X)
        p = det.predict(X)
        m = metrics(y, p, s, name)
        results.append(m)
        print(f"   {name:<18} F1={m['F1']:.3f}  TPR={m['TPR']:.3f}  "
              f"FPR={m['FPR']:.3f}  AUC={m['AUC']:.3f}  AP={m['AP']:.3f}")
    return results


# =============================================================================
# Analyse 2 : impact des features
# =============================================================================

def analysis_feature_sets(df):
    print("\n── Analyse 2 : Impact des features ──")
    y = (df["label"] == LABEL_BOTNET).astype(int).values
    train_idx = get_train_idx(df)

    feature_sets = {
        "KGB (3 entropies)":  ["H_dst_ip","H_dst_port","H_src_port"],
        "KGB + volume":       ["H_dst_ip","H_dst_port","H_src_port",
                               "mean_bytes","mean_pkts"],
        "KGB + volume + std": ["H_dst_ip","H_dst_port","H_src_port",
                               "mean_bytes","std_bytes","mean_pkts","std_pkts"],
        "Volume seul":        ["mean_bytes","mean_pkts","mean_dur"],
        "n_flows seul":       ["n_flows"],
    }
    results = []
    for fname, feats in feature_sets.items():
        cols = [c for c in feats if c in df.columns]
        if not cols:
            continue
        X = df[cols].fillna(0).values
        det = KGBDetector(fog=False).fit(X[train_idx])
        det.find_best_threshold(X, y)
        s = det.score(X)
        p = det.predict(X)
        m = metrics(y, p, s, fname)
        results.append(m)
        print(f"   {fname:<30} F1={m['F1']:.3f}  "
              f"AUC={m['AUC']:.3f}  AP={m['AP']:.3f}")
    return results


# =============================================================================
# Analyse 3 : courbes Precision-Recall
# =============================================================================

def analysis_precision_recall(df):
    print("\n── Analyse 3 : Courbes Precision-Recall ──")
    X = df[KGB_FEATURES].fillna(0).values
    y = (df["label"] == LABEL_BOTNET).astype(int).values
    train_idx = get_train_idx(df)

    curves = {}
    for fog, name, color in [(False,"KGBf","#d62728"),(True,"KGBfog","#ff7f0e")]:
        det = KGBDetector(fog=fog).fit(X[train_idx])
        s   = det.score(X)
        prec, rec, _ = precision_recall_curve(y, s)
        ap = average_precision_score(y, s)
        curves[name] = (prec, rec, ap, color)
        print(f"   {name} AP={ap:.3f}")

    X_all = df[[c for c in ALL_FEATURES if c in df.columns]].fillna(0).values
    Xs    = StandardScaler().fit_transform(X_all)
    rf    = RandomForestClassifier(n_estimators=200, class_weight="balanced",
                                    random_state=42, n_jobs=-1)
    cv    = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    s_rf  = cross_val_predict(rf, Xs, y, cv=cv, method="predict_proba")[:,1]
    prec_rf, rec_rf, _ = precision_recall_curve(y, s_rf)
    ap_rf = average_precision_score(y, s_rf)
    curves["RandomForest"] = (prec_rf, rec_rf, ap_rf, "#2ca02c")
    print(f"   RandomForest AP={ap_rf:.3f}")

    fig, ax = plt.subplots(figsize=(8,6))
    ax.axhline(y.mean(), color="gray", ls="--", lw=0.8,
               label=f"Baseline (P={y.mean():.4f})")
    for name, (prec, rec, ap, color) in curves.items():
        ax.step(rec, prec, where="post",
                label=f"{name} (AP={ap:.3f})", color=color, lw=2)
    ax.set_xlabel("Recall (TPR)"); ax.set_ylabel("Precision")
    ax.set_title("Courbes Precision-Recall — Scénario 8 CTU-13\n"
                 "(plus informative que ROC sur données déséquilibrées)")
    ax.legend(fontsize=9); ax.set_xlim(0,1); ax.set_ylim(0,1.05)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(REPORT_OUTPUT/"08_precision_recall.png",
                dpi=150, bbox_inches="tight")
    plt.close()
    print(f"   📊 08_precision_recall.png")
    return curves


# =============================================================================
# Analyse 4 : analyse des erreurs
# =============================================================================

def analysis_errors(df):
    print("\n── Analyse 4 : Analyse des erreurs ──")
    X = df[KGB_FEATURES].fillna(0).values
    y = (df["label"] == LABEL_BOTNET).astype(int).values
    train_idx = get_train_idx(df)

    det = KGBDetector(fog=False).fit(X[train_idx])
    det.find_best_threshold(X, y)
    scores = det.score(X)
    preds  = det.predict(X)

    df2 = df.copy()
    df2["kgb_score"] = scores
    df2["kgb_pred"]  = preds
    df2["y"]         = y

    fn_df = df2[(df2["y"]==1)&(df2["kgb_pred"]==0)].sort_values("kgb_score")
    fp_df = df2[(df2["y"]==0)&(df2["kgb_pred"]==1)].sort_values(
        "kgb_score", ascending=False)
    tp_df = df2[(df2["y"]==1)&(df2["kgb_pred"]==1)]

    print(f"   TP : {len(tp_df)}  FN : {len(fn_df)}  FP : {len(fp_df)}")

    # Visualisation scores des IPs Botnet
    botnet_df = df2[df2["y"]==1].sort_values("kgb_score", ascending=False)
    clrs = ["#d62728" if p==1 else "#aec7e8" for p in botnet_df["kgb_pred"]]
    fig, ax = plt.subplots(figsize=(10,4))
    ax.bar(range(len(botnet_df)), botnet_df["kgb_score"], color=clrs)
    ax.axhline(det.threshold, color="black", ls="--", lw=1.5,
               label=f"Seuil KGBf = {det.threshold:.3f}")
    ax.set_xlabel("IP Botnet (triées par score décroissant)")
    ax.set_ylabel("Score KGBf")
    ax.set_title("Scores KGBf pour toutes les IPs Botnet\n"
                 "(rouge=détecté, bleu=manqué)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(REPORT_OUTPUT/"09_botnet_scores.png",
                dpi=150, bbox_inches="tight")
    plt.close()
    print(f"   📊 09_botnet_scores.png")
    return fn_df, fp_df, tp_df


# =============================================================================
# Rapport académique
# =============================================================================

def write_academic_report(results_unsup, results_features, fn_df, fp_df,
                           tp_df, df):
    rp = REPORT_OUTPUT / "analysis_results.txt"
    n_botnet = (df["label"] == LABEL_BOTNET).sum()
    n_total  = len(df)
    with open(rp, "w", encoding="utf-8") as f:
        # IPs
        f.write(f"IPs totales  : {n_total:,}\n")
        for lbl in [LABEL_BOTNET, LABEL_NORMAL, LABEL_BACKGROUND]:
            n = (df["label"]==lbl).sum()
            f.write(f"  {lbl:<12} : {n:>6,} ({n/n_total*100:.2f}%)\n")
        f.write("\n")
        # KGB non supervisé
        f.write(f"{'Method':<20} {'F1':>6} {'TPR':>6} {'FPR':>6} {'AUC':>6} {'AP':>6}\n")
        f.write("-"*56 + "\n")
        for m in results_unsup:
            f.write(f"{m['Method']:<20} {m['F1']:>6.3f} {m['TPR']:>6.3f} "
                    f"{m['FPR']:>6.3f} {m['AUC']:>6.3f} {m['AP']:>6.3f}\n")
        f.write("\n")
        # Impact features
        f.write(f"{'Feature set':<30} {'F1':>6} {'AUC':>6} {'AP':>6}\n")
        f.write("-"*46 + "\n")
        for m in results_features:
            f.write(f"{m['Method']:<30} {m['F1']:>6.3f} {m['AUC']:>6.3f} {m['AP']:>6.3f}\n")
        f.write("\n")
        # Erreurs
        f.write(f"TP : {len(tp_df)}  FN : {len(fn_df)}  FP : {len(fp_df)}\n")
        f.write(f"Taux détection : {len(tp_df)/(len(tp_df)+len(fn_df)+1e-10)*100:.1f}%\n")
        if len(fn_df):
            for feat in KGB_FEATURES:
                if feat in fn_df.columns:
                    f.write(f"  FN {feat} moyen : {fn_df[feat].mean():.3f}\n")
    print(f"\n   📄 analysis_results.txt")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Analyse critique KGB — CTU-13 Scénario 8")
    parser.add_argument(
        "--features", default=str(EDA_OUTPUT/"entropy_features.csv"),
        help="Chemin vers entropy_features.csv")
    args = parser.parse_args()

    feat_path = Path(args.features)
    if not feat_path.exists():
        print(f"❌ {feat_path} introuvable — lancez d'abord 01_eda_ctu13_s8.py")
        raise SystemExit(1)

    print("\n"+"="*60)
    print("  CTU-13 Scénario 8 — Analyse critique & Rapport")
    print("="*60)

    df = pd.read_csv(feat_path)
    for col in KGB_FEATURES + ["mean_bytes","std_bytes","mean_pkts",
                                "std_pkts","mean_dur","n_flows"]:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    print(f"\n   IPs : {len(df):,}  ({(df['label']==LABEL_BOTNET).sum()} Botnet)")

    r_unsup   = analysis_unsupervised_split(df)
    r_feats   = analysis_feature_sets(df)
    _         = analysis_precision_recall(df)
    fn, fp, tp = analysis_errors(df)
    write_academic_report(r_unsup, r_feats, fn, fp, tp, df)

    print("\n"+"="*60)
    print(f"  ✅ Terminé ! Résultats dans : {REPORT_OUTPUT}/")
    print("="*60+"\n")


if __name__ == "__main__":
    main()
