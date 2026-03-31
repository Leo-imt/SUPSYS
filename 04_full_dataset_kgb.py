"""
=============================================================================
04_full_dataset_kgb.py — KGB sur le Dataset Complet (~12M flows)
CTU-13 Scénario 8 (Botnet-49 / Murlo)
=============================================================================
Stratégie mémoire pour 8-16 GB RAM :
  - Lecture par chunks de 500K lignes (pd.read_csv chunksize)
  - Agrégation incrémentale par IP (IncrementalAggregator)
  - gc.collect() après chaque chunk → pic mémoire ~3-5 GB
  - Calcul des entropies en une seule passe finale

Usage :
  python 04_full_dataset_kgb.py                       # utilise config.py
  python 04_full_dataset_kgb.py --file /chemin/...    # chemin explicite
  python 04_full_dataset_kgb.py --chunk-size 300000   # si peu de RAM

Sorties (dans full_output/) :
  full_entropy_features.csv, full_metrics.csv, full_report.txt
  10_full_dataset_results.png
=============================================================================
"""

import argparse
import gc
import time
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    f1_score, roc_auc_score, confusion_matrix,
    average_precision_score, roc_curve, precision_recall_curve
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler

# ── Configuration centrale
from config import (
    COLUMN_NAMES, LABEL_BOTNET, LABEL_NORMAL, LABEL_BACKGROUND,
    KGB_FEATURES, ALL_FEATURES, FULL_OUTPUT, get_data_path
)

warnings.filterwarnings("ignore")
FULL_OUTPUT.mkdir(exist_ok=True)


# =============================================================================
# Agrégation incrémentale (évite de tout charger en RAM)
# =============================================================================

class IncrementalAggregator:
    """
    Accumule les listes de valeurs par IP source chunk par chunk.
    Empreinte mémoire : ~3-5 GB pour 12M flows (vs ~10-14 GB intégral).
    """

    def __init__(self):
        self.dst_ips   = defaultdict(list)
        self.dst_ports = defaultdict(list)
        self.src_ports = defaultdict(list)
        self.bytes_    = defaultdict(list)
        self.pkts_     = defaultdict(list)
        self.durs_     = defaultdict(list)
        self.labels    = defaultdict(set)
        self.n_flows   = defaultdict(int)

    def update_fast(self, chunk: pd.DataFrame) -> None:
        """Agrégation vectorisée par groupby."""
        chunk = chunk.dropna(subset=["SrcAddr"])
        for src, g in chunk.groupby("SrcAddr"):
            self.n_flows[src] += len(g)
            self.labels[src].update(g["LabelClean"].dropna().unique().tolist())
            if "DstAddr" in g.columns:
                self.dst_ips[src].extend(
                    g["DstAddr"].dropna().astype(str).tolist())
            if "Dport" in g.columns:
                self.dst_ports[src].extend(
                    g["Dport"].dropna().astype(str).tolist())
            if "Sport" in g.columns:
                self.src_ports[src].extend(
                    g["Sport"].dropna().astype(str).tolist())
            if "TotBytes" in g.columns:
                self.bytes_[src].extend(
                    pd.to_numeric(g["TotBytes"],errors="coerce")
                    .dropna().tolist())
            if "TotPkts" in g.columns:
                self.pkts_[src].extend(
                    pd.to_numeric(g["TotPkts"],errors="coerce")
                    .dropna().tolist())
            if "Dur" in g.columns:
                self.durs_[src].extend(
                    pd.to_numeric(g["Dur"],errors="coerce")
                    .dropna().tolist())

    def compute_features(self) -> pd.DataFrame:
        """Calcule toutes les features à partir des données accumulées."""
        print("      Calcul des features sur toutes les IPs...")

        def H(lst):
            if not lst:
                return 0.0
            _, counts = np.unique(lst, return_counts=True)
            p = counts / counts.sum()
            return float(-np.sum(p * np.log2(p + 1e-10)))

        rows = []
        all_ips = list(self.n_flows.keys())
        for i, src in enumerate(all_ips):
            if i % 10000 == 0:
                print(f"        {i:>7}/{len(all_ips)} IPs...", end="\r")
            lbls = self.labels[src]
            label = (LABEL_BOTNET     if LABEL_BOTNET     in lbls else
                     LABEL_NORMAL     if LABEL_NORMAL     in lbls else
                     LABEL_BACKGROUND)
            b = self.bytes_[src]; p_ = self.pkts_[src]; d = self.durs_[src]
            rows.append({
                "src_ip":     src,
                "n_flows":    self.n_flows[src],
                "H_dst_ip":   H(self.dst_ips[src]),
                "H_dst_port": H(self.dst_ports[src]),
                "H_src_port": H(self.src_ports[src]),
                "mean_bytes": float(np.mean(b)) if b else 0.0,
                "std_bytes":  float(np.std(b))  if b else 0.0,
                "mean_pkts":  float(np.mean(p_)) if p_ else 0.0,
                "std_pkts":   float(np.std(p_))  if p_ else 0.0,
                "mean_dur":   float(np.mean(d))  if d else 0.0,
                "label":      label,
            })
        print()
        return pd.DataFrame(rows)


# =============================================================================
# Parsing
# =============================================================================

def classify_label(label: str) -> str:
    l = str(label).lower()
    if "botnet"     in l: return LABEL_BOTNET
    if "normal"     in l or "legitimate" in l: return LABEL_NORMAL
    if "background" in l: return LABEL_BACKGROUND
    return LABEL_BACKGROUND


def process_chunk(chunk: pd.DataFrame) -> pd.DataFrame:
    chunk.columns = COLUMN_NAMES[:len(chunk.columns)]
    for raw, addr, port in [
        ("SrcAddrPort","SrcAddr","Sport"),
        ("DstAddrPort","DstAddr","Dport"),
    ]:
        if raw in chunk.columns:
            split = chunk[raw].str.rsplit(":", n=1, expand=True)
            chunk[addr] = split[0].str.strip()
            chunk[port] = split[1].str.strip() if 1 in split.columns else ""
    chunk["LabelClean"] = chunk["Label"].apply(classify_label)
    return chunk


# =============================================================================
# Chargement complet par chunks
# =============================================================================

def load_full_dataset(path: Path, chunk_size: int) -> pd.DataFrame:
    print(f"\n[1/5] Lecture par chunks de {chunk_size:,}...")
    t0 = time.time()
    agg = IncrementalAggregator()
    total_rows = total_botnet = chunk_num = 0

    reader = pd.read_csv(
        path, sep="\t", skiprows=1, header=None,
        names=COLUMN_NAMES, chunksize=chunk_size,
        engine="c", dtype=str, on_bad_lines="skip",
    )
    for chunk in reader:
        chunk_num += 1
        chunk = process_chunk(chunk)
        n_bot = (chunk["LabelClean"] == LABEL_BOTNET).sum()
        total_botnet += n_bot
        total_rows   += len(chunk)
        print(f"      Chunk {chunk_num:>2} : {len(chunk):>7,} flows "
              f"| Botnet: {n_bot:>5} | Total: {total_rows:>10,}", end="\r")
        agg.update_fast(chunk)
        del chunk
        gc.collect()

    elapsed = time.time() - t0
    print(f"\n      ✅ {total_rows:,} flows en {elapsed:.1f}s")
    print(f"      Flows Botnet : {total_botnet:,}")

    print("\n[2/5] Calcul des features par IP...")
    df = agg.compute_features()
    print(f"\n      IPs uniques : {len(df):,}")
    for lbl in [LABEL_BOTNET, LABEL_NORMAL, LABEL_BACKGROUND]:
        n = (df["label"]==lbl).sum()
        print(f"        {lbl:<12}: {n:,}")

    df.to_csv(FULL_OUTPUT/"full_entropy_features.csv", index=False)
    print(f"      💾 full_entropy_features.csv")
    return df


# =============================================================================
# KGB + Random Forest
# =============================================================================

class KGBDetector:
    def __init__(self, fog=False, variance_threshold=0.90):
        self.fog=fog; self.variance_threshold=variance_threshold
        self.pca=self.scaler=None; self.n_pc=None; self.threshold=0.5

    def fit(self, X):
        self.scaler=StandardScaler(); Xs=self.scaler.fit_transform(X)
        n=Xs.shape[1]; self.pca=PCA(n_components=n).fit(Xs)
        cumvar=np.cumsum(self.pca.explained_variance_ratio_)
        k=int(np.searchsorted(cumvar,self.variance_threshold)+1)
        self.n_pc=max(1,min(k,n-1)); return self

    def score(self, X):
        Xs=self.scaler.transform(X); Z=self.pca.transform(Xs)
        recon=(Z[:,:self.n_pc]@self.pca.components_[:self.n_pc]
               if not self.fog
               else Z[:,self.n_pc:]@self.pca.components_[self.n_pc:])
        s=np.linalg.norm(Xs-recon,axis=1)
        return s/(s.max()+1e-10)

    def find_best_threshold(self, X, y):
        s=self.score(X); best_f1,best_t=0.0,0.5
        for t in np.linspace(0.01,0.99,300):
            f1=f1_score(y,(s>=t).astype(int),zero_division=0)
            if f1>best_f1: best_f1,best_t=f1,t
        self.threshold=best_t; return best_t,best_f1

    def predict(self, X):
        return (self.score(X)>=self.threshold).astype(int)


def metrics_dict(y_true, y_pred, y_score, name):
    tn,fp,fn,tp=confusion_matrix(y_true,y_pred,labels=[0,1]).ravel()
    eps=1e-10
    tpr=tp/(tp+fn+eps); tnr=tn/(tn+fp+eps)
    fpr=fp/(fp+tn+eps); fnr=fn/(fn+tp+eps)
    prec=tp/(tp+fp+eps); f1=2*prec*tpr/(prec+tpr+eps)
    auc=roc_auc_score(y_true,y_score) if y_true.sum()>0 else 0.5
    ap=average_precision_score(y_true,y_score) if y_true.sum()>0 else 0.0
    return dict(Method=name,TP=int(tp),TN=int(tn),FP=int(fp),FN=int(fn),
                TPR=round(tpr,4),TNR=round(tnr,4),FPR=round(fpr,4),
                FNR=round(fnr,4),Precision=round(prec,4),
                F1=round(f1,4),AUC=round(auc,4),AP=round(ap,4))


def run_full_pipeline(df: pd.DataFrame) -> None:
    X_kgb = df[KGB_FEATURES].fillna(0).values
    X_all = df[[c for c in ALL_FEATURES if c in df.columns]].fillna(0).values
    y     = (df["label"]==LABEL_BOTNET).astype(int).values
    n_b   = y.sum()
    print(f"\n      Botnet : {n_b} / {len(y)} ({n_b/len(y)*100:.3f}%)")

    bg_idx  = np.where(df["label"]==LABEL_BACKGROUND)[0]
    n_train = max(int(len(bg_idx)*0.20), 200)
    X_train = X_kgb[bg_idx[:n_train]]
    print(f"      Train Background : {n_train:,} IPs")

    all_metrics = []

    print("\n[3/5] KGBf + KGBfog (dataset complet)...")
    for fog, name in [(False,"KGBf"),(True,"KGBfog")]:
        det = KGBDetector(fog=fog).fit(X_train)
        det.find_best_threshold(X_kgb, y)
        s=det.score(X_kgb); p=det.predict(X_kgb)
        m=metrics_dict(y,p,s,name); all_metrics.append(m)
        print(f"      {name:<10} F1={m['F1']:.3f}  TPR={m['TPR']:.3f}  "
              f"FPR={m['FPR']:.3f}  AUC={m['AUC']:.3f}  AP={m['AP']:.3f}")

    print("\n[4/5] Random Forest supervisé (5-fold CV)...")
    scaler=StandardScaler(); Xs=scaler.fit_transform(X_all)
    rf=RandomForestClassifier(n_estimators=300,class_weight="balanced",
                               max_depth=12,random_state=42,n_jobs=-1)
    cv=StratifiedKFold(n_splits=5,shuffle=True,random_state=42)
    p_rf=cross_val_predict(rf,Xs,y,cv=cv,method="predict")
    s_rf=cross_val_predict(rf,Xs,y,cv=cv,method="predict_proba")[:,1]
    m_rf=metrics_dict(y,p_rf,s_rf,"RandomForest"); all_metrics.append(m_rf)
    print(f"      RandomForest  F1={m_rf['F1']:.3f}  TPR={m_rf['TPR']:.3f}  "
          f"FPR={m_rf['FPR']:.3f}  AUC={m_rf['AUC']:.3f}  AP={m_rf['AP']:.3f}")

    # Tableau
    print(f"\n{'Method':<14} {'F1':>6} {'TPR':>6} {'FPR':>6} "
          f"{'AUC':>6} {'AP':>6}")
    print("-"*46)
    for m in all_metrics:
        print(f"{m['Method']:<14} {m['F1']:>6.3f} {m['TPR']:>6.3f} "
              f"{m['FPR']:>6.3f} {m['AUC']:>6.3f} {m['AP']:>6.3f}")

    pd.DataFrame(all_metrics).to_csv(FULL_OUTPUT/"full_metrics.csv",
                                      index=False)
    print("\n[5/5] Graphiques...")
    _plot_full_results(df, X_kgb, Xs, y, all_metrics, rf, cv)
    _write_report(all_metrics, n_b, len(y))


def _plot_full_results(df, X_kgb, Xs, y, all_metrics, rf, cv):
    fig, axes = plt.subplots(2, 2, figsize=(14,10))
    colors={"KGBf":"#d62728","KGBfog":"#ff7f0e","RandomForest":"#2ca02c"}

    bg_idx  = np.where(df["label"]==LABEL_BACKGROUND)[0]
    X_train = X_kgb[bg_idx[:max(int(len(bg_idx)*0.20),200)]]

    # ROC
    ax=axes[0,0]
    for m in all_metrics:
        name=m["Method"]
        if name in ("KGBf","KGBfog"):
            det=KGBDetector(fog=(name=="KGBfog")).fit(X_train)
            s=det.score(X_kgb)
        else:
            s=cross_val_predict(rf,Xs,y,cv=cv,method="predict_proba")[:,1]
        fp_,tp_,_=roc_curve(y,s)
        ax.plot(fp_,tp_,label=f"{name} (AUC={m['AUC']:.3f})",
                color=colors.get(name,"#1f77b4"),lw=2)
    ax.plot([0,1],[0,1],"k--",lw=0.8)
    ax.set_title("Courbes ROC — Dataset Complet (~12M flows)")
    ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
    ax.legend(fontsize=9); ax.grid(True,alpha=0.3)

    # PR
    ax=axes[0,1]
    ax.axhline(y.mean(),color="gray",ls="--",lw=0.8,
               label=f"Baseline (P={y.mean():.5f})")
    for m in all_metrics:
        name=m["Method"]
        if name in ("KGBf","KGBfog"):
            det=KGBDetector(fog=(name=="KGBfog")).fit(X_train)
            s=det.score(X_kgb)
        else:
            s=cross_val_predict(rf,Xs,y,cv=cv,method="predict_proba")[:,1]
        prec,rec,_=precision_recall_curve(y,s)
        ax.step(rec,prec,where="post",
                label=f"{name} (AP={m['AP']:.4f})",
                color=colors.get(name,"#1f77b4"),lw=2)
    ax.set_title("Precision-Recall — Dataset Complet")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.legend(fontsize=9); ax.grid(True,alpha=0.3)

    # Distribution H_dst_port
    ax=axes[1,0]
    c_map={LABEL_BOTNET:"#d62728",LABEL_NORMAL:"#2ca02c",
           LABEL_BACKGROUND:"#aec7e8"}
    for lbl in [LABEL_BOTNET,LABEL_NORMAL,LABEL_BACKGROUND]:
        sub=df[df["label"]==lbl]["H_dst_port"].dropna()
        if len(sub):
            s=sub.sample(min(len(sub),5000),random_state=42)
            ax.hist(s,bins=50,alpha=0.6,label=f"{lbl} (n={len(sub):,})",
                    color=c_map[lbl],density=True)
    ax.set_title("Distribution H(dst_port) — Dataset Complet")
    ax.set_xlabel("Entropie (bits)"); ax.set_ylabel("Densité")
    ax.legend(fontsize=9)

    # Barplot métriques
    ax=axes[1,1]
    names=[m["Method"] for m in all_metrics]
    x=np.arange(len(names)); w=0.25
    ax.bar(x-w,[m["F1"]  for m in all_metrics],w,label="F1",
           color="#2ca02c",alpha=0.85)
    ax.bar(x,  [m["AUC"] for m in all_metrics],w,label="AUC",
           color="#1f77b4",alpha=0.85)
    ax.bar(x+w,[m["AP"]  for m in all_metrics],w,label="AP",
           color="#ff7f0e",alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(names,rotation=10)
    ax.set_ylim(0,1.1); ax.set_ylabel("Score")
    ax.set_title("Métriques — Dataset Complet")
    ax.legend(); ax.grid(axis="y",alpha=0.3)

    plt.suptitle("KGB — Dataset Complet CTU-13 Scénario 8 (~12M flows)",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(FULL_OUTPUT/"10_full_dataset_results.png",
                dpi=150, bbox_inches="tight")
    plt.close()
    print(f"      📊 10_full_dataset_results.png")


def _write_report(all_metrics, n_botnet, n_total):
    rp = FULL_OUTPUT / "full_results.txt"
    with open(rp, "w") as f:
        f.write(f"IPs analysées : {n_total:,}\n")
        f.write(f"IPs Botnet    : {n_botnet} ({n_botnet/n_total*100:.3f}%)\n\n")
        f.write(f"{'Method':<14} {'F1':>6} {'TPR':>6} {'FPR':>6} {'AUC':>6} {'AP':>6}\n")
        f.write("-"*46 + "\n")
        for m in all_metrics:
            f.write(f"{m['Method']:<14} {m['F1']:>6.3f} {m['TPR']:>6.3f} "
                    f"{m['FPR']:>6.3f} {m['AUC']:>6.3f} {m['AP']:>6.3f}\n")
        best = max(all_metrics, key=lambda m: m["F1"])
        f.write(f"\nBest (F1) : {best['Method']} — F1={best['F1']:.3f}  AUC={best['AUC']:.3f}\n")
    print(f"      📄 full_results.txt")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="KGB Dataset Complet — CTU-13 Scénario 8")
    parser.add_argument("--file","-f", default=None,
                        help="Chemin vers le .labeled (défaut : config.py)")
    parser.add_argument("--chunk-size", type=int, default=500_000,
                        help="Taille des chunks en lignes (défaut : 500000)")
    args = parser.parse_args()

    data_path = get_data_path(args.file)

    print("\n"+"="*60)
    print("  KGB — Dataset Complet CTU-13 Scénario 8")
    print("="*60)

    df = load_full_dataset(data_path, chunk_size=args.chunk_size)
    run_full_pipeline(df)

    print("\n"+"="*60)
    print(f"  ✅ Terminé ! Résultats dans : {FULL_OUTPUT}/")
    print("="*60+"\n")


if __name__ == "__main__":
    main()
