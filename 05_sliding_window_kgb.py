"""
=============================================================================
05_sliding_window_kgb.py — KGB avec Fenêtres Temporelles Glissantes
CTU-13 Scénario 8 (Botnet-49 / Murlo)
=============================================================================
Implémente l'approche temporelle du papier García et al. (2014) :
  - Découpage en fenêtres de W minutes (défaut : 5 min comme dans le papier)
  - Pour chaque fenêtre : agrégation par IP, entropies, score KGB
  - Warm-up sur les N premières fenêtres (Background uniquement)
  - Mise à jour adaptative du modèle PCA toutes les N fenêtres
  - Produit les "running metrics" (Fig. 12 du papier)

Usage :
  python 05_sliding_window_kgb.py                     # utilise config.py
  python 05_sliding_window_kgb.py --file /chemin/...  # chemin explicite
  python 05_sliding_window_kgb.py --window 5 --warmup 5 --max-rows 2000000

Sorties (dans sliding_output/) :
  sliding_kgbf.csv, sliding_kgbfog.csv, sliding_report.txt
  11_running_metrics.png, 12_window_analysis.png
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
import matplotlib.dates as mdates
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler

# ── Configuration centrale
from config import (
    COLUMN_NAMES, LABEL_BOTNET, LABEL_NORMAL, LABEL_BACKGROUND,
    KGB_FEATURES, SLIDING_OUTPUT, get_data_path
)

warnings.filterwarnings("ignore")
SLIDING_OUTPUT.mkdir(exist_ok=True)


# =============================================================================
# Chargement et parsing
# =============================================================================

def classify_label(label: str) -> str:
    l = str(label).lower()
    if "botnet"     in l: return LABEL_BOTNET
    if "normal"     in l or "legitimate" in l: return LABEL_NORMAL
    if "background" in l: return LABEL_BACKGROUND
    return LABEL_BACKGROUND


def load_and_parse(path: Path, max_rows: int = None) -> pd.DataFrame:
    """Charge et parse le fichier binetflow avec parsing temporel."""
    print(f"\n[1/4] Chargement : {path}")

    df = pd.read_csv(
        path, sep="\t", skiprows=1, header=None,
        names=COLUMN_NAMES, nrows=max_rows,
        engine="c", dtype=str, on_bad_lines="skip",
    )

    # Split IP:port
    for raw, addr, port in [
        ("SrcAddrPort","SrcAddr","Sport"),
        ("DstAddrPort","DstAddr","Dport"),
    ]:
        split = df[raw].str.rsplit(":", n=1, expand=True)
        df[addr] = split[0].str.strip()
        df[port] = split[1].str.strip() if 1 in split.columns else ""

    # Parsing temporel — format "2011-08-16 14:18:55.889"
    df["StartTime"]  = pd.to_datetime(df["StartTime"], errors="coerce")
    df = df.dropna(subset=["StartTime"])

    for col in ["TotBytes","TotPkts","Dur"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    df["LabelClean"] = df["Label"].apply(classify_label)

    print(f"      Flows chargés  : {len(df):,}")
    print(f"      Période        : {df['StartTime'].min()} "
          f"→ {df['StartTime'].max()}")
    duration_h = (df['StartTime'].max()-df['StartTime'].min()
                  ).total_seconds()/3600
    print(f"      Durée totale   : {duration_h:.1f} heures")
    for lbl in [LABEL_BOTNET, LABEL_NORMAL, LABEL_BACKGROUND]:
        n = (df["LabelClean"]==lbl).sum()
        print(f"      {lbl:<12}: {n:>8,} ({n/len(df)*100:.2f}%)")
    return df


# =============================================================================
# Agrégation par fenêtre temporelle
# =============================================================================

def shannon_entropy(series: pd.Series) -> float:
    vals = series.dropna().astype(str)
    if len(vals) == 0:
        return 0.0
    p = vals.value_counts(normalize=True).values
    return float(-np.sum(p * np.log2(p + 1e-10)))


def aggregate_window(window_df: pd.DataFrame) -> pd.DataFrame:
    """Agrège les NetFlows d'une fenêtre de W minutes par IP source."""
    if len(window_df) == 0:
        return pd.DataFrame()
    rows = []
    for src, g in window_df.groupby("SrcAddr"):
        if not src or pd.isna(src):
            continue
        lbl = (LABEL_BOTNET     if (g["LabelClean"]==LABEL_BOTNET).any() else
               LABEL_NORMAL     if (g["LabelClean"]==LABEL_NORMAL).any() else
               LABEL_BACKGROUND)
        rows.append({
            "SrcAddr":    src,
            "n_flows":    len(g),
            "H_dst_ip":   shannon_entropy(g["DstAddr"]),
            "H_dst_port": shannon_entropy(g["Dport"]),
            "H_src_port": shannon_entropy(g["Sport"]),
            "mean_bytes": g["TotBytes"].mean(),
            "mean_pkts":  g["TotPkts"].mean(),
            "label":      lbl,
        })
    return pd.DataFrame(rows) if rows else pd.DataFrame()


# =============================================================================
# KGB adaptatif avec sliding window
# =============================================================================

class AdaptiveKGB:
    """
    KGB avec mise à jour incrémentale du modèle PCA.
    Entraîné sur les N premières fenêtres (Background uniquement),
    puis mis à jour toutes les update_every fenêtres.
    """

    def __init__(self, fog=False, variance_threshold=0.90,
                 update_every=5):
        self.fog            = fog
        self.variance_threshold = variance_threshold
        self.update_every   = update_every
        self.pca = self.scaler = None
        self.n_pc    = None
        self.threshold = 0.15
        self._bg_buffer   = []
        self._window_count = 0

    def _fit_model(self, X_bg: np.ndarray) -> None:
        if len(X_bg) < 3:
            return
        self.scaler = StandardScaler()
        Xs = self.scaler.fit_transform(X_bg)
        n  = Xs.shape[1]
        pca = PCA(n_components=n).fit(Xs)
        cumvar = np.cumsum(pca.explained_variance_ratio_)
        k = int(np.searchsorted(cumvar, self.variance_threshold) + 1)
        self.n_pc = max(1, min(k, n - 1))
        self.pca  = pca

    def warmup(self, warmup_windows: list) -> None:
        """Entraîne le modèle sur les fenêtres initiales (Background)."""
        X_bg_all = []
        for wdf in warmup_windows:
            if len(wdf) == 0:
                continue
            bg = wdf[wdf["label"] == LABEL_BACKGROUND]
            if len(bg) > 0:
                X_bg_all.append(bg[KGB_FEATURES].fillna(0).values)
        if X_bg_all:
            self._fit_model(np.vstack(X_bg_all))
            n_ips = sum(len(x) for x in X_bg_all)
            print(f"      Warm-up : modèle entraîné sur {n_ips} IPs Background")

    def score_window(self, window_agg: pd.DataFrame) -> np.ndarray:
        """Score d'anomalie pour les IPs d'une fenêtre."""
        if self.pca is None or len(window_agg) == 0:
            return np.zeros(len(window_agg))
        X  = window_agg[KGB_FEATURES].fillna(0).values
        Xs = self.scaler.transform(X)
        Z  = self.pca.transform(Xs)
        if not self.fog:
            recon = Z[:, :self.n_pc] @ self.pca.components_[:self.n_pc]
        else:
            recon = Z[:, self.n_pc:] @ self.pca.components_[self.n_pc:]
        s = np.linalg.norm(Xs - recon, axis=1)
        mx = s.max()
        return s / (mx + 1e-10) if mx > 0 else s

    def update(self, window_agg: pd.DataFrame) -> None:
        """Mise à jour adaptative du modèle."""
        self._window_count += 1
        bg = window_agg[window_agg["label"] == LABEL_BACKGROUND]
        if len(bg) > 0:
            self._bg_buffer.append(bg[KGB_FEATURES].fillna(0).values)
        if (self._window_count % self.update_every == 0
                and self._bg_buffer):
            X_bg = np.vstack(self._bg_buffer[-self.update_every*5:])
            self._fit_model(X_bg)
            self._bg_buffer = self._bg_buffer[-self.update_every*5:]


# =============================================================================
# Pipeline sliding window
# =============================================================================

def run_sliding_window(df: pd.DataFrame, window_min: int,
                        warmup_n: int, fog: bool = False) -> pd.DataFrame:
    """Applique KGB sur des fenêtres glissantes de window_min minutes."""
    name = "KGBfog" if fog else "KGBf"
    print(f"\n   Running {name} "
          f"(window={window_min}min, warmup={warmup_n} fenêtres)...")

    df = df.sort_values("StartTime").reset_index(drop=True)
    t_start = df["StartTime"].min()
    t_end   = df["StartTime"].max()
    window_td = pd.Timedelta(minutes=window_min)

    # Génération des fenêtres
    windows = []
    t = t_start
    while t < t_end:
        mask = (df["StartTime"] >= t) & (df["StartTime"] < t + window_td)
        wdf  = df[mask]
        if len(wdf) > 0:
            agg = aggregate_window(wdf)
            windows.append((t, agg))
        t += window_td

    print(f"      {len(windows)} fenêtres de {window_min} min générées")

    # Warm-up
    kgb = AdaptiveKGB(fog=fog)
    kgb.warmup([w[1] for w in windows[:warmup_n]])

    # Évaluation fenêtre par fenêtre
    results = []
    for i, (t_win, wdf) in enumerate(windows):
        if len(wdf) == 0 or kgb.pca is None:
            continue

        y_true = (wdf["label"] == LABEL_BOTNET).astype(int).values
        scores = kgb.score_window(wdf)
        preds  = (scores >= kgb.threshold).astype(int)
        n_bot  = y_true.sum()
        n_tot  = len(y_true)

        if n_tot > 0 and len(np.unique(y_true)) > 1:
            tn,fp,fn,tp = confusion_matrix(
                y_true, preds, labels=[0,1]).ravel()
        else:
            tp = fp = fn = 0
            tn = (preds==0).sum()

        eps  = 1e-10
        tpr  = tp / (tp + fn + eps)
        fpr  = fp / (fp + tn + eps)
        prec = tp / (tp + fp + eps)
        f1   = 2*prec*tpr / (prec + tpr + eps)

        results.append({
            "window_start":   t_win,
            "n_flows_window": n_tot,
            "n_botnet_ips":   int(n_bot),
            "TP":int(tp),"FP":int(fp),"FN":int(fn),"TN":int(tn),
            "TPR":tpr,"FPR":fpr,"F1":f1,
        })
        kgb.update(wdf)

    results_df = pd.DataFrame(results)
    results_df.to_csv(SLIDING_OUTPUT/f"sliding_{name.lower()}.csv",
                      index=False)

    print(f"      {len(results_df)} fenêtres évaluées")
    bot_windows = results_df[results_df["n_botnet_ips"] > 0]
    if len(bot_windows):
        print(f"      TPR moyen (fenêtres avec botnet) : "
              f"{bot_windows['TPR'].mean():.3f}")
        print(f"      FPR moyen : {results_df['FPR'].mean():.4f}")
    return results_df


# =============================================================================
# Graphiques
# =============================================================================

def plot_running_metrics(res_f: pd.DataFrame, res_fog: pd.DataFrame,
                          window_min: int) -> None:
    """Reproduit les running metrics de la Fig. 12 du papier."""
    fig, axes = plt.subplots(3, 1, figsize=(14,12), sharex=True)

    for results, name, col in [
        (res_f,   "KGBf",   "#d62728"),
        (res_fog, "KGBfog", "#ff7f0e"),
    ]:
        if len(results) == 0:
            continue
        t = results["window_start"]
        axes[0].plot(t, results["TPR"], color=col, lw=1.5, alpha=0.8,
                     label=name)
        axes[1].plot(t, results["FPR"], color=col, lw=1.5, alpha=0.8)
        axes[2].plot(t, results["TP"].cumsum(), color=col, lw=2,
                     label=name)

    axes[0].set_ylabel("TPR (True Positive Rate)")
    axes[0].set_title(f"Running TPR — Fenêtres de {window_min} min\n"
                       "(style Fig. 12, García et al. 2014)")
    axes[0].set_ylim(-0.05, 1.05)
    axes[0].legend(); axes[0].grid(True, alpha=0.3)
    axes[0].axhline(0, color="gray", ls="--", lw=0.5)

    axes[1].set_ylabel("FPR (False Positive Rate)")
    axes[1].set_title(f"Running FPR — Fenêtres de {window_min} min")
    axes[1].set_ylim(-0.005, 0.1)
    axes[1].grid(True, alpha=0.3)

    axes[2].set_ylabel("TP cumulatifs")
    axes[2].set_title("IPs Botnet détectées (cumulatif)")
    axes[2].legend(); axes[2].grid(True, alpha=0.3)

    for ax in axes:
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))
    axes[2].set_xlabel("Heure (2011-08-16)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(SLIDING_OUTPUT/"11_running_metrics.png",
                dpi=150, bbox_inches="tight")
    plt.close()
    print(f"      📊 11_running_metrics.png")


def plot_window_heatmap(res_f: pd.DataFrame) -> None:
    if len(res_f) == 0:
        return
    bot = res_f[res_f["n_botnet_ips"] > 0]
    if len(bot) == 0:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14,5))

    axes[0].hist(bot["TPR"], bins=20, color="#d62728", alpha=0.7,
                 edgecolor="black", lw=0.5)
    axes[0].set_xlabel("TPR par fenêtre"); axes[0].set_ylabel("Nb fenêtres")
    axes[0].set_title("Distribution du TPR\n"
                       "(fenêtres contenant des IPs Botnet)")
    axes[0].axvline(bot["TPR"].mean(), color="black", ls="--",
                    label=f"Moyenne = {bot['TPR'].mean():.3f}")
    axes[0].legend()

    sc = axes[1].scatter(res_f["FPR"], res_f["TPR"],
                          c=res_f["n_botnet_ips"],
                          cmap="Reds", alpha=0.6, s=30)
    plt.colorbar(sc, ax=axes[1], label="Nb IPs Botnet / fenêtre")
    axes[1].set_xlabel("FPR"); axes[1].set_ylabel("TPR")
    axes[1].set_title("TPR vs FPR par fenêtre temporelle")
    axes[1].grid(True, alpha=0.3)

    plt.suptitle("Analyse par fenêtre temporelle — KGBf", fontsize=13)
    plt.tight_layout()
    plt.savefig(SLIDING_OUTPUT/"12_window_analysis.png",
                dpi=150, bbox_inches="tight")
    plt.close()
    print(f"      📊 12_window_analysis.png")


def write_sliding_report(res_f: pd.DataFrame, res_fog: pd.DataFrame,
                          window_min: int, warmup_n: int) -> None:
    rp = SLIDING_OUTPUT / "sliding_results.txt"
    with open(rp, "w") as f:
        f.write(f"window={window_min}min  warmup={warmup_n}\n\n")
for res, name in [(res_f, "KGBf"), (res_fog, "KGBfog")]:
    if len(res) == 0:
        continue
    bot = res[res["n_botnet_ips"] > 0]
    f.write(f"[{name}]\n")
    f.write(f"  windows_total   : {len(res)}\n")
    f.write(f"  windows_botnet  : {len(bot)}\n")
    if len(bot):
        f.write(f"  TPR_mean        : {bot['TPR'].mean():.3f}\n")
        f.write(f"  TPR_max         : {bot['TPR'].max():.3f}\n")
        f.write(f"  TPR_min         : {bot['TPR'].min():.3f}\n")
    f.write(f"  FPR_mean        : {res['FPR'].mean():.4f}\n")
    f.write(f"  TP_total        : {res['TP'].sum()}\n")
    f.write(f"  FN_total        : {res['FN'].sum()}\n")
    f.write(f"  FP_total        : {res['FP'].sum()}\n\n")
    print(f"      📄 sliding_results.txt")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="KGB Sliding Window — CTU-13 Scénario 8")
    parser.add_argument("--file","-f", default=None,
                        help="Chemin vers le .labeled (défaut : config.py)")
    parser.add_argument("--window","-w", type=int, default=5,
                        help="Largeur de fenêtre en minutes (défaut : 5)")
    parser.add_argument("--warmup","-u", type=int, default=5,
                        help="Nombre de fenêtres warm-up (défaut : 5)")
    parser.add_argument("--max-rows","-n", type=int, default=None,
                        help="Limiter le nombre de lignes (test rapide)")
    args = parser.parse_args()

    data_path = get_data_path(args.file)

    print("\n"+"="*60)
    print("  KGB Sliding Window — CTU-13 Scénario 8")
    print("="*60)

    df  = load_and_parse(data_path, max_rows=args.max_rows)

    print(f"\n[2/4] Sliding Window "
          f"(W={args.window}min, warmup={args.warmup})...")
    res_f   = run_sliding_window(df, args.window, args.warmup, fog=False)
    res_fog = run_sliding_window(df, args.window, args.warmup, fog=True)

    print("\n[3/4] Graphiques running metrics...")
    plot_running_metrics(res_f, res_fog, args.window)
    plot_window_heatmap(res_f)

    print("\n[4/4] Rapport...")
    write_sliding_report(res_f, res_fog, args.window, args.warmup)

    print("\n"+"="*60)
    print(f"  ✅ Terminé ! Résultats dans : {SLIDING_OUTPUT}/")
    print("="*60+"\n")


if __name__ == "__main__":
    main()
