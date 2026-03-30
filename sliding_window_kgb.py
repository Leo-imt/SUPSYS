"""
=============================================================================
CTU-13 Scénario 8 — KGB avec Fenêtres Temporelles Glissantes
=============================================================================
Implémente l'approche temporelle du papier García et al. (2014) :
  - Fenêtres de W minutes (défaut : 5 min, comme dans le papier)
  - Pour chaque fenêtre : agrégation par IP, calcul entropies, score KGB
  - "Running metrics" : TPR/FPR évoluant dans le temps (Fig. 12 du papier)
  - KGB entraîné sur les N premières fenêtres de Background (warm-up)

Usage :
  python 05_sliding_window_kgb.py --file capture20110816-3.pcap.netflow.labeled
  python 05_sliding_window_kgb.py --file ... --window 5 --warmup 5
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

warnings.filterwarnings("ignore")

OUTPUT_DIR = Path("sliding_output")
OUTPUT_DIR.mkdir(exist_ok=True)

COLUMN_NAMES = [
    "StartTime", "Dur", "Proto",
    "SrcAddrPort", "Dir", "DstAddrPort",
    "Flags", "Tos", "TotPkts", "TotBytes", "Flows", "Label"
]

LABEL_BOTNET     = "Botnet"
LABEL_NORMAL     = "Normal"
LABEL_BACKGROUND = "Background"
KGB_FEATURES     = ["H_dst_ip", "H_dst_port", "H_src_port"]


# =============================================================================
# Chargement et parsing
# =============================================================================

def classify_label(label: str) -> str:
    l = str(label).lower()
    if "botnet"     in l: return LABEL_BOTNET
    if "normal"     in l or "legitimate" in l: return LABEL_NORMAL
    if "background" in l: return LABEL_BACKGROUND
    return LABEL_BACKGROUND


def load_and_parse(path: str, max_rows: int = None) -> pd.DataFrame:
    """Charge le fichier et parse les colonnes."""
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
    df["StartTime"] = pd.to_datetime(df["StartTime"], errors="coerce")
    df = df.dropna(subset=["StartTime"])

    # Conversions numériques
    for col in ["TotBytes","TotPkts","Dur"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    df["LabelClean"] = df["Label"].apply(classify_label)

    print(f"      Flows chargés  : {len(df):,}")
    print(f"      Période        : {df['StartTime'].min()} → {df['StartTime'].max()}")
    print(f"      Durée totale   : "
          f"{(df['StartTime'].max()-df['StartTime'].min()).total_seconds()/3600:.1f} heures")

    lc = df["LabelClean"].value_counts()
    for lbl, cnt in lc.items():
        print(f"      {lbl:<12}: {cnt:>8,} flows ({cnt/len(df)*100:.2f}%)")

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
    """
    Agrège les NetFlows d'une fenêtre temporelle par IP source.
    Retourne un DataFrame avec une ligne par IP source.
    """
    if len(window_df) == 0:
        return pd.DataFrame()

    rows = []
    for src, grp in window_df.groupby("SrcAddr"):
        if not src or pd.isna(src):
            continue

        # Label de l'IP : Botnet si au moins 1 flux botnet
        lbl = (LABEL_BOTNET     if (grp["LabelClean"]==LABEL_BOTNET).any() else
               LABEL_NORMAL     if (grp["LabelClean"]==LABEL_NORMAL).any() else
               LABEL_BACKGROUND)

        rows.append({
            "SrcAddr":    src,
            "n_flows":    len(grp),
            "H_dst_ip":   shannon_entropy(grp["DstAddr"]),
            "H_dst_port": shannon_entropy(grp["Dport"]),
            "H_src_port": shannon_entropy(grp["Sport"]),
            "mean_bytes": grp["TotBytes"].mean(),
            "mean_pkts":  grp["TotPkts"].mean(),
            "label":      lbl,
        })

    return pd.DataFrame(rows) if rows else pd.DataFrame()


# =============================================================================
# KGB adaptatif avec sliding window
# =============================================================================

class AdaptiveKGB:
    """
    KGB avec mise à jour incrémentale du modèle PCA.
    Le modèle est entraîné sur les N premières fenêtres (warm-up),
    puis mis à jour périodiquement sur le Background observé.
    """

    def __init__(self, fog=False, variance_threshold=0.90,
                 update_every=5):
        self.fog = fog
        self.variance_threshold = variance_threshold
        self.update_every = update_every   # mise à jour du modèle toutes les N fenêtres
        self.pca    = None
        self.scaler = None
        self.n_pc   = None
        self.threshold = 0.15
        self._bg_buffer = []               # buffer de données Background pour MAJ
        self._window_count = 0

    def _fit_model(self, X_bg: np.ndarray) -> None:
        """Entraîne/met à jour le modèle PCA sur des données Background."""
        if len(X_bg) < 3:
            return
        self.scaler = StandardScaler()
        Xs = self.scaler.fit_transform(X_bg)
        n  = Xs.shape[1]
        pca = PCA(n_components=n)
        pca.fit(Xs)
        cumvar = np.cumsum(pca.explained_variance_ratio_)
        k = int(np.searchsorted(cumvar, self.variance_threshold) + 1)
        self.n_pc = max(1, min(k, n - 1))
        self.pca  = pca

    def warmup(self, warmup_windows: list) -> None:
        """Entraîne le modèle sur les fenêtres initiales (Background uniquement)."""
        X_bg_all = []
        for wdf in warmup_windows:
            if len(wdf) == 0:
                continue
            bg = wdf[wdf["label"] == LABEL_BACKGROUND]
            if len(bg) > 0:
                X_bg_all.append(bg[KGB_FEATURES].fillna(0).values)
        if X_bg_all:
            self._fit_model(np.vstack(X_bg_all))
            print(f"      Warm-up : modèle entraîné sur "
                  f"{sum(len(x) for x in X_bg_all)} IPs Background")

    def score_window(self, window_agg: pd.DataFrame) -> np.ndarray:
        """Calcule le score d'anomalie pour les IPs d'une fenêtre."""
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
        max_s = s.max()
        return s / (max_s + 1e-10) if max_s > 0 else s

    def update(self, window_agg: pd.DataFrame) -> None:
        """Met à jour le modèle avec les nouvelles données Background."""
        self._window_count += 1
        bg = window_agg[window_agg["label"] == LABEL_BACKGROUND]
        if len(bg) > 0:
            self._bg_buffer.append(bg[KGB_FEATURES].fillna(0).values)

        # Mise à jour périodique
        if self._window_count % self.update_every == 0 and self._bg_buffer:
            X_bg = np.vstack(self._bg_buffer[-self.update_every*5:])
            self._fit_model(X_bg)
            # Purge du buffer (garde uniquement les N dernières fenêtres)
            self._bg_buffer = self._bg_buffer[-self.update_every*5:]


# =============================================================================
# Pipeline Sliding Window
# =============================================================================

def run_sliding_window(df: pd.DataFrame, window_min: int, warmup_n: int,
                       fog: bool = False) -> dict:
    """
    Applique KGB avec fenêtres glissantes.

    Retourne un dict avec les métriques par fenêtre temporelle.
    """
    name = "KGBfog" if fog else "KGBf"
    print(f"\n   Running {name} (window={window_min}min, warmup={warmup_n} windows)...")

    # Tri temporel
    df = df.sort_values("StartTime").reset_index(drop=True)
    t_start = df["StartTime"].min()
    t_end   = df["StartTime"].max()

    # Découpage en fenêtres
    window_td = pd.Timedelta(minutes=window_min)
    windows   = []
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
    warmup_data = [w[1] for w in windows[:warmup_n]]
    kgb.warmup(warmup_data)

    # Running metrics
    results = []
    for i, (t_win, wdf) in enumerate(windows):
        if len(wdf) == 0 or kgb.pca is None:
            continue

        y_true = (wdf["label"] == LABEL_BOTNET).astype(int).values
        scores = kgb.score_window(wdf)
        preds  = (scores >= kgb.threshold).astype(int)

        n_botnet = y_true.sum()
        n_total  = len(y_true)

        if n_total > 0:
            tn, fp, fn, tp = confusion_matrix(
                y_true, preds, labels=[0,1]
            ).ravel() if len(np.unique(y_true)) > 1 else (
                (preds==0).sum(), 0, 0, 0
            )
            eps  = 1e-10
            tpr  = tp / (tp + fn + eps)
            fpr  = fp / (fp + tn + eps)
            prec = tp / (tp + fp + eps)
            f1   = 2*prec*tpr / (prec + tpr + eps)
        else:
            tpr = fpr = f1 = 0.0
            tp = fp = fn = tn = 0

        results.append({
            "window_start": t_win,
            "n_flows_window": n_total,
            "n_botnet_ips":   int(n_botnet),
            "TP": int(tp), "FP": int(fp), "FN": int(fn), "TN": int(tn),
            "TPR": tpr, "FPR": fpr, "F1": f1,
        })

        # Mise à jour du modèle
        kgb.update(wdf)

    results_df = pd.DataFrame(results)
    results_df.to_csv(OUTPUT_DIR / f"sliding_{name.lower()}.csv", index=False)
    print(f"      {len(results_df)} fenêtres évaluées")
    if len(results_df) > 0:
        mean_tpr = results_df[results_df["n_botnet_ips"]>0]["TPR"].mean()
        mean_fpr = results_df["FPR"].mean()
        print(f"      TPR moyen (fenêtres avec botnet) : {mean_tpr:.3f}")
        print(f"      FPR moyen : {mean_fpr:.3f}")

    return results_df


def plot_running_metrics(results_kgbf: pd.DataFrame,
                          results_kgbfog: pd.DataFrame,
                          window_min: int) -> None:
    """
    Reproduit les courbes 'running metrics' de la Fig. 12 du papier.
    TPR et FPR évoluant dans le temps.
    """
    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)

    colors = {"KGBf":"#d62728", "KGBfog":"#ff7f0e"}

    for results, name in [(results_kgbf,"KGBf"), (results_kgbfog,"KGBfog")]:
        if len(results) == 0:
            continue
        t = results["window_start"]
        c = colors[name]

        # TPR
        axes[0].plot(t, results["TPR"], color=c, lw=1.5,
                     alpha=0.8, label=name)
        # FPR
        axes[1].plot(t, results["FPR"], color=c, lw=1.5, alpha=0.8)
        # Nombre d'IPs Botnet détectées (TP cumulatif)
        tp_cum = results["TP"].cumsum()
        axes[2].plot(t, tp_cum, color=c, lw=2, label=name)

    # Formatage axes
    axes[0].set_ylabel("TPR (True Positive Rate)")
    axes[0].set_title(f"Running TPR — Fenêtres de {window_min} min\n"
                       "(style Fig. 12 du papier García et al. 2014)")
    axes[0].set_ylim(-0.05, 1.05)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].axhline(0, color="gray", ls="--", lw=0.5)

    axes[1].set_ylabel("FPR (False Positive Rate)")
    axes[1].set_title(f"Running FPR — Fenêtres de {window_min} min")
    axes[1].set_ylim(-0.005, 0.1)
    axes[1].grid(True, alpha=0.3)
    axes[1].axhline(0, color="gray", ls="--", lw=0.5)

    axes[2].set_ylabel("TP cumulatifs")
    axes[2].set_title("IPs Botnet détectées (cumulatif)")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    # Format dates sur l'axe X
    for ax in axes:
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))
    axes[2].set_xlabel("Heure (2011-08-16)")
    plt.xticks(rotation=45)

    plt.tight_layout()
    out = OUTPUT_DIR / "11_running_metrics.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"      📊 11_running_metrics.png")


def plot_window_heatmap(results_kgbf: pd.DataFrame) -> None:
    """Heatmap TPR/FPR par fenêtre temporelle."""
    if len(results_kgbf) == 0:
        return

    # Filtre les fenêtres avec des IPs Botnet
    bot_windows = results_kgbf[results_kgbf["n_botnet_ips"] > 0].copy()
    if len(bot_windows) == 0:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Distribution du TPR par fenêtre
    axes[0].hist(bot_windows["TPR"], bins=20, color="#d62728", alpha=0.7,
                 edgecolor="black", lw=0.5)
    axes[0].set_xlabel("TPR par fenêtre")
    axes[0].set_ylabel("Nombre de fenêtres")
    axes[0].set_title("Distribution du TPR\n(fenêtres contenant des IPs Botnet)")
    axes[0].axvline(bot_windows["TPR"].mean(), color="black", ls="--",
                    label=f"Moyenne = {bot_windows['TPR'].mean():.3f}")
    axes[0].legend()

    # Scatter TPR vs FPR par fenêtre
    scatter = axes[1].scatter(
        results_kgbf["FPR"], results_kgbf["TPR"],
        c=results_kgbf["n_botnet_ips"],
        cmap="Reds", alpha=0.6, s=30
    )
    plt.colorbar(scatter, ax=axes[1], label="Nb IPs Botnet dans la fenêtre")
    axes[1].set_xlabel("FPR")
    axes[1].set_ylabel("TPR")
    axes[1].set_title("TPR vs FPR par fenêtre temporelle\n(couleur = nb IPs Botnet)")
    axes[1].grid(True, alpha=0.3)

    plt.suptitle("Analyse par fenêtre temporelle — KGBf", fontsize=13)
    plt.tight_layout()
    out = OUTPUT_DIR / "12_window_analysis.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"      📊 12_window_analysis.png")


def write_sliding_report(res_f: pd.DataFrame, res_fog: pd.DataFrame,
                          window_min: int, warmup_n: int) -> None:
    rp = OUTPUT_DIR / "sliding_report.txt"
    with open(rp, "w") as f:
        f.write("="*65 + "\n")
        f.write("RAPPORT — KGB Sliding Window CTU-13 Scénario 8\n")
        f.write("="*65 + "\n\n")
        f.write(f"Paramètres :\n")
        f.write(f"  Largeur de fenêtre : {window_min} minutes\n")
        f.write(f"  Fenêtres warm-up   : {warmup_n}\n\n")

        for res, name in [(res_f,"KGBf"), (res_fog,"KGBfog")]:
            if len(res) == 0:
                continue
            bot = res[res["n_botnet_ips"] > 0]
            f.write(f"Résultats {name} :\n")
            f.write(f"  Fenêtres totales          : {len(res)}\n")
            f.write(f"  Fenêtres avec Botnet      : {len(bot)}\n")
            if len(bot):
                f.write(f"  TPR moyen (bot windows)   : {bot['TPR'].mean():.3f}\n")
                f.write(f"  TPR max                   : {bot['TPR'].max():.3f}\n")
                f.write(f"  TPR min                   : {bot['TPR'].min():.3f}\n")
            f.write(f"  FPR moyen (toutes windows): {res['FPR'].mean():.4f}\n")
            f.write(f"  TP total                  : {res['TP'].sum()}\n")
            f.write(f"  FN total                  : {res['FN'].sum()}\n")
            f.write(f"  FP total                  : {res['FP'].sum()}\n\n")

        f.write("Interprétation :\n")
        f.write(
            "  L'approche par fenêtre temporelle permet de capturer\n"
            "  l'évolution du comportement du botnet Murlo au fil du temps.\n"
            "  Contrairement à l'agrégation globale, elle détecte les\n"
            "  phases d'activité intense (scans DCERPC) distinctement\n"
            "  des phases de communication C&C discrètes.\n"
        )
    print(f"      📄 sliding_report.txt")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file",      "-f", required=True)
    parser.add_argument("--window",    "-w", type=int, default=5,
                        help="Largeur de fenêtre en minutes (défaut: 5)")
    parser.add_argument("--warmup",    "-u", type=int, default=5,
                        help="Nombre de fenêtres warm-up (défaut: 5)")
    parser.add_argument("--max-rows",  "-n", type=int, default=None,
                        help="Limiter le nombre de lignes (test rapide)")
    args = parser.parse_args()

    print("\n" + "="*60)
    print("  KGB Sliding Window — CTU-13 Scénario 8")
    print("="*60)

    df = load_and_parse(args.file, max_rows=args.max_rows)

    print(f"\n[2/4] Sliding Window (W={args.window}min, warmup={args.warmup})...")
    res_f   = run_sliding_window(df, args.window, args.warmup, fog=False)
    res_fog = run_sliding_window(df, args.window, args.warmup, fog=True)

    print("\n[3/4] Graphiques running metrics...")
    plot_running_metrics(res_f, res_fog, args.window)
    plot_window_heatmap(res_f)

    print("\n[4/4] Rapport...")
    write_sliding_report(res_f, res_fog, args.window, args.warmup)

    print("\n" + "="*60)
    print(f"  ✅ Terminé ! Résultats dans : {OUTPUT_DIR}/")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()