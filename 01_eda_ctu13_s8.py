"""
=============================================================================
01_eda_ctu13_s8.py — Analyse Exploratoire des Données
CTU-13 Scénario 8 (Botnet-49 / Murlo)
=============================================================================
Format du fichier binetflow CTU-13 (découvert par cat -A) :
  - En-tête  : mélange espaces + tabulation → ignoré (skiprows=1)
  - Données  : 12 champs séparés par tabulations (\t)
  - Colonne SrcAddrPort : format 'IP:port' → splitté en SrcAddr + Sport

Usage :
  python 01_eda_ctu13_s8.py                        # utilise config.py
  python 01_eda_ctu13_s8.py --file /chemin/fichier # chemin explicite
  python 01_eda_ctu13_s8.py --max-rows 500000      # test rapide

Sorties (dans eda_output/) :
  entropy_features.csv, ip_aggregated.csv
  01_label_distribution.png, 02_entropy_distributions.png,
  03_kgb_feature_space.png, eda_report.txt
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
import matplotlib.ticker as mtick

# ── Configuration centrale du projet
from config import (
    COLUMN_NAMES, LABEL_BOTNET, LABEL_NORMAL, LABEL_BACKGROUND,
    KGB_FEATURES, EDA_OUTPUT, get_data_path
)

warnings.filterwarnings("ignore")
EDA_OUTPUT.mkdir(exist_ok=True)


# =============================================================================
# 1. Chargement et parsing
# =============================================================================

def load_binetflow(path: Path, max_rows: int = None) -> pd.DataFrame:
    """
    Charge le fichier binetflow CTU-13.
    Stratégie : skiprows=1 (ignore l'en-tête non standard) + noms imposés.
    Séparateur : tabulation (\t) — confirmé par inspection cat -A.
    """
    print(f"\n[1/6] Chargement : {path}")

    df = pd.read_csv(
        path,
        sep="\t",
        skiprows=1,
        header=None,
        names=COLUMN_NAMES,
        nrows=max_rows,
        engine="c",
        dtype=str,
        on_bad_lines="skip",
    )

    for col in ["Dur", "TotPkts", "TotBytes", "Flows", "Tos"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    print(f"      Lignes chargées : {len(df):,}")
    print(f"      Exemple ligne 0 :")
    print(f"        StartTime   : {df['StartTime'].iloc[0]}")
    print(f"        SrcAddrPort : {df['SrcAddrPort'].iloc[0]}")
    print(f"        Label       : {df['Label'].iloc[0]}")
    return df


def split_addr_port(df: pd.DataFrame) -> pd.DataFrame:
    """Sépare 'IP:port' → SrcAddr + Sport  /  DstAddr + Dport."""
    for raw, addr, port in [
        ("SrcAddrPort", "SrcAddr", "Sport"),
        ("DstAddrPort", "DstAddr", "Dport"),
    ]:
        if raw not in df.columns:
            continue
        split = df[raw].str.rsplit(":", n=1, expand=True)
        df[addr] = split[0].str.strip()
        df[port] = split[1].str.strip() if 1 in split.columns else np.nan
    return df


# =============================================================================
# 2. Normalisation des labels
# =============================================================================

def normalize_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Labels bruts observés dans CTU-13 Scénario 8 :
      'Background'               → Background
      'Botnet'                   → Botnet
      'Botnet FILTER_LEGITIMATE' → Botnet
      'LEGITIMATE'               → Normal
      '1'                        → Unknown (ignoré)
    """
    print("\n[2/6] Normalisation des étiquettes...")
    raw = df["Label"].astype(str).str.strip()

    unique_labels = raw.unique()
    print(f"      Labels bruts uniques ({len(unique_labels)}) :")
    for l in unique_labels[:15]:
        print(f"        '{l}'")

    def classify(label: str) -> str:
        l = label.lower()
        if "botnet"     in l: return LABEL_BOTNET
        if "normal"     in l or "legitimate" in l: return LABEL_NORMAL
        if "background" in l: return LABEL_BACKGROUND
        return "Unknown"

    df["LabelClean"] = raw.apply(classify)
    return df


# =============================================================================
# 3. Distribution des labels
# =============================================================================

def print_label_distribution(df: pd.DataFrame) -> None:
    print("\n[3/6] Distribution des étiquettes")
    counts = df["LabelClean"].value_counts()
    total  = len(df)

    print(f"      {'Label':<15} {'Count':>12} {'%':>8}")
    print(f"      {'-'*38}")
    for label, count in counts.items():
        print(f"      {label:<15} {count:>12,} {count/total*100:>7.2f}%")
    print("\n      Référence papier (Scénario 8 complet, ~12M flows) :")
    print("      Background : 95.47% | Botnet : 0.10% | Normal : 4.42%")

    colors = {LABEL_BOTNET:"#d62728", LABEL_NORMAL:"#2ca02c",
              LABEL_BACKGROUND:"#1f77b4", "Unknown":"gray"}
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    bar_c = [colors.get(l, "gray") for l in counts.index]
    axes[0].bar(counts.index, counts.values, color=bar_c,
                edgecolor="black", lw=0.5)
    axes[0].set_title("Distribution des labels")
    axes[0].set_ylabel("NetFlows")
    axes[0].yaxis.set_major_formatter(mtick.FuncFormatter(
        lambda x, _: f"{x:,.0f}"))
    axes[0].tick_params(axis="x", rotation=15)

    no_bg = counts[~counts.index.isin([LABEL_BACKGROUND, "Unknown"])]
    if len(no_bg):
        axes[1].pie(no_bg.values, labels=no_bg.index,
                    autopct="%1.2f%%",
                    colors=[colors.get(l, "gray") for l in no_bg.index],
                    startangle=90)
        axes[1].set_title("Botnet vs Normal (sans Background)")

    plt.tight_layout()
    plt.savefig(EDA_OUTPUT / "01_label_distribution.png",
                dpi=150, bbox_inches="tight")
    plt.close()
    print(f"      📊 01_label_distribution.png")


# =============================================================================
# 4. Agrégation par IP source
# =============================================================================

def analyze_per_ip(df: pd.DataFrame) -> pd.DataFrame:
    """Agrège les NetFlows par IP source (requis pour KGB)."""
    print("\n[4/6] Agrégation par IP source...")

    agg = df.groupby("SrcAddr").agg(
        n_flows      = ("Label",      "count"),
        botnet_flows = ("LabelClean", lambda x: (x == LABEL_BOTNET).sum()),
        normal_flows = ("LabelClean", lambda x: (x == LABEL_NORMAL).sum()),
        bg_flows     = ("LabelClean", lambda x: (x == LABEL_BACKGROUND).sum()),
        n_dst_ips    = ("DstAddr",    "nunique"),
        n_dst_ports  = ("Dport",      "nunique"),
        n_src_ports  = ("Sport",      "nunique"),
        total_bytes  = ("TotBytes",   "sum"),
        total_pkts   = ("TotPkts",    "sum"),
        mean_dur     = ("Dur",        "mean"),
    ).reset_index()

    agg["label"] = LABEL_BACKGROUND
    agg.loc[agg["normal_flows"] > agg["bg_flows"], "label"] = LABEL_NORMAL
    agg.loc[agg["botnet_flows"] > 0, "label"] = LABEL_BOTNET

    print(f"      IPs uniques    : {len(agg):,}")
    for lbl in [LABEL_BOTNET, LABEL_NORMAL, LABEL_BACKGROUND]:
        print(f"      IPs {lbl:<12}: {(agg['label']==lbl).sum():,}")

    agg.to_csv(EDA_OUTPUT / "ip_aggregated.csv", index=False)
    print(f"      💾 ip_aggregated.csv")
    return agg


# =============================================================================
# 5. Calcul des entropies de Shannon (features KGB)
# =============================================================================

def compute_entropy_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcule pour chaque IP source les 3 entropies de Shannon :
      H(dst_ip)   — diversité des IPs destination
      H(dst_port) — diversité des ports destination
      H(src_port) — diversité des ports source
    Ces 3 features forment le vecteur d'entrée du détecteur KGB.
    """
    print("\n[5/6] Calcul des entropies par IP source (features KGB)...")

    def shannon_entropy(series: pd.Series) -> float:
        p = series.value_counts(normalize=True).values
        return float(-np.sum(p * np.log2(p + 1e-10)))

    rows = []
    grouped   = df.groupby("SrcAddr")
    total_ips = len(grouped)

    for i, (src_ip, g) in enumerate(grouped):
        if i % 3000 == 0:
            print(f"      {i:>6}/{total_ips} IPs...", end="\r")

        label = (LABEL_BOTNET     if (g["LabelClean"] == LABEL_BOTNET).any() else
                 LABEL_NORMAL     if (g["LabelClean"] == LABEL_NORMAL).any() else
                 LABEL_BACKGROUND)

        rows.append({
            "src_ip":     src_ip,
            "n_flows":    len(g),
            "H_dst_ip":   shannon_entropy(g["DstAddr"]),
            "H_dst_port": shannon_entropy(g["Dport"].astype(str)),
            "H_src_port": shannon_entropy(g["Sport"].astype(str)),
            "mean_bytes": g["TotBytes"].mean(),
            "std_bytes":  g["TotBytes"].std(),
            "mean_pkts":  g["TotPkts"].mean(),
            "std_pkts":   g["TotPkts"].std(),
            "mean_dur":   g["Dur"].mean(),
            "label":      label,
        })

    print()
    edf = pd.DataFrame(rows)

    print(f"\n      Entropies moyennes par classe :")
    print("      " + f"{'Classe':<12}" +
          "".join(f"{c:>13}" for c in KGB_FEATURES) + f"{'#IPs':>8}")
    print("      " + "-"*57)
    for lbl in [LABEL_BOTNET, LABEL_NORMAL, LABEL_BACKGROUND]:
        sub = edf[edf["label"] == lbl]
        if len(sub):
            vals = "".join(f"{sub[c].mean():>13.3f}" for c in KGB_FEATURES)
            print(f"      {lbl:<12}{vals}{len(sub):>8,}")

    edf.to_csv(EDA_OUTPUT / "entropy_features.csv", index=False)
    print(f"\n      💾 entropy_features.csv")

    # Histogrammes
    colors = {LABEL_BOTNET:"#d62728", LABEL_NORMAL:"#2ca02c",
              LABEL_BACKGROUND:"#aec7e8"}
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, feat in zip(axes, KGB_FEATURES):
        for lbl in [LABEL_BOTNET, LABEL_NORMAL, LABEL_BACKGROUND]:
            sub = edf[edf["label"] == lbl][feat].dropna()
            if len(sub):
                ax.hist(sub, bins=40, alpha=0.65, label=lbl,
                        color=colors[lbl], density=True)
        ax.set_title(feat)
        ax.set_xlabel("Entropie (bits)")
        ax.set_ylabel("Densité")
        ax.legend(fontsize=8)
    plt.suptitle("Features KGB — Entropies de Shannon par IP source "
                 "(Scénario 8)", fontsize=12)
    plt.tight_layout()
    plt.savefig(EDA_OUTPUT / "02_entropy_distributions.png",
                dpi=150, bbox_inches="tight")
    plt.close()
    print(f"      📊 02_entropy_distributions.png")
    return edf


# =============================================================================
# 6. Visualisation dans l'espace KGB
# =============================================================================

def plot_kgb_space(edf: pd.DataFrame) -> None:
    print("\n[6/6] Visualisation dans l'espace KGB...")

    colors  = {LABEL_BOTNET:"#d62728", LABEL_NORMAL:"#2ca02c",
               LABEL_BACKGROUND:"#aec7e8"}
    markers = {LABEL_BOTNET:"*",  LABEL_NORMAL:"o", LABEL_BACKGROUND:"."}
    sizes   = {LABEL_BOTNET:150,  LABEL_NORMAL:25,  LABEL_BACKGROUND:5}

    pairs = [("H_dst_ip","H_dst_port"), ("H_dst_ip","H_src_port"),
             ("H_dst_port","H_src_port")]
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for ax, (fx, fy) in zip(axes, pairs):
        for lbl in [LABEL_BACKGROUND, LABEL_NORMAL, LABEL_BOTNET]:
            sub = edf[edf["label"] == lbl]
            n   = len(sub)
            if lbl == LABEL_BACKGROUND and n > 2000:
                sub = sub.sample(2000, random_state=42)
            ax.scatter(sub[fx], sub[fy], c=colors[lbl], marker=markers[lbl],
                       s=sizes[lbl], alpha=0.7, label=f"{lbl} (n={n:,})")
        ax.set_xlabel(fx); ax.set_ylabel(fy)
        ax.set_title(f"{fx} vs {fy}")
        ax.legend(fontsize=8, markerscale=1.5)

    plt.suptitle("Espace features KGB — Scénario 8 CTU-13 (Murlo)",
                 fontsize=13)
    plt.tight_layout()
    plt.savefig(EDA_OUTPUT / "03_kgb_feature_space.png",
                dpi=150, bbox_inches="tight")
    plt.close()
    print(f"      📊 03_kgb_feature_space.png")


# =============================================================================
# Rapport
# =============================================================================

def generate_report(df: pd.DataFrame, edf: pd.DataFrame) -> None:
    rp = EDA_OUTPUT / "eda_results.txt"
    counts = df["LabelClean"].value_counts()
    bc = (df["LabelClean"] == LABEL_BOTNET).sum()
    oc = len(df) - bc
    with open(rp, "w") as f:
        f.write(f"{'Label':<18} {'Count':>12} {'%':>8}\n")
        f.write("-"*42 + "\n")
        for lbl, cnt in counts.items():
            f.write(f"{lbl:<18} {cnt:>12,} {cnt/len(df)*100:>7.2f}%\n")
        f.write(f"{'TOTAL':<18} {len(df):>12,}\n\n")
        f.write(f"{'Classe':<18} {'IPs':>8}\n")
        f.write("-"*28 + "\n")
        for lbl, cnt in edf["label"].value_counts().items():
            f.write(f"{lbl:<18} {cnt:>8,}\n")
        f.write("\n")
        f.write(f"{'Classe':<12}" + "".join(f"{c:>13}" for c in KGB_FEATURES) + f"{'n_flows_moy':>13}\n")
        f.write("-"*64 + "\n")
        for lbl in [LABEL_BOTNET, LABEL_NORMAL, LABEL_BACKGROUND]:
            sub = edf[edf["label"] == lbl]
            if len(sub):
                vals = "".join(f"{sub[c].mean():>13.3f}" for c in KGB_FEATURES)
                f.write(f"{lbl:<12}{vals}{sub['n_flows'].mean():>13.1f}\n")
        f.write("\n")
        f.write(f"Botnet/Total : {bc/len(df)*100:.3f}%")
        if bc:
            f.write(f"  (ratio 1:{int(oc/bc)})\n")
    print(f"\n      📄 eda_results.txt")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="EDA CTU-13 Scénario 8 — Détecteur KGB")
    parser.add_argument(
        "--file", "-f", default=None,
        help="Chemin vers capture20110816-3.pcap.netflow.labeled "
             "(défaut : DATA_FILE dans config.py)")
    parser.add_argument(
        "--max-rows", "-n", type=int, default=None,
        help="Nombre max de lignes (ex: 500000 pour test rapide)")
    args = parser.parse_args()

    data_path = get_data_path(args.file)

    print("\n" + "="*60)
    print("  CTU-13 Scénario 8 — EDA pour le détecteur KGB")
    print("="*60)

    df  = load_binetflow(data_path, max_rows=args.max_rows)
    df  = split_addr_port(df)
    df  = normalize_labels(df)
    print_label_distribution(df)
    _   = analyze_per_ip(df)
    edf = compute_entropy_features(df)
    plot_kgb_space(edf)
    generate_report(df, edf)

    print("\n" + "="*60)
    print(f"  ✅ EDA terminée ! Résultats dans : {EDA_OUTPUT}/")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
