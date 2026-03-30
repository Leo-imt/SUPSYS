"""
=============================================================================
CTU-13 Scenario 8 (Botnet-49 / Murlo) — Analyse Exploratoire des Données
=============================================================================
Format réel du fichier (découvert par cat -A) :

  EN-TÊTE : colonnes en texte libre séparées par espaces + 1 tabulation
            → on ignore l'en-tête et on impose les noms de colonnes

  DONNÉES : 12 champs séparés par des TABULATIONS (\t)
    [0]  StartTime   ex: 2011-08-16 14:18:55.889  (date + espace + heure)
    [1]  Dur         ex: 4.748
    [2]  Proto       ex: TCP
    [3]  SrcAddrPort ex: 88.176.79.163:49375
    [4]  Dir         ex: ->
    [5]  DstAddrPort ex: 147.32.84.172:46696
    [6]  Flags       ex: A_
    [7]  Tos         ex: 0
    [8]  TotPkts     ex: 88
    [9]  TotBytes    ex: 5916
    [10] Flows       ex: 1
    [11] Label       ex: Background  /  flow=From-Botnet-...  /  LEGITIMATE

Usage :
  python 01_eda_ctu13_s8.py --file capture20110816-3.pcap.netflow.labeled
  python 01_eda_ctu13_s8.py --file ... --max-rows 500000
=============================================================================
"""

import argparse
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
OUTPUT_DIR = Path("eda_output")
OUTPUT_DIR.mkdir(exist_ok=True)

# Noms de colonnes imposés (l'en-tête du fichier est inutilisable tel quel)
COLUMN_NAMES = [
    "StartTime", "Dur", "Proto",
    "SrcAddrPort", "Dir", "DstAddrPort",
    "Flags", "Tos", "TotPkts", "TotBytes", "Flows", "Label"
]

LABEL_BOTNET     = "Botnet"
LABEL_NORMAL     = "Normal"
LABEL_BACKGROUND = "Background"


# ---------------------------------------------------------------------------
# Chargement
# ---------------------------------------------------------------------------

def load_binetflow(path: str, max_rows: int = None) -> pd.DataFrame:
    """
    Charge le fichier binetflow CTU-13.
    - Séparateur : tabulation
    - Skip ligne 0 (en-tête mal formaté)
    - Colonnes imposées manuellement
    """
    print(f"\n[1/6] Chargement : {path}")

    df = pd.read_csv(
        path,
        sep="\t",
        skiprows=1,            # ignore l'en-tête d'origine
        header=None,
        names=COLUMN_NAMES,
        nrows=max_rows,
        engine="c",            # moteur C : rapide, pas de problème low_memory avec header=None
        dtype=str,             # tout en string d'abord, on convertira ensuite
        on_bad_lines="skip",
    )

    # Conversion numériques
    for col in ["Dur", "TotPkts", "TotBytes", "Flows", "Tos"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    print(f"      Lignes chargées : {len(df):,}")
    print(f"      Exemple ligne 0 :")
    print(f"        StartTime   : {df['StartTime'].iloc[0]}")
    print(f"        SrcAddrPort : {df['SrcAddrPort'].iloc[0]}")
    print(f"        DstAddrPort : {df['DstAddrPort'].iloc[0]}")
    print(f"        Label       : {df['Label'].iloc[0]}")

    return df


def split_addr_port(df: pd.DataFrame) -> pd.DataFrame:
    """
    Sépare SrcAddrPort → SrcAddr + Sport
            DstAddrPort → DstAddr + Dport
    Format : '88.176.79.163:49375'
    """
    for raw_col, addr_col, port_col in [
        ("SrcAddrPort", "SrcAddr", "Sport"),
        ("DstAddrPort", "DstAddr", "Dport"),
    ]:
        if raw_col not in df.columns:
            continue
        # rsplit sur ':' pour gérer les IPv6 (::1:80)
        split = df[raw_col].str.rsplit(":", n=1, expand=True)
        df[addr_col] = split[0].str.strip()
        df[port_col] = split[1].str.strip() if 1 in split.columns else np.nan

    return df


# ---------------------------------------------------------------------------
# Normalisation des labels
# ---------------------------------------------------------------------------

def normalize_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Labels observés dans le fichier CTU-13 Scénario 8 :
      Background                          → Background
      flow=From-Botnet-V43-UDP-...        → Botnet
      flow=To-Botnet-...                  → Botnet
      LEGITIMATE                          → Normal
      flow=From-Normal-...                → Normal
      flow=To-Normal-...                  → Normal
    """
    print("\n[2/6] Normalisation des étiquettes...")

    raw = df["Label"].astype(str).str.strip()

    # Aperçu des labels bruts
    unique_labels = raw.unique()
    print(f"      Labels bruts uniques ({len(unique_labels)}) :")
    for l in unique_labels[:15]:
        print(f"        '{l}'")

    def classify(label: str) -> str:
        l = label.lower()
        if "botnet" in l:
            return LABEL_BOTNET
        if "normal" in l or "legitimate" in l:
            return LABEL_NORMAL
        if "background" in l:
            return LABEL_BACKGROUND
        return "Unknown"

    df["LabelClean"] = raw.apply(classify)
    return df


# ---------------------------------------------------------------------------
# Étape 3 : Distribution
# ---------------------------------------------------------------------------

def print_label_distribution(df: pd.DataFrame) -> None:
    print("\n[3/6] Distribution des étiquettes")
    counts = df["LabelClean"].value_counts()
    total  = len(df)

    print(f"      {'Label':<15} {'Count':>12} {'%':>8}")
    print(f"      {'-'*38}")
    for label, count in counts.items():
        print(f"      {label:<15} {count:>12,} {count/total*100:>7.2f}%")

    print()
    print("      Référence papier (Scénario 8 complet, ~12M flows) :")
    print("      Background : 95.47% | Botnet : 0.10% | Normal : 4.42%")

    # Graphiques
    colors = {LABEL_BOTNET:"#d62728", LABEL_NORMAL:"#2ca02c",
              LABEL_BACKGROUND:"#1f77b4", "Unknown":"gray"}
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    bar_c = [colors.get(l, "gray") for l in counts.index]
    axes[0].bar(counts.index, counts.values, color=bar_c, edgecolor="black", lw=0.5)
    axes[0].set_title("Distribution des labels")
    axes[0].set_ylabel("NetFlows")
    axes[0].yaxis.set_major_formatter(mtick.FuncFormatter(lambda x,_: f"{x:,.0f}"))
    axes[0].tick_params(axis="x", rotation=15)

    no_bg = counts[~counts.index.isin([LABEL_BACKGROUND, "Unknown"])]
    if len(no_bg):
        axes[1].pie(no_bg.values, labels=no_bg.index,
                    autopct="%1.2f%%",
                    colors=[colors.get(l,"gray") for l in no_bg.index],
                    startangle=90)
        axes[1].set_title("Botnet vs Normal (sans Background)")

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR/"01_label_distribution.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"      📊 01_label_distribution.png")


# ---------------------------------------------------------------------------
# Étape 4 : Agrégation par IP
# ---------------------------------------------------------------------------

def analyze_per_ip(df: pd.DataFrame) -> pd.DataFrame:
    print("\n[4/6] Agrégation par IP source...")

    agg = df.groupby("SrcAddr").agg(
        n_flows      = ("Label",      "count"),
        botnet_flows = ("LabelClean", lambda x: (x == LABEL_BOTNET).sum()),
        normal_flows = ("LabelClean", lambda x: (x == LABEL_NORMAL).sum()),
        bg_flows     = ("LabelClean", lambda x: (x == LABEL_BACKGROUND).sum()),
        n_dst_ips    = ("DstAddr",    "nunique"),
        n_dst_ports  = ("Dport",      "nunique"),
        n_src_ports  = ("Sport",      "nunique"),
        n_protocols  = ("Proto",      "nunique"),
        total_bytes  = ("TotBytes",   "sum"),
        total_pkts   = ("TotPkts",    "sum"),
        mean_dur     = ("Dur",        "mean"),
        std_dur      = ("Dur",        "std"),
    ).reset_index()

    # Label majoritaire par IP (botnet prioritaire)
    agg["label"] = LABEL_BACKGROUND
    agg.loc[agg["normal_flows"] > agg["bg_flows"], "label"] = LABEL_NORMAL
    agg.loc[agg["botnet_flows"] > 0, "label"] = LABEL_BOTNET

    print(f"      IPs uniques    : {len(agg):,}")
    for lbl in [LABEL_BOTNET, LABEL_NORMAL, LABEL_BACKGROUND]:
        n = (agg["label"] == lbl).sum()
        print(f"      IPs {lbl:<12}: {n:,}")

    agg.to_csv(OUTPUT_DIR/"ip_aggregated.csv", index=False)
    print(f"      💾 ip_aggregated.csv")
    return agg


# ---------------------------------------------------------------------------
# Étape 5 : Entropies (features KGB)
# ---------------------------------------------------------------------------

def compute_entropy_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Features KGB (Lakhina Entropy, Pevný 2012) :
      H(dst_ip)   — diversité des IPs contactées
      H(dst_port) — diversité des ports destination
      H(src_port) — diversité des ports source
    + features de volume : mean/std bytes & packets
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

    feat_cols = ["H_dst_ip", "H_dst_port", "H_src_port"]
    print(f"\n      Entropies moyennes par classe :")
    print("      " + f"{'Classe':<12}" + "".join(f"{c:>13}" for c in feat_cols) + f"{'#IPs':>8}")
    print("      " + "-"*57)
    for lbl in [LABEL_BOTNET, LABEL_NORMAL, LABEL_BACKGROUND]:
        sub = edf[edf["label"] == lbl]
        if len(sub):
            vals = "".join(f"{sub[c].mean():>13.3f}" for c in feat_cols)
            print(f"      {lbl:<12}{vals}{len(sub):>8,}")

    edf.to_csv(OUTPUT_DIR/"entropy_features.csv", index=False)
    print(f"\n      💾 entropy_features.csv")

    # Histogrammes des entropies
    colors = {LABEL_BOTNET:"#d62728", LABEL_NORMAL:"#2ca02c", LABEL_BACKGROUND:"#aec7e8"}
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, feat in zip(axes, feat_cols):
        for lbl in [LABEL_BOTNET, LABEL_NORMAL, LABEL_BACKGROUND]:
            sub = edf[edf["label"] == lbl][feat].dropna()
            if len(sub):
                ax.hist(sub, bins=40, alpha=0.65, label=lbl,
                        color=colors[lbl], density=True)
        ax.set_title(f"{feat}")
        ax.set_xlabel("Entropie (bits)")
        ax.set_ylabel("Densité")
        ax.legend(fontsize=8)
    plt.suptitle("Features KGB — Entropies de Shannon par IP source (Scénario 8)", fontsize=12)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR/"02_entropy_distributions.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"      📊 02_entropy_distributions.png")

    return edf


# ---------------------------------------------------------------------------
# Étape 6 : Scatter dans l'espace KGB
# ---------------------------------------------------------------------------

def plot_kgb_space(edf: pd.DataFrame) -> None:
    print("\n[6/6] Visualisation dans l'espace KGB...")

    colors  = {LABEL_BOTNET:"#d62728", LABEL_NORMAL:"#2ca02c", LABEL_BACKGROUND:"#aec7e8"}
    markers = {LABEL_BOTNET:"*",       LABEL_NORMAL:"o",       LABEL_BACKGROUND:"."}
    sizes   = {LABEL_BOTNET:150,       LABEL_NORMAL:25,        LABEL_BACKGROUND:5}

    pairs = [("H_dst_ip","H_dst_port"), ("H_dst_ip","H_src_port"), ("H_dst_port","H_src_port")]
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for ax, (fx, fy) in zip(axes, pairs):
        for lbl in [LABEL_BACKGROUND, LABEL_NORMAL, LABEL_BOTNET]:
            sub     = edf[edf["label"] == lbl]
            n_total = len(sub)
            # Sous-échantillonnage Background pour lisibilité
            if lbl == LABEL_BACKGROUND and n_total > 2000:
                sub = sub.sample(2000, random_state=42)
            ax.scatter(sub[fx], sub[fy],
                       c=colors[lbl], marker=markers[lbl], s=sizes[lbl],
                       alpha=0.7, label=f"{lbl} (n={n_total:,})")
        ax.set_xlabel(fx)
        ax.set_ylabel(fy)
        ax.set_title(f"{fx} vs {fy}")
        ax.legend(fontsize=8, markerscale=1.5)

    plt.suptitle("Espace des features KGB — Scénario 8 CTU-13 (Murlo botnet)", fontsize=13)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR/"03_kgb_feature_space.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"      📊 03_kgb_feature_space.png")


# ---------------------------------------------------------------------------
# Rapport final
# ---------------------------------------------------------------------------

def generate_report(df: pd.DataFrame, edf: pd.DataFrame) -> None:
    rp = OUTPUT_DIR / "eda_report.txt"
    feat_cols = ["H_dst_ip", "H_dst_port", "H_src_port"]

    with open(rp, "w") as f:
        f.write("="*70 + "\n")
        f.write("RAPPORT EDA — CTU-13 Scénario 8 (Botnet-49 / Murlo)\n")
        f.write("Détecteur KGB — Pevný et al. (2012)\n")
        f.write("="*70 + "\n\n")

        f.write("FORMAT DU FICHIER\n" + "-"*40 + "\n")
        f.write("Séparateur : tabulation (\\t)\n")
        f.write("Colonnes   : StartTime | Dur | Proto | SrcAddrPort | Dir |\n")
        f.write("             DstAddrPort | Flags | Tos | TotPkts | TotBytes |\n")
        f.write("             Flows | Label\n\n")

        f.write("STATISTIQUES GÉNÉRALES\n" + "-"*40 + "\n")
        counts = df["LabelClean"].value_counts()
        f.write(f"Total NetFlows analysés : {len(df):,}\n")
        for lbl, cnt in counts.items():
            f.write(f"  {lbl:<18}: {cnt:>10,} ({cnt/len(df)*100:.2f}%)\n")

        f.write("\nIPS SOURCES UNIQUES\n" + "-"*40 + "\n")
        for lbl, cnt in edf["label"].value_counts().items():
            f.write(f"  {lbl:<18}: {cnt:>6,} IPs\n")

        f.write("\nFEATURES KGB — ENTROPIES MOYENNES\n" + "-"*40 + "\n")
        f.write(f"{'Classe':<12}" + "".join(f"{c:>13}" for c in feat_cols) + f"{'n_flows_moy':>13}\n")
        for lbl in [LABEL_BOTNET, LABEL_NORMAL, LABEL_BACKGROUND]:
            sub = edf[edf["label"] == lbl]
            if len(sub):
                vals = "".join(f"{sub[c].mean():>13.3f}" for c in feat_cols)
                f.write(f"{lbl:<12}{vals}{sub['n_flows'].mean():>13.1f}\n")

        f.write("\nDÉSÉQUILIBRE DES CLASSES\n" + "-"*40 + "\n")
        bc = (df["LabelClean"] == LABEL_BOTNET).sum()
        oc = len(df) - bc
        f.write(f"Botnet / Total  : {bc/len(df)*100:.3f}%\n")
        if bc > 0:
            f.write(f"Ratio imbalance : 1 Botnet pour {int(oc/bc)} autres\n")
        f.write("\nRecommandations pour le ML :\n")
        f.write("  - Agréger au niveau IP (comme KGB) réduit le déséquilibre\n")
        f.write("  - class_weight='balanced' pour SVM/RF\n")
        f.write("  - Métriques : F1-score, AUC-ROC (pas l'Accuracy seule)\n")
        f.write("  - Comparer KGBf (PC haute variance) vs KGBfog (PC basse variance)\n")

        f.write("\nFICHIERS GÉNÉRÉS\n" + "-"*40 + "\n")
        for p in sorted(OUTPUT_DIR.iterdir()):
            f.write(f"  {p.name}\n")

    print(f"\n      📄 eda_report.txt")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="EDA CTU-13 Scénario 8 — Détecteur KGB")
    parser.add_argument("--file", "-f", required=True,
                        help="Chemin vers capture20110816-3.pcap.netflow.labeled")
    parser.add_argument("--max-rows", "-n", type=int, default=None,
                        help="Nombre max de lignes (ex: 500000 pour test rapide)")
    args = parser.parse_args()

    if not os.path.isfile(args.file):
        print(f"❌ Fichier introuvable : {args.file}")
        sys.exit(1)

    print("\n" + "="*60)
    print("  CTU-13 Scénario 8 — EDA pour le détecteur KGB")
    print("="*60)

    df  = load_binetflow(args.file, max_rows=args.max_rows)
    df  = split_addr_port(df)
    df  = normalize_labels(df)
    print_label_distribution(df)
    _   = analyze_per_ip(df)
    edf = compute_entropy_features(df)
    plot_kgb_space(edf)
    generate_report(df, edf)

    print("\n" + "="*60)
    print(f"  ✅ EDA terminée ! Résultats dans : {OUTPUT_DIR}/")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()