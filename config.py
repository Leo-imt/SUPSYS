"""
Configuration du projet KGB CTU-13.
Modifier DATA_FILE pour pointer vers le fichier téléchargé localement.

Dataset source officielle :
  https://mcfp.felk.cvut.cz/publicDatasets/CTU-Malware-Capture-Botnet-49/
  Fichier : capture20110816-3.pcap.netflow.labeled
"""
from pathlib import Path

# ── À MODIFIER selon votre installation ──────────────────────
DATA_FILE = Path("data/capture20110816-3.pcap.netflow.labeled")
# ─────────────────────────────────────────────────────────────

# Dossiers de sortie (créés automatiquement)
EDA_OUTPUT     = Path("eda_output")
KGB_OUTPUT     = Path("kgb_output")
REPORT_OUTPUT  = Path("report_output")
FULL_OUTPUT    = Path("full_output")
SLIDING_OUTPUT = Path("sliding_output")

# Vérification au démarrage
if not DATA_FILE.exists():
    print(f"""
⚠️  Fichier de données introuvable : {DATA_FILE}

Téléchargez le dataset CTU-13 Scénario 8 :
  URL : https://mcfp.felk.cvut.cz/publicDatasets/CTU-Malware-Capture-Botnet-49/
  Fichier : capture20110816-3.pcap.netflow.labeled
  Placez-le dans : {DATA_FILE.parent.resolve()}/

Puis relancez le script.
""")
    raise FileNotFoundError(f"Dataset introuvable : {DATA_FILE}")
