# KGB Botnet Detector — CTU-13 Scenario 8

Implémentation du détecteur KGB (Pevný et al., 2012) sur le dataset
CTU-13 Scénario 8 (Botnet Murlo). Projet SUPSYS — IMT Atlantique 2026.

## Reproduire les résultats

### 1. Installer les dépendances
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Télécharger le dataset
URL officielle :
https://mcfp.felk.cvut.cz/publicDatasets/CTU-Malware-Capture-Botnet-49/

Fichier requis : `capture20110816-3.pcap.netflow.labeled` (~1.5 GB)
⚠️ L'archive contient un malware — ne pas ouvrir les fichiers .exe

Placer le fichier dans `data/` puis vérifier `config.py`.

### 3. Lancer les scripts dans l'ordre
```bash
python scripts/01_eda_ctu13_s8.py        # ~5 min sur 500K flows
python scripts/02_kgb_pipeline.py
python scripts/03_analysis_report.py
python scripts/04_full_dataset_kgb.py    # ~30 min, 12M flows
python scripts/05_sliding_window_kgb.py  # ~45 min, 12M flows
```

## Résultats obtenus
| Méthode | F1 | AUC | AP |
|---|---|---|---|
| KGBf (dataset complet) | 0.132 | 0.932 | 0.084 |
| KGBfog | 0.109 | 0.812 | 0.023 |
| RandomForest supervisé | 0.449 | 0.946 | 0.479 |
| Sliding Window KGBf | TPR=0.549 | — | — |

## Référence
García et al., "An empirical comparison of botnet detection methods",
Computers & Security 45 (2014) 100-123.
