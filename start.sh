#!/usr/bin/env bash
# Richtet die Umgebung ein, lädt bei Bedarf die Daten von UCI und öffnet das Notebook in JupyterLab.
#   ./start.sh            → JupyterLab mit notebooks/kundenanalyse.ipynb
#   ./start.sh --neu      → Notebook ohne Oberfläche komplett neu ausführen (Ergebnisse, Diagramme, eval/)
set -euo pipefail
cd "$(dirname "$0")"

command -v uv >/dev/null || { echo "uv fehlt: https://docs.astral.sh/uv/"; exit 1; }
uv sync --quiet

# Einmalig ca. 45 MB laden; das Einlesen der Excel-Datei braucht 2–3 Minuten und kurz ~600 MB RAM.
[ -f data/online_retail_ii.parquet ] || uv run python -m trace_kunden.daten

if [ "${1:-}" = "--neu" ]; then
  uv run jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=900 \
    notebooks/kundenanalyse.ipynb
  echo "Fertig: notebooks/kundenanalyse.ipynb, docs/*.png und eval/ergebnisse.json sind aktualisiert."
else
  uv run jupyter lab notebooks/kundenanalyse.ipynb
fi
