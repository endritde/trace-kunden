"""Zentrale Einstellungen: Pfade, Stichtage, Zufallsstartwert und Tracewell-Farben."""

from pathlib import Path

import pandas as pd

PROJEKT = Path(__file__).resolve().parents[2]
DATEN = PROJEKT / "data"
ROH_ZIP = DATEN / "online_retail_ii.zip"
ROH_PARQUET = DATEN / "online_retail_ii.parquet"
DOCS = PROJEKT / "docs"

QUELLE_URL = "https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip"

SEED = 42

# Churn: Ein Kunde gilt als abgewandert, wenn er in den 90 Tagen nach dem Stichtag nichts kauft.
CHURN_FENSTER_TAGE = 90
# Nur Kunden, die in den 365 Tagen vor dem Stichtag gekauft haben, gehören zur Prognose-Population.
AKTIV_FENSTER_TAGE = 365
# Training: Stichtag A, dessen 90-Tage-Fenster vor Stichtag B endet. Test: Stichtag B (Out-of-Time).
STICHTAG_TRAIN = pd.Timestamp("2011-06-10")
STICHTAG_TEST = pd.Timestamp("2011-09-09")
# Robustheitsprüfung: Training im selben Saisonfenster ein Jahr früher.
STICHTAG_TRAIN_VORJAHR = pd.Timestamp("2010-09-09")

# Tracewell-Farben (aus ~/tracewell-neu kopiert). Kategoriale Reihenfolge ist fest und mit dem
# Paletten-Validator geprüft (Helligkeit, Sättigung, Farbfehlsichtigkeit, Kontrast).
FARBEN = {
    "ziegel": "#a85a3c",
    "blau": "#4f68b8",
    "gold": "#b98400",
    "petrol": "#00927a",
    "violett": "#9b6fc4",
    "tinte": "#1b1a19",
    "gedaempft": "#6b6762",
    "linie": "#e4e2dd",
    "papier": "#f3f2ef",
    "ziegel_hell": "#f6ebe5",
}
KATEGORIAL = [FARBEN[k] for k in ("ziegel", "blau", "gold", "petrol", "violett")]
