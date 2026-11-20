"""Zeitlich sauberer Churn-Datensatz: Features nur vor dem Stichtag, Label aus den 90 Tagen danach.

Grundregel gegen Data Leakage: Alles, was das Modell als Eingabe sieht, muss am Stichtag bereits bekannt
gewesen sein. Deshalb filtert ``kundenfeatures`` hart auf ``datum < stichtag`` und das Label schaut
ausschließlich in das Fenster ``[stichtag, stichtag + 90 Tage)``.
"""

import numpy as np
import pandas as pd

from trace_kunden.konfig import AKTIV_FENSTER_TAGE, CHURN_FENSTER_TAGE

# Merkmale für die Modelle.
FEATURES = [
    "recency_tage",
    "frequenz",
    "umsatz_gesamt",
    "kundendauer_tage",
    "kaeufe_90",
    "kaeufe_vorher_90",
    "umsatz_90",
    "aktive_monate_12",
    "artikel_vielfalt",
    "storno_quote",
    "ist_uk",
]
# Abgeleitete Verhältnisse aus der ersten Iteration. Sie sind exakte Kombinationen anderer Merkmale
# (z. B. Warenkorb = Umsatz / Frequenz, nach Log-Transformation eine reine Differenz). In der Logistic
# Regression erzeugen sie Kollinearität und unplausible Vorzeichen, ohne die Güte zu verbessern.
ABGELEITET = ["warenkorb_mittel", "mittlerer_abstand_tage", "ueberfaellig"]


def population(verkaeufe: pd.DataFrame, stichtag: pd.Timestamp,
               aktiv_tage: int = AKTIV_FENSTER_TAGE) -> pd.Index:
    """Kunden mit mindestens einem Kauf in den ``aktiv_tage`` vor dem Stichtag.

    Wer seit über einem Jahr nicht gekauft hat, ist bereits weg; ihn mitzuzählen würde die Aufgabe
    künstlich leicht machen und die Kennzahlen schönen.
    """
    v = verkaeufe[(verkaeufe["datum"] < stichtag)
                  & (verkaeufe["datum"] >= stichtag - pd.Timedelta(days=aktiv_tage))]
    return pd.Index(np.sort(v["kunde"].unique()), name="kunde")


def kundenfeatures(verkaeufe: pd.DataFrame, stornos: pd.DataFrame, stichtag: pd.Timestamp) -> pd.DataFrame:
    """Baut die Merkmale je Kunde ausschließlich aus Daten vor dem Stichtag."""
    v = verkaeufe[verkaeufe["datum"] < stichtag]
    s = stornos[stornos["datum"] < stichtag]
    tage_vor = (stichtag - v["datum"]).dt.days

    rechnungen = (v.groupby(["kunde", "rechnung"])
                  .agg(datum=("datum", "min"), umsatz=("umsatz", "sum"))
                  .reset_index())
    r_tage = (stichtag - rechnungen["datum"]).dt.days
    g = rechnungen.groupby("kunde")

    f = pd.DataFrame({
        "recency_tage": (stichtag - g["datum"].max()).dt.days,
        "frequenz": g.size(),
        "umsatz_gesamt": g["umsatz"].sum(),
        "warenkorb_mittel": g["umsatz"].mean(),
        "kundendauer_tage": (stichtag - g["datum"].min()).dt.days,
    })
    # Mittlerer Abstand zwischen Käufen; Einmalkäufer bekommen das Aktiv-Fenster als neutralen Wert.
    f["mittlerer_abstand_tage"] = np.where(
        f["frequenz"] > 1, f["kundendauer_tage"] / (f["frequenz"] - 1).clip(lower=1), AKTIV_FENSTER_TAGE)
    # > 1 heißt: länger kein Kauf als im eigenen Rhythmus üblich.
    f["ueberfaellig"] = f["recency_tage"] / f["mittlerer_abstand_tage"].clip(lower=1)
    f["kaeufe_90"] = rechnungen[r_tage < 90].groupby("kunde").size()
    f["kaeufe_vorher_90"] = rechnungen[(r_tage >= 90) & (r_tage < 180)].groupby("kunde").size()
    f["umsatz_90"] = rechnungen[r_tage < 90].groupby("kunde")["umsatz"].sum()
    monate = v[tage_vor < 365].assign(monat=lambda d: d["datum"].dt.to_period("M"))
    f["aktive_monate_12"] = monate.groupby("kunde")["monat"].nunique()
    f["artikel_vielfalt"] = v.groupby("kunde")["artikel"].nunique()
    storno_zeilen = s.groupby("kunde").size()
    f["storno_quote"] = storno_zeilen.reindex(f.index, fill_value=0) / (
        v.groupby("kunde").size() + storno_zeilen.reindex(f.index, fill_value=0))
    land = v.groupby("kunde")["land"].agg(lambda x: x.mode().iat[0])
    f["ist_uk"] = (land == "United Kingdom").astype(int)

    zaehler = ["kaeufe_90", "kaeufe_vorher_90", "umsatz_90", "aktive_monate_12"]
    f[zaehler] = f[zaehler].fillna(0)
    return f[FEATURES + ABGELEITET].astype(float)


def churn_label(verkaeufe: pd.DataFrame, kunden: pd.Index, stichtag: pd.Timestamp,
                fenster_tage: int = CHURN_FENSTER_TAGE) -> pd.Series:
    """1 = kein Kauf im Fenster [Stichtag, Stichtag + fenster_tage), sonst 0."""
    ende = stichtag + pd.Timedelta(days=fenster_tage)
    kaeufer = verkaeufe.loc[(verkaeufe["datum"] >= stichtag) & (verkaeufe["datum"] < ende), "kunde"].unique()
    return pd.Series((~kunden.isin(kaeufer)).astype(int), index=kunden, name="churn")


def churn_datensatz(verkaeufe: pd.DataFrame, stornos: pd.DataFrame, stichtag: pd.Timestamp) -> pd.DataFrame:
    """Features + Label für die aktive Population zum Stichtag."""
    ende = stichtag + pd.Timedelta(days=CHURN_FENSTER_TAGE)
    if verkaeufe["datum"].max() < ende - pd.Timedelta(days=1):
        raise ValueError(f"Das Label-Fenster bis {ende:%d.%m.%Y} reicht über das Datenende hinaus.")
    kunden = population(verkaeufe, stichtag)
    x = kundenfeatures(verkaeufe, stornos, stichtag).reindex(kunden)
    y = churn_label(verkaeufe, kunden, stichtag)
    return x.join(y)
