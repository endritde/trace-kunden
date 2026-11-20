"""RFM-Analyse und Kundensegmentierung mit K-Means."""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from trace_kunden.konfig import SEED


def rfm_tabelle(verkaeufe: pd.DataFrame, referenz: pd.Timestamp) -> pd.DataFrame:
    """Recency (Tage seit letztem Kauf), Frequency (Anzahl Rechnungen), Monetary (Umsatz) je Kunde.

    Es zählen nur Käufe vor ``referenz``; Recency wird in ganzen Tagen bis zum Referenzdatum gemessen.
    """
    v = verkaeufe[verkaeufe["datum"] < referenz]
    g = v.groupby("kunde")
    rfm = pd.DataFrame({
        "recency": (referenz - g["datum"].max()).dt.days,
        "frequency": g["rechnung"].nunique(),
        "monetary": g["umsatz"].sum(),
    })
    return rfm


def rfm_scores(rfm: pd.DataFrame) -> pd.DataFrame:
    """Klassische RFM-Scores 1–5 (Quintile). Bei Recency ist 5 = kürzlich gekauft."""
    def quintil(s: pd.Series, umgekehrt: bool = False) -> pd.Series:
        r = s.rank(method="first", ascending=not umgekehrt)
        return pd.qcut(r, 5, labels=[1, 2, 3, 4, 5]).astype(int)

    return pd.DataFrame({
        "R": quintil(rfm["recency"], umgekehrt=True),
        "F": quintil(rfm["frequency"]),
        "M": quintil(rfm["monetary"]),
    }, index=rfm.index)


def rfm_matrix(rfm: pd.DataFrame) -> tuple[np.ndarray, StandardScaler]:
    """Log-transformiert und standardisiert R, F, M für K-Means.

    Alle drei Größen sind stark rechtsschief (wenige Großhändler mit sehr hohem Umsatz). Ohne Log würde
    K-Means fast nur nach diesen Ausreißern clustern; ohne Skalierung dominierte der Umsatz die Distanz.
    """
    x = np.log1p(rfm[["recency", "frequency", "monetary"]].clip(lower=0).to_numpy(dtype=float))
    skalierer = StandardScaler().fit(x)
    return skalierer.transform(x), skalierer


def k_vergleich(x: np.ndarray, ks: range = range(2, 11)) -> pd.DataFrame:
    """Inertia (für die Elbow-Methode) und Silhouette-Koeffizient je k."""
    zeilen = []
    for k in ks:
        km = KMeans(n_clusters=k, n_init=10, random_state=SEED).fit(x)
        zeilen.append({"k": k, "inertia": km.inertia_, "silhouette": silhouette_score(x, km.labels_)})
    return pd.DataFrame(zeilen).set_index("k")


def clustern(x: np.ndarray, k: int) -> KMeans:
    """K-Means mit festem Startwert, damit die Segmente reproduzierbar sind."""
    return KMeans(n_clusters=k, n_init=10, random_state=SEED).fit(x)


def segment_profil(rfm: pd.DataFrame, segment: pd.Series) -> pd.DataFrame:
    """Kennzahlen je Segment: Größe, Median von R/F/M und Anteil am Gesamtumsatz."""
    d = rfm.assign(segment=segment)
    g = d.groupby("segment")
    profil = pd.DataFrame({
        "kunden": g.size(),
        "anteil_kunden": g.size() / len(d),
        "recency_median": g["recency"].median(),
        "frequency_median": g["frequency"].median(),
        "monetary_median": g["monetary"].median(),
        "umsatz_anteil": g["monetary"].sum() / d["monetary"].sum(),
    })
    return profil


def segmente_benennen(rfm: pd.DataFrame, cluster: np.ndarray) -> pd.Series:
    """Gibt jedem Cluster einen fachlichen Namen anhand seiner Mediane.

    Die Regeln sind bewusst einfach und unabhängig von der Cluster-Nummer (die K-Means zufällig vergibt):
    - aktiv (Recency-Median ≤ 90 Tage) und viele Käufe → „Top-Kunden“
    - aktiv, aber wenige Käufe → „Neue & Gelegenheitskunden“
    - nicht mehr aktiv, aber früher viele Käufe → „Gefährdete Stammkunden“
    - lange inaktiv, wenige Käufe → „Verlorene Einmalkäufer“
    Zwei Cluster mit demselben Namen werden zusammengelegt und damit gemeinsam ausgewertet.
    """
    d = rfm.assign(cluster=cluster)
    med = d.groupby("cluster")[["recency", "frequency"]].median()
    f_median_gesamt = rfm["frequency"].median()
    namen = {}
    for c, zeile in med.iterrows():
        aktiv = zeile["recency"] <= 90
        viele = zeile["frequency"] > f_median_gesamt
        if aktiv and viele:
            namen[c] = "Top-Kunden"
        elif aktiv:
            namen[c] = "Neue & Gelegenheitskunden"
        elif viele:
            namen[c] = "Gefährdete Stammkunden"
        else:
            namen[c] = "Verlorene Einmalkäufer"
    return d["cluster"].map(namen).rename("segment")
