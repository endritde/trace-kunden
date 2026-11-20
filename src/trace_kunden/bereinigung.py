"""Datenbereinigung mit Protokoll: Jeder Schritt ist begründet und zählt die betroffenen Zeilen.

Ergebnis sind zwei Tabellen:
- ``verkaeufe``: echte Produktverkäufe an identifizierbare Kunden (Grundlage für RFM und Churn)
- ``stornos``: Stornozeilen identifizierbarer Kunden (werden nicht als Umsatz gezählt, liefern aber das
  Merkmal „Storno-Quote“)
"""

from dataclasses import dataclass, field

import pandas as pd

# Produktartikel haben eine fünfstellige Nummer, ggf. mit Buchstaben-Variante (z. B. 85123A).
# Alles andere sind Porto, Gebühren, manuelle Buchungen oder Testartikel (POST, DOT, M, BANK CHARGES …).
PRODUKT_MUSTER = r"^\d{5}[A-Za-z]{0,2}$"


@dataclass
class Protokoll:
    """Sammelt je Schritt: entfernte Zeilen, verbleibende Zeilen, Begründung."""

    schritte: list[dict] = field(default_factory=list)

    def eintragen(self, schritt: str, vorher: int, nachher: int, grund: str) -> None:
        self.schritte.append({"schritt": schritt, "entfernt": vorher - nachher, "verbleibend": nachher,
                              "begründung": grund})

    def tabelle(self) -> pd.DataFrame:
        return pd.DataFrame(self.schritte).set_index("schritt")


def ist_storno(rechnung: pd.Series) -> pd.Series:
    """Stornorechnungen beginnen mit „C“."""
    return rechnung.str.startswith("C")


def stornierte_kaeufe(verkaeufe: pd.DataFrame, stornos: pd.DataFrame) -> pd.Index:
    """Index der Verkaufszeilen, die später vollständig storniert wurden.

    Ein Storno passt zu einem Kauf, wenn Kunde, Artikel und Menge übereinstimmen und der Kauf vorher lag.
    Zugeordnet wird jeweils der letzte passende Kauf vor dem Storno (``merge_asof`` rückwärts).
    """
    k = verkaeufe.reset_index(names="zeile").sort_values("datum")
    s = stornos.assign(menge=-stornos["menge"]).sort_values("datum")
    treffer = pd.merge_asof(s[["datum", "kunde", "artikel", "menge"]],
                            k[["datum", "kunde", "artikel", "menge", "zeile"]],
                            on="datum", by=["kunde", "artikel", "menge"], direction="backward")
    return pd.Index(treffer["zeile"].dropna().astype("int64").unique())


def bereinigen(roh: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, Protokoll]:
    """Wendet alle Schritte der Reihe nach an und gibt (verkaeufe, stornos, protokoll) zurück."""
    p = Protokoll()
    df = roh.drop(columns=["blatt"], errors="ignore")

    n = len(df)
    df = df.drop_duplicates()
    p.eintragen("Exakte Duplikate", n, len(df),
                "Identische Zeilen (u. a. Überlappung der beiden Excel-Blätter im Dezember 2010) "
                "würden Umsatz doppelt zählen.")

    n = len(df)
    df = df[df["kunde"].notna()]
    p.eintragen("Fehlende Kunden-ID", n, len(df),
                "Ohne Kunden-ID lässt sich der Kauf keinem Kunden zuordnen; für RFM und Churn unbrauchbar.")

    df = df.assign(kunde=df["kunde"].astype("int64"))
    storno = ist_storno(df["rechnung"])
    stornos = df[storno].copy()
    n = len(df)
    df = df[~storno]
    p.eintragen("Stornos ausgegliedert", n, len(df),
                "Stornos sind kein Umsatz. Sie werden separat aufbewahrt und als Merkmal genutzt.")

    n = len(df)
    df = df.drop(index=stornierte_kaeufe(df, stornos))
    p.eintragen("Stornierte Käufe (Ausreißer)", n, len(df),
                "Käufe mit passender Storno-Gegenbuchung haben nie stattgefunden. Darunter die beiden "
                "Extrem-Ausreißer (80.995 und 74.215 Stück), die Minuten später storniert wurden.")

    n = len(df)
    df = df[df["artikel"].str.match(PRODUKT_MUSTER)]
    p.eintragen("Keine Produkte", n, len(df),
                "Porto, Gebühren, manuelle Korrekturen und Testbuchungen sind kein Kaufverhalten.")

    n = len(df)
    df = df[(df["menge"] > 0) & (df["preis"] > 0)]
    p.eintragen("Menge oder Preis ≤ 0", n, len(df),
                "Gratis- oder Korrekturbuchungen ohne Storno-Kennzeichen; kein echter Verkauf.")

    verkaeufe = df.assign(umsatz=df["menge"] * df["preis"])
    stornos = stornos.assign(umsatz=stornos["menge"] * stornos["preis"])
    return verkaeufe.reset_index(drop=True), stornos.reset_index(drop=True), p
