"""Kleine, handgebaute Datensätze mit bekanntem Ergebnis."""

import pandas as pd
import pytest


def zeile(rechnung: str, artikel: str, menge: int, preis: float, datum: str, kunde: float | None,
          land: str = "United Kingdom") -> dict:
    return {"rechnung": rechnung, "artikel": artikel, "beschreibung": "x", "menge": menge, "datum":
            pd.Timestamp(datum), "preis": preis, "kunde": kunde, "land": land}


@pytest.fixture
def roh() -> pd.DataFrame:
    zeilen = [
        zeile("1001", "85123A", 6, 2.5, "2011-01-05", 1),
        zeile("1001", "85123A", 6, 2.5, "2011-01-05", 1),          # Duplikat
        zeile("1002", "22423", 2, 10.0, "2011-02-10", None),        # ohne Kunde
        zeile("C1003", "22423", -1, 10.0, "2011-02-11", 1),         # Storno
        zeile("1004", "POST", 1, 18.0, "2011-02-12", 2),            # Porto
        zeile("1005", "21212", 0, 0.5, "2011-03-01", 2),            # Menge 0
        zeile("1006", "21212", 3, 0.0, "2011-03-02", 2),            # Preis 0
        zeile("1007", "23843", 80995, 2.08, "2011-03-03", 3),       # Ausreißer
        zeile("1008", "21212", 10, 0.5, "2011-03-04", 2),
        zeile("C1009", "23843", -80995, 2.08, "2011-03-03 10:05", 3),  # storniert den Ausreißer
    ]
    df = pd.DataFrame(zeilen)
    df["kunde"] = df["kunde"].astype("Int64")
    return df


@pytest.fixture
def verkaeufe() -> pd.DataFrame:
    """Drei Kunden, Stichtag 2011-06-01.

    - Kunde 1: kauft regelmäßig, auch nach dem Stichtag → kein Churn
    - Kunde 2: letzter Kauf vor dem Stichtag, danach nichts → Churn
    - Kunde 3: letzter Kauf vor über einem Jahr → nicht in der Population
    """
    zeilen = [
        ("1", 1, "2011-01-10", 100.0, "A"), ("2", 1, "2011-03-10", 50.0, "B"),
        ("3", 1, "2011-05-20", 70.0, "A"), ("4", 1, "2011-06-15", 80.0, "C"),
        ("5", 2, "2011-04-01", 30.0, "A"),
        ("6", 3, "2010-03-01", 500.0, "D"), ("7", 3, "2011-07-01", 20.0, "D"),
    ]
    df = pd.DataFrame(zeilen, columns=["rechnung", "kunde", "datum", "umsatz", "artikel"])
    df["datum"] = pd.to_datetime(df["datum"])
    df["land"] = "United Kingdom"
    df.loc[df["kunde"] == 2, "land"] = "Germany"
    return df


@pytest.fixture
def stornos() -> pd.DataFrame:
    df = pd.DataFrame({"rechnung": ["C9", "C10"], "kunde": [2, 2],
                       "datum": pd.to_datetime(["2011-04-02", "2011-07-01"]), "umsatz": [-10.0, -5.0]})
    return df
