"""Lädt den öffentlichen Datensatz „Online Retail II“ (UCI) herunter und speichert ihn als Parquet.

Aufruf: ``uv run python -m trace_kunden.daten``. Die Rohdaten landen in ``data/`` und werden nicht
committet. Das Einlesen der Excel-Datei dauert einige Minuten und braucht kurzzeitig ca. 1–2 GB RAM;
danach arbeiten Notebook und Tests nur noch mit der kompakten Parquet-Datei.
"""

import io
import urllib.request
import zipfile

import pandas as pd

from trace_kunden.konfig import DATEN, QUELLE_URL, ROH_PARQUET, ROH_ZIP

SPALTEN = {
    "Invoice": "rechnung",
    "StockCode": "artikel",
    "Description": "beschreibung",
    "Quantity": "menge",
    "InvoiceDate": "datum",
    "Price": "preis",
    "Customer ID": "kunde",
    "Country": "land",
}


def herunterladen() -> None:
    """Lädt das ZIP von UCI, falls es noch nicht lokal liegt."""
    if ROH_ZIP.exists():
        return
    DATEN.mkdir(exist_ok=True)
    print(f"Lade {QUELLE_URL} …")
    urllib.request.urlretrieve(QUELLE_URL, ROH_ZIP)


def excel_zu_parquet() -> pd.DataFrame:
    """Liest beide Tabellenblätter (2009–2010, 2010–2011) nacheinander ein und vereinheitlicht die Typen."""
    with zipfile.ZipFile(ROH_ZIP) as z:
        name = next(n for n in z.namelist() if n.endswith(".xlsx"))
        inhalt = io.BytesIO(z.read(name))
    teile = []
    for blatt in ("Year 2009-2010", "Year 2010-2011"):
        print(f"Lese Blatt {blatt} …")
        df = pd.read_excel(inhalt, sheet_name=blatt, engine="openpyxl", dtype={"Invoice": str, "StockCode": str})
        df = df.rename(columns=SPALTEN)
        df["blatt"] = blatt.removeprefix("Year ")
        teile.append(df)
    df = pd.concat(teile, ignore_index=True)
    df["rechnung"] = df["rechnung"].astype(str)
    df["artikel"] = df["artikel"].astype(str)
    df["beschreibung"] = df["beschreibung"].astype("string")
    df["land"] = df["land"].astype("category")
    df["kunde"] = df["kunde"].astype("Int64")
    df["datum"] = pd.to_datetime(df["datum"])
    df["menge"] = df["menge"].astype("int32")
    df["preis"] = df["preis"].astype("float64")
    df["blatt"] = df["blatt"].astype("category")
    df.to_parquet(ROH_PARQUET, index=False)
    return df


def lade_rohdaten() -> pd.DataFrame:
    """Gibt die Rohdaten zurück (beide Blätter, unverändert bis auf Spaltennamen und Typen)."""
    if not ROH_PARQUET.exists():
        raise FileNotFoundError("Rohdaten fehlen. Bitte zuerst `uv run python -m trace_kunden.daten` ausführen.")
    return pd.read_parquet(ROH_PARQUET)


if __name__ == "__main__":
    herunterladen()
    roh = excel_zu_parquet()
    print(f"{len(roh):,} Zeilen gespeichert in {ROH_PARQUET}")
