"""Grobe Wertabschätzung einer Rückgewinnungskampagne mit offen genannten Annahmen.

Rechnung je Strategie (rückblickend auf dem Test-Stichtag, an dem das echte Kaufverhalten bekannt ist):

    Nutzen  = zurückgewonnene Kunden × Wert je zurückgewonnenem Kunden
    Kosten  = angeschriebene Kunden × Kosten je Kontakt
    Ergebnis = Nutzen − Kosten

- Zurückgewonnen werden kann nur, wer ohne Kampagne tatsächlich abgewandert wäre (Label = 1), und davon
  nur der Anteil ``erfolgsquote``.
- Wert je Kunde = sein durchschnittlicher Quartalsumsatz der letzten 12 Monate × ``marge`` × ``quartale``.
- Kosten je Kontakt decken Gutschein und Versand; ein Gutschein für Kunden, die ohnehin gekauft hätten
  (Mitnahmeeffekt), ist darin pauschal enthalten.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Annahmen:
    """Annahmen eines Szenarios. Alle Beträge in Pfund Sterling (Währung des Datensatzes)."""

    name: str
    erfolgsquote: float
    kosten_je_kontakt: float
    marge: float
    quartale: int = 1


SZENARIEN = [
    Annahmen("pessimistisch", erfolgsquote=0.05, kosten_je_kontakt=8.0, marge=0.20),
    Annahmen("realistisch", erfolgsquote=0.10, kosten_je_kontakt=5.0, marge=0.25),
    Annahmen("optimistisch", erfolgsquote=0.15, kosten_je_kontakt=3.0, marge=0.30, quartale=2),
]


def quartalswert(umsatz_12m: pd.Series) -> pd.Series:
    """Durchschnittlicher Umsatz je Quartal aus den letzten 12 Monaten."""
    return umsatz_12m / 4


def kampagnenwert(auswahl: np.ndarray, churn: np.ndarray, wert_quartal: np.ndarray, a: Annahmen) -> dict:
    """Bewertet eine Auswahl angeschriebener Kunden (bool-Maske) unter den Annahmen ``a``."""
    auswahl = np.asarray(auswahl, dtype=bool)
    churn = np.asarray(churn, dtype=bool)
    getroffen = auswahl & churn
    nutzen = a.erfolgsquote * float(np.sum(np.asarray(wert_quartal)[getroffen])) * a.marge * a.quartale
    kosten = a.kosten_je_kontakt * int(auswahl.sum())
    return {
        "angeschrieben": int(auswahl.sum()),
        "davon_abwanderer": int(getroffen.sum()),
        "trefferquote": float(getroffen.sum() / max(auswahl.sum(), 1)),
        "nutzen": nutzen,
        "kosten": kosten,
        "ergebnis": nutzen - kosten,
        "roi": (nutzen - kosten) / kosten if kosten else float("nan"),
    }


def top_anteil(score: np.ndarray, anteil: float) -> np.ndarray:
    """Maske der ``anteil`` Kunden mit dem höchsten Score."""
    n = int(round(anteil * len(score)))
    maske = np.zeros(len(score), dtype=bool)
    maske[np.argsort(-np.asarray(score), kind="stable")[:n]] = True
    return maske
