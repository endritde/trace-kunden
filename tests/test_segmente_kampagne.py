import numpy as np
import pandas as pd
import pytest

from trace_kunden.kampagne import Annahmen, kampagnenwert, quartalswert, top_anteil
from trace_kunden.modelle import bewerten
from trace_kunden.segmente import rfm_matrix, rfm_scores, rfm_tabelle, segmente_benennen


def test_rfm_tabelle(verkaeufe):
    rfm = rfm_tabelle(verkaeufe, pd.Timestamp("2011-06-01"))
    assert rfm.loc[1].tolist() == [12, 3, 220]
    assert rfm.loc[3, "recency"] == 457
    assert 3 in rfm.index and len(rfm) == 3


def test_rfm_scores_recency_umgekehrt():
    rfm = pd.DataFrame({"recency": range(1, 11), "frequency": range(1, 11), "monetary": range(1, 11)})
    s = rfm_scores(rfm)
    assert s["R"].iloc[0] == 5 and s["R"].iloc[-1] == 1
    assert s["F"].iloc[0] == 1 and s["F"].iloc[-1] == 5


def test_rfm_matrix_ist_standardisiert():
    rng = np.random.default_rng(0)
    rfm = pd.DataFrame({"recency": rng.integers(0, 700, 200), "frequency": rng.integers(1, 50, 200),
                        "monetary": rng.lognormal(6, 1.5, 200)})
    x, _ = rfm_matrix(rfm)
    assert np.allclose(x.mean(axis=0), 0, atol=1e-9) and np.allclose(x.std(axis=0), 1)


def test_segmentnamen_haengen_nicht_an_der_clusternummer():
    rfm = pd.DataFrame({"recency": [10, 12, 300, 320, 20, 25, 400, 420],
                        "frequency": [20, 25, 15, 18, 1, 1, 1, 2],
                        "monetary": [1] * 8})
    a = segmente_benennen(rfm, np.array([0, 0, 1, 1, 2, 2, 3, 3]))
    b = segmente_benennen(rfm, np.array([3, 3, 2, 2, 1, 1, 0, 0]))
    assert a.tolist() == b.tolist()
    assert a.iloc[0] == "Top-Kunden" and a.iloc[2] == "Gefährdete Stammkunden"


def test_kampagnenwert_rechnet_wie_dokumentiert():
    a = Annahmen("test", erfolgsquote=0.1, kosten_je_kontakt=5.0, marge=0.25, quartale=2)
    auswahl = np.array([True, True, False, True])
    churn = np.array([1, 0, 1, 1])
    wert = np.array([1000.0, 500.0, 9999.0, 200.0])
    r = kampagnenwert(auswahl, churn, wert, a)
    assert r["angeschrieben"] == 3 and r["davon_abwanderer"] == 2
    assert r["nutzen"] == pytest.approx(0.1 * 1200 * 0.25 * 2)
    assert r["kosten"] == 15 and r["ergebnis"] == pytest.approx(60 - 15)


def test_top_anteil_waehlt_hoechste_scores():
    assert top_anteil(np.array([0.1, 0.9, 0.5, 0.7, 0.2]), 0.4).tolist() == [False, True, False, True, False]


def test_bewertung_perfektes_modell():
    y = pd.Series([0, 0, 1, 1, 1])
    b = bewerten(y, np.array([0.1, 0.2, 0.8, 0.9, 0.7]))
    assert b.roc_auc == 1 and b.precision == 1 and b.recall == 1
    assert b.konfusion.tolist() == [[2, 0], [0, 3]]


def test_quartalswert_nur_letzte_12_monate_vor_stichtag(verkaeufe):
    w = quartalswert(verkaeufe, pd.Index([1, 2, 3]), pd.Timestamp("2011-06-01"))
    assert w.tolist() == [220 / 4, 30 / 4, 0.0]  # Kunde 3: Kauf 2010-03 zu alt, 2011-07 zu spät
