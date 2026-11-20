import pandas as pd
import pytest

from trace_kunden.churn import (
    FEATURES,
    churn_datensatz,
    churn_label,
    kundenfeatures,
    population,
)

STICHTAG = pd.Timestamp("2011-06-01")


def test_population_nur_kunden_mit_kauf_im_letzten_jahr(verkaeufe):
    assert population(verkaeufe, STICHTAG).tolist() == [1, 2]


def test_label_schaut_nur_ins_fenster_nach_dem_stichtag(verkaeufe):
    y = churn_label(verkaeufe, pd.Index([1, 2]), STICHTAG)
    assert y.to_dict() == {1: 0, 2: 1}
    # Liegt der Folgekauf außerhalb der 90 Tage, zählt er nicht.
    assert churn_label(verkaeufe, pd.Index([1]), STICHTAG, fenster_tage=10).iloc[0] == 1


def test_features_nutzen_keine_daten_ab_dem_stichtag(verkaeufe, stornos):
    """Kernprüfung gegen Data Leakage: Daten ab dem Stichtag dürfen die Features nicht verändern."""
    vorher = kundenfeatures(verkaeufe, stornos, STICHTAG)
    zukunft = verkaeufe[verkaeufe["datum"] >= STICHTAG].assign(umsatz=1e9)
    manipuliert = pd.concat([verkaeufe[verkaeufe["datum"] < STICHTAG], zukunft])
    stornos_manipuliert = stornos.assign(umsatz=-1e9)
    nachher = kundenfeatures(manipuliert, stornos_manipuliert, STICHTAG)
    pd.testing.assert_frame_equal(vorher, nachher)


def test_features_werte(verkaeufe, stornos):
    f = kundenfeatures(verkaeufe, stornos, STICHTAG)
    k1 = f.loc[1]
    assert list(f.columns) == FEATURES
    assert k1["recency_tage"] == 12          # 20.05. → 01.06.
    assert k1["frequenz"] == 3
    assert k1["umsatz_gesamt"] == 220
    assert k1["kaeufe_90"] == 2              # 10.03. und 20.05. liegen in den 90 Tagen davor
    assert k1["artikel_vielfalt"] == 2
    k2 = f.loc[2]
    assert k2["mittlerer_abstand_tage"] == 365   # Einmalkäufer
    assert k2["storno_quote"] == pytest.approx(0.5)  # 1 Storno vor dem Stichtag, 1 Verkaufszeile
    assert k2["ist_uk"] == 0


def test_datensatz_verweigert_unvollstaendiges_label_fenster(verkaeufe, stornos):
    with pytest.raises(ValueError, match="Datenende"):
        churn_datensatz(verkaeufe, stornos, pd.Timestamp("2011-06-20"))
