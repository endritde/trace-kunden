from trace_kunden.bereinigung import bereinigen


def test_jeder_schritt_entfernt_genau_die_erwartete_zeile(roh):
    verkaeufe, _stornos, protokoll = bereinigen(roh)
    tabelle = protokoll.tabelle()
    assert tabelle["entfernt"].tolist() == [1, 1, 2, 1, 1, 2]
    assert tabelle["verbleibend"].iloc[-1] == len(verkaeufe) == 2


def test_verkaeufe_sind_sauber(roh):
    verkaeufe, _, _ = bereinigen(roh)
    assert (verkaeufe["menge"] > 0).all() and (verkaeufe["preis"] > 0).all()
    assert not verkaeufe["rechnung"].str.startswith("C").any()
    assert set(verkaeufe["artikel"]) == {"85123A", "21212"}
    assert verkaeufe["umsatz"].tolist() == [15.0, 5.0]


def test_stornos_werden_aufbewahrt_nicht_verworfen(roh):
    _, stornos, _ = bereinigen(roh)
    assert stornos["rechnung"].tolist() == ["C1003", "C1009"]
    assert stornos["umsatz"].iloc[0] == -10.0


def test_storno_entfernt_nur_den_passenden_frueheren_kauf(roh):
    verkaeufe, _, _ = bereinigen(roh)
    assert 80995 not in verkaeufe["menge"].tolist()
    # Der Storno C1003 (Kunde 1, Artikel 22423) hat keinen passenden Kauf und entfernt nichts.
    assert (verkaeufe["kunde"] == 1).sum() == 1
