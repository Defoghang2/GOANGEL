import os, sys, tempfile
os.environ["BZZOIRO_TOKENS"] = "tok_aaaaaaaaaaaa"
sys.path[:0] = ["tests/stubs", "."]
from pathlib import Path
import numpy as np
import pandas as pd
import footballdata as FD

NOW = pd.Timestamp("2026-03-01")


def odds_for(r, home_wins, flip=False):
    base = np.array([2.6, 3.4, 2.9]) if r.random() < 0.5 else np.array([2.1, 3.5, 3.9])
    if home_wins is True:
        base = np.array([1.7, 3.8, 5.2])
    elif home_wins is False:
        base = np.array([4.6, 3.6, 1.8])
    h, d, a = base * (1 + r.normal(0, .02, 3))
    return (a, d, h) if flip else (h, d, a)


FD_TO_HIST = {"Man United": "Manchester United", "Man City": "Manchester City", "Nott'm Forest": "Nottingham Forest",
              "Arsenal": "Arsenal FC", "Chelsea": "Chelsea FC", "Liverpool": "Liverpool FC", "Zxq Utd": "Qwerty Town"}


def make_world(seed=0, flip=False):
    r = np.random.default_rng(seed)
    teams = list(FD_TO_HIST)
    hist_rows, e0 = [], []
    day = pd.Timestamp("2025-08-16")
    for i in range(140):
        h, a = r.choice(len(teams), 2, replace=False)
        gh, ga = int(r.poisson(1.5)), int(r.poisson(1.1))
        d = day + pd.Timedelta(days=i)
        hist_rows.append({"competition_code": "1", "date_parsed": (d + pd.Timedelta(hours=23, minutes=30)).tz_localize("UTC"),
                          "home_raw": FD_TO_HIST[teams[h]], "away_raw": FD_TO_HIST[teams[a]], "goals_home": gh, "goals_away": ga})
        ps = odds_for(r, True if gh > ga else (False if ga > gh else None), flip)
        use_ps = r.random() < 0.7
        b365 = tuple(round(x * 0.98, 2) for x in ps)
        e0.append({"Div": "E0", "Date": d.strftime("%d/%m/%Y"), "HomeTeam": teams[h], "AwayTeam": teams[a], "FTHG": gh, "FTAG": ga,
                   "B365H": b365[0], "B365D": b365[1], "B365A": b365[2],
                   "PSH": round(ps[0], 2) if use_ps else "", "PSD": round(ps[1], 2) if use_ps else "", "PSA": round(ps[2], 2) if use_ps else ""})
    bra, mex = [], []
    for i in range(60):
        d = pd.Timestamp("2025-09-01") + pd.Timedelta(days=i)
        gh, ga = int(r.poisson(1.4)), int(r.poisson(1.0))
        ps = odds_for(r, True if gh > ga else (False if ga > gh else None), flip)
        for comp, league, home, away in (("9", " Serie A", "Flamengo", "Palmeiras"), ("34", "Serie B ", "Santos", "Vila Nova"), ("19", "Liga MX", "Club America", "Chivas")):
            if comp == "19" and i >= 30:
                comp = "20"
            hist_rows.append({"competition_code": comp, "date_parsed": (d + pd.Timedelta(hours=2)).tz_localize("UTC"),
                              "home_raw": home, "away_raw": away, "goals_home": gh, "goals_away": ga})
            row = {"Country": "X", "League": league, "Season": "2025", "Date": d.strftime("%d/%m/%y"), "Home": home, "Away": away,
                   "HG": gh, "AG": ga, "PSCH": round(ps[0], 2), "PSCD": round(ps[1], 2), "PSCA": round(ps[2], 2),
                   "PSH": 9.9, "PSD": 9.9, "PSA": 9.9, "Unnamed: 25": "", "Unnamed: 31": ""}
            (bra if comp in ("9", "34") else mex).append(row)
    hist_rows.append({"competition_code": "7", "date_parsed": pd.Timestamp("2025-09-17", tz="UTC"), "home_raw": "A", "away_raw": "B", "goals_home": 1, "goals_away": 0})
    mex = mex + mex[:5]
    return pd.DataFrame(hist_rows), pd.DataFrame(e0), pd.DataFrame(bra), pd.DataFrame(mex)


def csv_bytes(df, encoding):
    return df.to_csv(index=False).encode(encoding)


def make_getter(e0, bra, mex, counter=None):
    files = {
        f"{FD.BASE_MMZ}/2526/E0.csv": csv_bytes(e0, "latin-1"),
        f"{FD.BASE_NEW}/BRA.csv": b"\xef\xbb\xbf" + csv_bytes(bra, "utf-8"),
        f"{FD.BASE_NEW}/MEX.csv": b"\xef\xbb\xbf" + csv_bytes(mex, "utf-8"),
    }

    def getter(url):
        if counter is not None:
            counter.append(url)
        return files.get(url)
    return getter


def test_parse_mmz_latin1_priority_and_two_date_formats():
    content = ("Div,Date,HomeTeam,AwayTeam,FTHG,FTAG,B365H,B365D,B365A,PSH,PSD,PSA\n"
               "E0,16/08/2025,Köln,Nott'm Forest,1,0,2.0,3.4,3.9,2.1,3.5,4.0\n"
               "E0,17/08/25,Man City,Arsenal,2,2,1.9,3.5,4.1,,,\n"
               "E0,18/08/2025,Chelsea,Liverpool,0,1,abc,3.5,4.1,,,\n").encode("latin-1")
    df = FD.parse_mmz(content)
    assert len(df) == 2
    assert list(df.loc[0, ["h", "d", "a"]]) == [2.1, 3.5, 4.0]
    assert list(df.loc[1, ["h", "d", "a"]]) == [1.9, 3.5, 4.1]
    assert df.loc[0, "date"] == pd.Timestamp("2025-08-16") and df.loc[1, "date"] == pd.Timestamp("2025-08-17")


def test_parse_new_bom_unnamed_strip_filter_and_priority():
    raw = ("\ufeffCountry,League,Season,Date,Home,Away,HG,AG,PSCH,PSCD,PSCA,PSH,PSD,PSA,B365CH,B365CD,B365CA,Unnamed: 25\n"
           "Brazil, Serie A,2025,05/04/2025,Flamengo,Santos,2,1,1.8,3.6,4.9,1.9,3.5,4.5,1.7,3.5,5.0,\n"
           "Brazil,Serie B ,2025,06/04/25,Vila Nova,Coritiba,0,0,,,,2.4,3.1,3.0,2.3,3.0,3.1,\n"
           "Brazil, Serie A,2025,07/04/2025,Bahia,Sport,1,1,,,,,,,2.2,3.2,3.3,\n").encode("utf-8")
    a = FD.parse_new(raw, "Serie A")
    assert list(a["home"]) == ["Flamengo", "Bahia"]
    assert list(a.loc[0, ["h", "d", "a"]]) == [1.8, 3.6, 4.9]
    assert list(a.loc[1, ["h", "d", "a"]]) == [2.2, 3.2, 3.3]
    b = FD.parse_new(raw, "Serie B")
    assert list(b.loc[0, ["h", "d", "a"]]) == [2.4, 3.1, 3.0]
    assert not any(str(c).startswith("Unnamed") for c in FD.parse_new(raw).columns)


def test_name_matching_abbreviations_without_manual_help():
    hist = ["Manchester United", "Manchester City", "Nottingham Forest", "Arsenal FC", "Paris Saint-Germain", "Borussia Mönchengladbach"]
    m = FD.build_name_map(["Man United", "Man City", "Nott'm Forest", "Arsenal", "Paris SG", "M'gladbach"], hist)
    assert m["Man United"] == "Manchester United" and m["Man City"] == "Manchester City"
    assert m["Nott'm Forest"] == "Nottingham Forest" and m["Arsenal"] == "Arsenal FC"
    assert m["Paris SG"] == "Paris Saint-Germain" and m["M'gladbach"] == "Borussia Mönchengladbach"


def test_attach_end_to_end_coverage_overround_and_traps():
    hist, e0, bra, mex = make_world()
    out, rep = FD.attach_odds(hist, Path(tempfile.mkdtemp()), now=NOW, getter=make_getter(e0, bra, mex))
    print(rep["coverage_all"], rep["mean_overround"], rep["home_signal"])
    avail = hist["competition_code"] != "7"
    assert out.loc[avail, "odds_markets"].notna().all() and out.loc[~avail, "odds_markets"].isna().all()
    assert rep["rows_unavailable_leagues"] == 1
    assert 1.02 <= rep["mean_overround"] <= 1.15 and rep["home_signal"] > 0.03
    assert rep["coverage_all"] > 0.99
    sample = out[out["competition_code"] == "9"].iloc[0]["odds_markets"]
    assert set(sample) == {"home", "draw", "away"}
    brazil_b = out[out["competition_code"] == "34"].iloc[0]["odds_markets"]
    assert brazil_b["home"] != 9.9
    assert rep["per_comp"]["1"]["hist_matched"] == 140
    assert rep["per_comp"]["1"]["unmatched_teams"] == []


def test_team_learned_from_scores_when_name_is_unrecognisable():
    hist, e0, bra, mex = make_world()
    out, rep = FD.attach_odds(hist, Path(tempfile.mkdtemp()), now=NOW, getter=make_getter(e0, bra, mex))
    rows_q = hist[(hist["competition_code"] == "1") & ((hist["home_raw"] == "Qwerty Town") | (hist["away_raw"] == "Qwerty Town"))]
    assert len(rows_q) > 5 and out.loc[rows_q.index, "odds_markets"].notna().mean() > 0.9


def test_inverted_home_away_is_rejected():
    hist, e0, bra, mex = make_world(flip=True)
    try:
        FD.attach_odds(hist, Path(tempfile.mkdtemp()), now=NOW, getter=make_getter(e0, bra, mex))
    except FD.OddsSourceError as exc:
        assert "inversion" in str(exc) or "overround" in str(exc)
        return
    raise AssertionError("inversion non détectée")


def test_overround_out_of_range_is_rejected():
    hist, e0, bra, mex = make_world()
    for col in ("B365H", "B365D", "B365A", "PSH", "PSD", "PSA"):
        e0[col] = 3.0
    for col in ("PSCH", "PSCD", "PSCA"):
        bra[col] = 3.0
        mex[col] = 3.0
    try:
        FD.attach_odds(hist, Path(tempfile.mkdtemp()), now=NOW, getter=make_getter(e0, bra, mex))
    except FD.OddsSourceError as exc:
        assert "overround" in str(exc)
        return
    raise AssertionError("overround hors bornes non détecté")


def test_no_network_and_no_cache_raises_instead_of_wiping_odds():
    hist, *_ = make_world()
    try:
        FD.attach_odds(hist, Path(tempfile.mkdtemp()), now=NOW, getter=lambda url: None)
    except FD.OddsSourceError as exc:
        assert "aucun fichier" in str(exc)
        return
    raise AssertionError("absence de données non signalée")


def test_cache_is_reused_and_404_is_remembered():
    hist, e0, bra, mex = make_world()
    cache = Path(tempfile.mkdtemp())
    calls = []
    FD.attach_odds(hist, cache, now=NOW, getter=make_getter(e0, bra, mex, calls))
    first = len(calls)
    FD.attach_odds(hist, cache, now=NOW, getter=make_getter(e0, bra, mex, calls))
    assert len(calls) == first
    assert list(cache.glob("*.404")) == []


def test_unmatched_teams_report_is_written():
    hist, e0, bra, mex = make_world()
    mask = (hist["home_raw"] == "Qwerty Town") | (hist["away_raw"] == "Qwerty Town")
    hist.loc[mask, ["goals_home", "goals_away"]] += 7
    out, rep = FD.attach_odds(hist, Path(tempfile.mkdtemp()), now=NOW, getter=make_getter(e0, bra, mex))
    path = Path(tempfile.mkdtemp()) / "unmatched.txt"
    n = FD.write_unmatched(rep, path)
    assert n >= 1 and "Zxq Utd" in path.read_text(encoding="utf-8")
