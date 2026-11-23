import os, sys, tempfile
os.environ["BZZOIRO_TOKENS"] = "tok_aaaaaaaaaaaa"
sys.path[:0] = ["tests/stubs", "tests", "."]
from pathlib import Path
import numpy as np
import pandas as pd
import features as F
import footballdata as FD
import test_ratings as TR
from test_footballdata import csv_bytes


def build_hist_and_csv():
    rows, _ = TR.world(1200, comps=1, T=12)
    df = F.process_matches(rows, allowed_statuses=("FINISHED",))
    teams = {}
    for r in df.itertuples():
        teams[r.home_raw] = f"FD {r.home_raw}"
        teams[r.away_raw] = f"FD {r.away_raw}"
    fd = pd.DataFrame({
        "Div": "E0", "Date": pd.to_datetime(df["date_parsed"]).dt.strftime("%d/%m/%Y"),
        "HomeTeam": df["home_raw"].map(teams), "AwayTeam": df["away_raw"].map(teams),
        "FTHG": df["goals_home"], "FTAG": df["goals_away"],
        "PSH": [o["home"] * 0.98 for o in df["odds_markets"]], "PSD": [o["draw"] * 0.98 for o in df["odds_markets"]],
        "PSA": [o["away"] * 0.98 for o in df["odds_markets"]],
    })
    return df, fd


def test_footballdata_replaces_bzzoiro_odds_in_the_features():
    import main as M
    df, fd = build_hist_and_csv()
    years = sorted(set(FD.seasons_for_range(pd.to_datetime(df["date_parsed"]).min().tz_localize(None), pd.to_datetime(df["date_parsed"]).max().tz_localize(None))))
    files = {f"{FD.BASE_MMZ}/{FD.season_code(y)}/E0.csv": csv_bytes(fd, "latin-1") for y in years}
    FD.Fetcher._http_get = staticmethod(lambda url: files.get(url))
    M._FOOTBALLDATA_DIR = Path(tempfile.mkdtemp())
    out = M._attach_footballdata_odds(df)
    covered = out["odds_markets"].map(lambda m: isinstance(m, dict)).mean()
    assert covered > 0.97
    before = df.iloc[0]["odds_markets"]["home"]
    after = out.iloc[0]["odds_markets"]["home"]
    assert abs(after - before * 0.98) < 0.02
    X, *_ = F.create_features_with_context(out, strict_competition=True)
    names = F.FEATURE_NAMES
    assert np.isfinite(X[:, names.index("mkt_p_home")]).mean() > 0.97
    overround = X[:, names.index("mkt_overround")]
    assert 1.02 <= np.nanmean(overround) <= 1.15


def test_check_odds_cli_reports_criteria(capsys=None):
    import main as M
    df, fd = build_hist_and_csv()
    years = sorted(set(FD.seasons_for_range(pd.to_datetime(df["date_parsed"]).min().tz_localize(None), pd.to_datetime(df["date_parsed"]).max().tz_localize(None))))
    files = {f"{FD.BASE_MMZ}/{FD.season_code(y)}/E0.csv": csv_bytes(fd, "latin-1") for y in years}
    FD.Fetcher._http_get = staticmethod(lambda url: files.get(url))
    M._FOOTBALLDATA_DIR = Path(tempfile.mkdtemp())
    out = M._attach_footballdata_odds(df)
    M._atomic_write_pickle(Path(M.CACHE_DIR) / "historical.pkl", out)
    M._stage_save_features(out)
    import io, contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        code = M._check_odds_report()
    text = buf.getvalue()
    print(text)
    assert "1. Couverture" in text and "2. Overround" in text and "OK" in text


def test_training_logs_top5_importances_with_market_features():
    import io, contextlib, main as M
    df, fd = build_hist_and_csv()
    years = sorted(set(FD.seasons_for_range(pd.to_datetime(df["date_parsed"]).min().tz_localize(None), pd.to_datetime(df["date_parsed"]).max().tz_localize(None))))
    files = {f"{FD.BASE_MMZ}/{FD.season_code(y)}/E0.csv": csv_bytes(fd, "latin-1") for y in years}
    FD.Fetcher._http_get = staticmethod(lambda url: files.get(url))
    M._FOOTBALLDATA_DIR = Path(tempfile.mkdtemp())
    out = M._attach_footballdata_odds(df)
    M._atomic_write_pickle(Path(M.CACHE_DIR) / "historical.pkl", out)
    M._stage_save_features(out)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        M._stage_train_ml_model("RF")
    line = [l for l in buf.getvalue().split("\n") if "top 5 importances" in l]
    print(line)
    assert line and "mkt_" in line[0]
