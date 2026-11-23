import difflib
import io
import logging
import re
import time
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests

logger = logging.getLogger("goangel.footballdata")

BASE_MMZ = "https://www.football-data.co.uk/mmz4281"
BASE_NEW = "https://www.football-data.co.uk/new"

MMZ_CODES = {
    "1": "E0", "3": "SP1", "4": "I1", "5": "D1", "6": "F1", "10": "N1", "2": "P1",
    "12": "E1", "14": "B1", "11": "T1", "13": "SC0", "94": "D2", "100": "I2",
    "89": "F2", "38": "SP2",
}
NEW_CODES = {
    "9": ("BRA", "Serie A"), "34": ("BRA", "Serie B"), "18": ("USA", None),
    "85": ("ARG", None), "49": ("JPN", None), "96": ("AUT", None),
    "84": ("DNK", None), "15": ("SWZ", None), "19": ("MEX", None), "20": ("MEX", None),
}
UNAVAILABLE = ("7", "8", "17", "32", "33")

MMZ_ODDS = (("PSH", "PSD", "PSA"), ("B365H", "B365D", "B365A"))
NEW_ODDS = (
    ("PSCH", "PSCD", "PSCA"), ("PSH", "PSD", "PSA"),
    ("B365CH", "B365CD", "B365CA"), ("B365H", "B365D", "B365A"),
)

FUZZY_CUTOFF = 0.85
OVERROUND_RANGE = (1.02, 1.15)
MIN_HOME_SIGNAL = 0.03
FIRST_SEASON_YEAR = 2018
DATE_TOLERANCE_DAYS = 1
CURRENT_SEASON_REFRESH_HOURS = 20.0
REQUEST_PAUSE_SECONDS = 0.25
REQUEST_TIMEOUT_SECONDS = 30

GENERIC_TOKENS = {"fc", "cf", "afc", "sc", "ac", "as", "us", "ss", "ssc", "fk", "sk", "bk", "if", "club", "de", "la", "el", "cd", "ud", "sd", "rc", "ca"}

TEAM_ALIASES = {
    "man united": ["manchester united"], "man city": ["manchester city"],
    "nott m forest": ["nottingham forest"], "m gladbach": ["borussia monchengladbach", "borussia m gladbach", "monchengladbach"],
    "paris sg": ["paris saint germain", "paris saint-germain"], "ath madrid": ["atletico madrid"],
    "ath bilbao": ["athletic club", "athletic bilbao"], "sp gijon": ["sporting gijon"],
    "betis": ["real betis"], "vallecano": ["rayo vallecano"], "espanol": ["espanyol"],
    "la coruna": ["deportivo la coruna"], "celta": ["celta vigo"], "sociedad": ["real sociedad"],
    "alaves": ["deportivo alaves"], "wolves": ["wolverhampton wanderers", "wolverhampton"],
    "spurs": ["tottenham hotspur", "tottenham"], "newcastle": ["newcastle united"],
    "west ham": ["west ham united"], "west brom": ["west bromwich albion"],
    "sheffield united": ["sheffield utd"], "sheffield weds": ["sheffield wednesday"],
    "leeds": ["leeds united"], "norwich": ["norwich city"], "brighton": ["brighton hove albion", "brighton and hove albion"],
    "inter": ["internazionale", "inter milan"], "milan": ["ac milan"], "roma": ["as roma"],
    "verona": ["hellas verona"], "ein frankfurt": ["eintracht frankfurt"], "leverkusen": ["bayer leverkusen", "bayer 04 leverkusen"],
    "dortmund": ["borussia dortmund"], "fc koln": ["koln", "1 fc koln", "fc cologne"], "mainz": ["mainz 05", "1 fsv mainz 05"],
    "hertha": ["hertha berlin", "hertha bsc"], "st pauli": ["fc st pauli"], "stuttgart": ["vfb stuttgart"],
    "bayern munich": ["bayern munchen", "fc bayern munchen"], "hoffenheim": ["tsg hoffenheim", "tsg 1899 hoffenheim"],
    "lyon": ["olympique lyonnais"], "marseille": ["olympique de marseille", "olympique marseille"],
    "st etienne": ["saint etienne", "as saint etienne"], "monaco": ["as monaco"], "nice": ["ogc nice"], "lille": ["losc lille", "lille osc"],
    "brest": ["stade brestois", "stade brestois 29"], "reims": ["stade de reims"], "rennes": ["stade rennais", "stade rennais fc"],
    "sporting cp": ["sporting lisbon", "sporting"], "benfica": ["sl benfica"], "porto": ["fc porto"],
    "galatasaray": ["galatasaray sk"], "fenerbahce": ["fenerbahce sk"], "besiktas": ["besiktas jk"],
    "ajax": ["ajax amsterdam"], "psv": ["psv eindhoven"], "feyenoord": ["feyenoord rotterdam"],
    "celtic": ["celtic glasgow"], "rangers": ["rangers glasgow"],
}


class OddsSourceError(RuntimeError):
    pass


def normalize(name: Any) -> str:
    text = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode("ascii").lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    tokens = ["saint" if t == "st" else t for t in text.split()]
    return " ".join(tokens)


def _core_tokens(name: str) -> List[str]:
    tokens = [t for t in normalize(name).split() if t not in GENERIC_TOKENS]
    return tokens or normalize(name).split()


def name_score(fd_name: str, hist_name: str) -> float:
    a, b = normalize(fd_name), normalize(hist_name)
    if a == b:
        return 1.0
    for alias in TEAM_ALIASES.get(a, []):
        if normalize(alias) == b:
            return 1.0
    ta, tb = _core_tokens(fd_name), _core_tokens(hist_name)
    if ta == tb:
        return 0.98
    if ta and len(ta) <= len(tb):
        pool = list(tb)
        matched = 0
        for token in ta:
            hit = next((p for p in pool if p == token or (len(token) >= 3 and p.startswith(token))), None)
            if hit is not None:
                pool.remove(hit)
                matched += 1
        if matched == len(ta) and len(tb) - len(ta) <= 2:
            return 0.9
    return difflib.SequenceMatcher(None, " ".join(ta), " ".join(tb)).ratio()


def _parse_dates(series: pd.Series) -> pd.Series:
    text = series.astype(str).str.strip()
    parsed = pd.to_datetime(text, format="%d/%m/%Y", errors="coerce")
    fallback = pd.to_datetime(text, format="%d/%m/%y", errors="coerce")
    return parsed.fillna(fallback).dt.normalize()


def _pick_odds(df: pd.DataFrame, triples: Tuple[Tuple[str, str, str], ...]) -> np.ndarray:
    out = np.full((len(df), 3), np.nan)
    for cols in triples:
        if not all(c in df.columns for c in cols):
            continue
        vals = np.column_stack([pd.to_numeric(df[c], errors="coerce").to_numpy(dtype=float) for c in cols])
        valid = np.isfinite(vals).all(axis=1) & (vals > 1.0).all(axis=1) & (vals < 1000.0).all(axis=1)
        fill = valid & ~np.isfinite(out).all(axis=1)
        out[fill] = vals[fill]
    return out


def _finish(df: pd.DataFrame, home: str, away: str, hg: str, ag: str, triples) -> pd.DataFrame:
    odds = _pick_odds(df, triples)
    out = pd.DataFrame({
        "date": _parse_dates(df["Date"]),
        "home": df[home].astype(str).str.strip(),
        "away": df[away].astype(str).str.strip(),
        "hg": pd.to_numeric(df[hg], errors="coerce") if hg in df.columns else np.nan,
        "ag": pd.to_numeric(df[ag], errors="coerce") if ag in df.columns else np.nan,
        "h": odds[:, 0], "d": odds[:, 1], "a": odds[:, 2],
    })
    out = out[out["date"].notna() & out["home"].ne("") & out["away"].ne("") & np.isfinite(out[["h", "d", "a"]]).all(axis=1)]
    return out.drop_duplicates(subset=["date", "home", "away"]).reset_index(drop=True)


def parse_mmz(content: bytes) -> pd.DataFrame:
    df = pd.read_csv(io.BytesIO(content), encoding="latin-1", dtype=str, on_bad_lines="skip")
    df.columns = [str(c).strip() for c in df.columns]
    if not {"Date", "HomeTeam", "AwayTeam"} <= set(df.columns):
        return pd.DataFrame(columns=["date", "home", "away", "hg", "ag", "h", "d", "a"])
    return _finish(df, "HomeTeam", "AwayTeam", "FTHG", "FTAG", MMZ_ODDS)


def parse_new(content: bytes, league: Optional[str] = None) -> pd.DataFrame:
    df = pd.read_csv(io.BytesIO(content), encoding="utf-8-sig", dtype=str, on_bad_lines="skip")
    df = df[[c for c in df.columns if not str(c).startswith("Unnamed")]]
    df.columns = [str(c).strip() for c in df.columns]
    if not {"Date", "Home", "Away"} <= set(df.columns):
        return pd.DataFrame(columns=["date", "home", "away", "hg", "ag", "h", "d", "a"])
    if league is not None and "League" in df.columns:
        df = df[df["League"].astype(str).str.strip() == league]
    return _finish(df, "Home", "Away", "HG", "AG", NEW_ODDS)


def season_code(year: int) -> str:
    return f"{year % 100:02d}{(year + 1) % 100:02d}"


def season_year(ts: pd.Timestamp) -> int:
    return ts.year if ts.month >= 7 else ts.year - 1


def seasons_for_range(min_date: pd.Timestamp, max_date: pd.Timestamp) -> List[int]:
    first = max(season_year(min_date), FIRST_SEASON_YEAR)
    return list(range(first, season_year(max_date) + 1))


class Fetcher:
    def __init__(self, cache_dir: Path, current_season: int,
                 getter: Optional[Callable[[str], Optional[bytes]]] = None, pause: float = REQUEST_PAUSE_SECONDS):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.current_season = current_season
        self.pause = pause
        self.getter = getter or self._http_get
        self.downloaded = 0
        self.from_cache = 0
        self.missing: List[str] = []

    @staticmethod
    def _http_get(url: str) -> Optional[bytes]:
        for attempt in range(3):
            try:
                resp = requests.get(url, timeout=REQUEST_TIMEOUT_SECONDS)
            except requests.RequestException as exc:
                logger.warning("football-data %s : %s", url, exc)
                time.sleep(1.5 * (attempt + 1))
                continue
            if resp.status_code == 200:
                return resp.content
            if resp.status_code == 404:
                return None
            time.sleep(1.5 * (attempt + 1))
        return None

    def get(self, url: str, cache_name: str, is_current: bool) -> Optional[bytes]:
        path = self.cache_dir / cache_name
        absent = self.cache_dir / (cache_name + ".404")
        if path.exists():
            age_hours = (time.time() - path.stat().st_mtime) / 3600.0
            if not is_current or age_hours < CURRENT_SEASON_REFRESH_HOURS:
                self.from_cache += 1
                return path.read_bytes()
        elif absent.exists() and not is_current:
            return None
        time.sleep(self.pause)
        content = self.getter(url)
        if content is None:
            if path.exists():
                self.from_cache += 1
                return path.read_bytes()
            absent.write_text(datetime.now(timezone.utc).isoformat())
            self.missing.append(url)
            return None
        path.write_bytes(content)
        absent.unlink(missing_ok=True)
        self.downloaded += 1
        return content


def load_sources(comps: List[str], min_date: pd.Timestamp, max_date: pd.Timestamp, fetcher: Fetcher) -> Dict[str, pd.DataFrame]:
    years = seasons_for_range(min_date, max_date)
    out: Dict[str, pd.DataFrame] = {}
    new_cache: Dict[Tuple[str, Optional[str]], pd.DataFrame] = {}
    for comp in comps:
        if comp in MMZ_CODES:
            frames = []
            for year in years:
                code = season_code(year)
                content = fetcher.get(f"{BASE_MMZ}/{code}/{MMZ_CODES[comp]}.csv", f"mmz_{code}_{MMZ_CODES[comp]}.csv", year == fetcher.current_season)
                if content:
                    frames.append(parse_mmz(content))
            out[comp] = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        elif comp in NEW_CODES:
            country, league = NEW_CODES[comp]
            key = (country, league)
            if key not in new_cache:
                content = fetcher.get(f"{BASE_NEW}/{country}.csv", f"new_{country}.csv", True)
                new_cache[key] = parse_new(content, league) if content else pd.DataFrame()
            out[comp] = new_cache[key]
    return out


def build_name_map(fd_names: List[str], hist_names: List[str]) -> Dict[str, str]:
    scored = []
    for fd in fd_names:
        for hist in hist_names:
            score = name_score(fd, hist)
            if score >= FUZZY_CUTOFF:
                scored.append((score, fd, hist))
    mapping: Dict[str, str] = {}
    used: set = set()
    for score, fd, hist in sorted(scored, key=lambda t: -t[0]):
        if fd not in mapping and hist not in used:
            mapping[fd] = hist
            used.add(hist)
    return mapping


def suggestions(fd_name: str, hist_names: List[str], n: int = 3) -> List[Tuple[str, float]]:
    ranked = sorted(((h, round(name_score(fd_name, h), 2)) for h in hist_names), key=lambda t: -t[1])
    return ranked[:n]


def _match_competition(hist: pd.DataFrame, positions: np.ndarray, fd: pd.DataFrame) -> Tuple[Dict[int, Tuple[float, float, float]], Dict[str, Any]]:
    hist_home = hist["home_raw"].to_numpy()[positions]
    hist_away = hist["away_raw"].to_numpy()[positions]
    hist_day = pd.to_datetime(hist["date_parsed"], utc=True).dt.tz_localize(None).dt.normalize().to_numpy()[positions]
    hist_hg = pd.to_numeric(hist["goals_home"], errors="coerce").to_numpy()[positions]
    hist_ag = pd.to_numeric(hist["goals_away"], errors="coerce").to_numpy()[positions]
    hist_names = sorted(set(hist_home) | set(hist_away))
    fd_names = sorted(set(fd["home"]) | set(fd["away"]))
    mapping = build_name_map(fd_names, hist_names)

    by_day: Dict[Any, List[int]] = {}
    for i, day in enumerate(hist_day):
        by_day.setdefault(day, []).append(i)
    tol = [pd.Timedelta(days=k).to_timedelta64() for k in range(-DATE_TOLERANCE_DAYS, DATE_TOLERANCE_DAYS + 1)]

    def candidates(day) -> List[int]:
        out: List[int] = []
        for delta in tol:
            out.extend(by_day.get(day + delta, []))
        return out

    used_hist_names = set(mapping.values())
    for row in fd.itertuples(index=False):
        if np.isnan(row.hg) or np.isnan(row.ag):
            continue
        need_home, need_away = row.home not in mapping, row.away not in mapping
        if not (need_home or need_away):
            continue
        cands = [i for i in candidates(np.datetime64(row.date)) if hist_hg[i] == row.hg and hist_ag[i] == row.ag]
        if not need_home:
            cands = [i for i in cands if hist_home[i] == mapping[row.home]]
        if not need_away:
            cands = [i for i in cands if hist_away[i] == mapping[row.away]]
        if len(cands) != 1:
            continue
        i = cands[0]
        if need_home and hist_home[i] not in used_hist_names:
            mapping[row.home] = hist_home[i]
            used_hist_names.add(hist_home[i])
        if need_away and hist_away[i] not in used_hist_names:
            mapping[row.away] = hist_away[i]
            used_hist_names.add(hist_away[i])

    index: Dict[Tuple[str, str], List[int]] = {}
    for i in range(len(positions)):
        index.setdefault((hist_home[i], hist_away[i]), []).append(i)
    matched: Dict[int, Tuple[float, float, float]] = {}
    n_matched_rows = 0
    for row in fd.itertuples(index=False):
        if row.home not in mapping or row.away not in mapping:
            continue
        day = np.datetime64(row.date)
        for i in index.get((mapping[row.home], mapping[row.away]), []):
            if abs((hist_day[i] - day) / np.timedelta64(1, "D")) <= DATE_TOLERANCE_DAYS and int(positions[i]) not in matched:
                matched[int(positions[i])] = (float(row.h), float(row.d), float(row.a))
                n_matched_rows += 1
                break
    unmatched = sorted(n for n in fd_names if n not in mapping)
    report = {
        "fd_rows": int(len(fd)), "fd_matched": n_matched_rows, "hist_rows": int(len(positions)),
        "hist_matched": len(matched), "unmatched_teams": [(n, suggestions(n, hist_names)) for n in unmatched],
    }
    return matched, report


def attach_odds(df: pd.DataFrame, cache_dir: Path, now: Optional[pd.Timestamp] = None,
                getter: Optional[Callable[[str], Optional[bytes]]] = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    if df is None or df.empty:
        raise OddsSourceError("historique vide")
    dates = pd.to_datetime(df["date_parsed"], utc=True).dt.tz_localize(None)
    now = now if now is not None else pd.Timestamp.now(tz="UTC").tz_localize(None)
    fetcher = Fetcher(cache_dir, season_year(now), getter=getter)
    comps = sorted(df["competition_code"].dropna().astype(str).unique())
    wanted = [c for c in comps if c in MMZ_CODES or c in NEW_CODES]
    skipped = [c for c in comps if c in UNAVAILABLE]
    sources = load_sources(wanted, dates.min(), dates.max(), fetcher)
    if fetcher.downloaded + fetcher.from_cache == 0:
        raise OddsSourceError("aucun fichier football-data disponible (réseau ou cache)")

    odds_col: List[Optional[Dict[str, float]]] = [None] * len(df)
    comp_values = df["competition_code"].astype(str).to_numpy()
    per_comp: Dict[str, Any] = {}
    all_matched: Dict[int, Tuple[float, float, float]] = {}
    for comp in wanted:
        fd = sources.get(comp)
        positions = np.where(comp_values == comp)[0]
        if fd is None or fd.empty or len(positions) == 0:
            per_comp[comp] = {"fd_rows": 0 if fd is None else int(len(fd)), "fd_matched": 0, "hist_rows": int(len(positions)), "hist_matched": 0, "unmatched_teams": []}
            continue
        matched, rep = _match_competition(df, positions, fd)
        per_comp[comp] = rep
        all_matched.update(matched)
    for pos, (h, d, a) in all_matched.items():
        odds_col[pos] = {"home": h, "draw": d, "away": a}

    out = df.copy()
    out["odds_markets"] = odds_col
    covered_rows = int(np.isin(comp_values, wanted).sum())
    arr = np.array(list(all_matched.values()), dtype=float).reshape(-1, 3)
    report: Dict[str, Any] = {
        "rows": int(len(df)), "matched": len(all_matched),
        "coverage_all": len(all_matched) / len(df),
        "coverage_available_leagues": len(all_matched) / covered_rows if covered_rows else 0.0,
        "rows_unavailable_leagues": int(np.isin(comp_values, skipped).sum()),
        "downloaded": fetcher.downloaded, "from_cache": fetcher.from_cache, "missing_urls": fetcher.missing,
        "per_comp": per_comp,
    }
    if len(arr) == 0:
        raise OddsSourceError("aucune cote rapprochée avec l'historique : vérifier le mapping des équipes")
    inv = 1.0 / arr
    overround = inv.sum(axis=1)
    p_home = inv[:, 0] / overround
    positions_matched = np.array(list(all_matched.keys()))
    hg = pd.to_numeric(df["goals_home"], errors="coerce").to_numpy()[positions_matched]
    ag = pd.to_numeric(df["goals_away"], errors="coerce").to_numpy()[positions_matched]
    home_win, away_win = hg > ag, ag > hg
    signal = float(p_home[home_win].mean() - p_home[away_win].mean()) if home_win.any() and away_win.any() else float("nan")
    report["mean_overround"] = float(overround.mean())
    report["home_signal"] = signal
    low, high = OVERROUND_RANGE
    if not (low <= report["mean_overround"] <= high):
        raise OddsSourceError(f"overround moyen {report['mean_overround']:.3f} hors de [{low}, {high}] : parsing suspect (colonnes ou inversion)")
    if not np.isfinite(signal) or signal < MIN_HOME_SIGNAL:
        raise OddsSourceError(f"cotes incohérentes avec les résultats (signal domicile {signal:.3f}) : inversion domicile/extérieur probable")
    return out, report


def format_report(report: Dict[str, Any]) -> List[str]:
    lines = [
        f"💰 football-data : {report['matched']}/{report['rows']} matchs avec cotes "
        f"({report['coverage_all'] * 100:.1f}% de l'historique, {report['coverage_available_leagues'] * 100:.1f}% des ligues disponibles)",
        f"   overround moyen {report['mean_overround']:.3f} | signal domicile {report['home_signal']:+.3f} | "
        f"fichiers : {report['downloaded']} téléchargés, {report['from_cache']} en cache, {len(report['missing_urls'])} absents",
        f"   matchs de ligues indisponibles (UCL, UEL, Saudi, Libertadores, Sudamericana) : {report['rows_unavailable_leagues']}",
    ]
    for comp, rep in sorted(report["per_comp"].items(), key=lambda kv: kv[1]["hist_rows"] - kv[1]["hist_matched"], reverse=True)[:6]:
        if rep["hist_rows"]:
            lines.append(f"   ligue {comp}: {rep['hist_matched']}/{rep['hist_rows']} rapprochés, {len(rep['unmatched_teams'])} équipes FD sans correspondance")
    return lines


def write_unmatched(report: Dict[str, Any], path: Path) -> int:
    rows = []
    for comp, rep in sorted(report["per_comp"].items()):
        for name, cands in rep["unmatched_teams"]:
            rows.append(f"{comp}\t{name}\t" + " | ".join(f"{h} ({s})" for h, s in cands))
    Path(path).write_text("\n".join(rows), encoding="utf-8")
    return len(rows)
