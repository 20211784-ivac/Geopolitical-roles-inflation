#!/usr/bin/env python3
"""Rebuild the UCDP conflict-onset stacked event study from official v4-2024 data.

The script does NOT invent event dates or country assignments. It downloads the
public UCDP Conflict Termination Dataset v4-2024 (conflict-level CSV), audits
its episode structure, maps UCDP location names to the countries in the supplied
World Bank CPI panel, builds episode-specific treated/control stacks, estimates
stacked event-study regressions, and produces episode-level sensitivity results.

Primary treatment definition: countries in UCDP `gwno_loc` / `location` (the
location of incompatibility; both primary countries for interstate conflicts,
one location country for internalized conflicts). Secondary troop contributors
are not silently added to the primary treated group. The pipeline runs a separate
sensitivity specification that adds secondary states recorded at the threshold-
crossing year to the treated group.

All coefficients are conditional event-time associations, not automatically
causal effects. See README_UCDP_PIPELINE.md before interpreting output.
"""
from __future__ import annotations
import argparse
import hashlib
import itertools
import json
import math
import re
import sys
import unicodedata
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats as st
from scipy.sparse import csr_matrix
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
DEFAULT_PANEL = HERE.parent / "source" / "PAPER GEOPOLITICA_.dta"
DEFAULT_UCDP = HERE / "data" / "UCDPConflictTerminationDataset_v4_2024_Conflict.csv"
OUT = HERE / "output"
OUT.mkdir(exist_ok=True, parents=True)
UCDP_URL = "https://ucdp.uu.se/downloads/monadterm/UCDPConflictTerminationDataset_v4_2024_Conflict.csv"
CODEBOOK_URL = "https://ucdp.uu.se/downloads/monadterm/UCDPConflictTerminationDataset_v4_2024_Codebook.pdf"
DUPLICATE_CPI_COUNTRIES = {"USA", "GBR", "GAB"}
BASELINE_UNIVERSE_EXCLUSIONS = {"XKX", "WSM"}
EXPECTED_REPLICATION = {"episodes": 22, "treated_country_events": 24, "stack_observations": 68350}


def normalize(s: object) -> str:
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return ""
    x = unicodedata.normalize("NFKD", str(s))
    x = "".join(ch for ch in x if not unicodedata.combining(ch))
    x = x.lower().strip()
    x = x.replace("&", " and ")
    x = re.sub(r"[^a-z0-9]+", " ", x)
    return re.sub(r"\s+", " ", x).strip()


# UCDP/World Bank country-name differences likely to occur in this panel.
ALIASES = {
    "iran": "IRN", "iran islamic republic of": "IRN", "iran islamic rep": "IRN",
    "venezuela": "VEN", "venezuela bolivarian republic of": "VEN", "venezuela rb": "VEN",
    "egypt": "EGY", "egypt arab republic of": "EGY", "egypt arab rep": "EGY",
    "syria": "SYR", "syrian arab republic": "SYR",
    "russia": "RUS", "russian federation": "RUS",
    "south korea": "KOR", "republic of korea": "KOR", "korea republic of": "KOR",
    "north korea": "PRK", "democratic peoples republic of korea": "PRK",
    "korea democratic peoples republic of": "PRK",
    "laos": "LAO", "lao peoples democratic republic": "LAO",
    "moldova": "MDA", "republic of moldova": "MDA",
    "brunei": "BRN", "brunei darussalam": "BRN",
    "cape verde": "CPV", "cabo verde": "CPV",
    "ivory coast": "CIV", "cote d ivoire": "CIV", "cote divoire": "CIV",
    "tanzania": "TZA", "united republic of tanzania": "TZA",
    "bolivia": "BOL", "bolivia plurinational state of": "BOL",
    "micronesia": "FSM", "federated states of micronesia": "FSM",
    "bosnia herzegovina": "BIH", "bosnia and herzegovina": "BIH", "bosnia": "BIH",
    "macedonia": "MKD", "north macedonia": "MKD", "republic of north macedonia": "MKD",
    "czech republic": "CZE", "czechia": "CZE",
    "swaziland": "SWZ", "eswatini": "SWZ",
    "burma": "MMR", "myanmar": "MMR",
    "east timor": "TLS", "timor leste": "TLS",
    "congo brazzaville": "COG", "republic of the congo": "COG", "congo republic": "COG",
    "congo kinshasa": "COD", "democratic republic of the congo": "COD",
    "dr congo": "COD", "drc": "COD", "congo democratic republic": "COD",
    "united states": "USA", "united states of america": "USA", "usa": "USA",
    "united kingdom": "GBR", "great britain": "GBR", "uk": "GBR",
    "turkey": "TUR", "turkiye": "TUR",
    "palestine": "PSE", "palestinian territories": "PSE", "west bank": "PSE",
    "israel": "ISR", "china": "CHN", "india": "IND", "pakistan": "PAK",
    "afghanistan": "AFG", "iran islamic republic": "IRN",
    "azerbaijan": "AZE", "armenia": "ARM", "georgia": "GEO",
    "niger": "NER", "cambodia": "KHM", "kampuchea": "KHM", "cambodia kampuchea": "KHM", "thailand": "THA",
    "nigeria": "NGA", "cameroon": "CMR", "ethiopia": "ETH", "mali": "MLI",
    "rwanda": "RWA", "burkina faso": "BFA", "uganda": "UGA", "eritrea": "ERI",
    "kyrgyzstan": "KGZ", "tajikistan": "TJK", "sudan": "SDN", "south sudan": "SSD",
    "somalia": "SOM", "kenya": "KEN", "libya": "LBY", "algeria": "DZA",
    "senegal": "SEN", "chad": "TCD", "central african republic": "CAF",
    "democratic republic of congo": "COD", "republic of congo": "COG",
    "yemen north yemen": "YEM", "north yemen": "YEM", "south yemen": "YEM",
    "russia soviet union": "RUS", "serbia yugoslavia": "SRB",
    "dr congo zaire": "COD", "democratic republic of congo zaire": "COD",
    "congo": "COG", "congo zaire": "COD",
    "cambodia kampuchea": "KHM", "kampuchea": "KHM",
    "cote d ivoire": "CIV", "ivory coast": "CIV",
    "burkina faso": "BFA", "gambia": "GMB", "the gambia": "GMB",
    "macedonia fyrom": "MKD", "serbia montenegro": "SRB",
    "yugoslavia": "SRB", "vietnam north vietnam": "VNM", "south vietnam": "VNM",
    "zimbabwe rhodesia": "ZWE", "yemen north yemen": "YEM", "peru": "PER",
}


def load_country_maps(panel: pd.DataFrame, overrides_path: Path | None = None):
    names = panel[["iso3", "country"]].dropna().drop_duplicates()
    iso3_set = set(names.iso3.astype(str))
    exact = {}
    for r in names.itertuples(index=False):
        exact[normalize(r.country)] = str(r.iso3)
        exact[normalize(r.iso3)] = str(r.iso3)
    aliases = {normalize(k): v for k, v in ALIASES.items()}
    if overrides_path is not None and overrides_path.exists():
        ov = pd.read_csv(overrides_path)
        if not {"ucdp_location_token", "iso3"}.issubset(ov.columns):
            raise ValueError("Override CSV must contain ucdp_location_token,iso3 columns")
        for r in ov.dropna(subset=["ucdp_location_token", "iso3"]).itertuples(index=False):
            aliases[normalize(r.ucdp_location_token)] = str(r.iso3).upper()
    # Optional pycountry fallback. Exact aliases and panel names take precedence.
    try:
        import pycountry  # type: ignore
    except Exception:
        pycountry = None

    def map_one(token: str):
        n = normalize(token)
        if n.startswith("government of "):
            n = n[len("government of "):].strip()
        if n in exact:
            return exact[n], 1.0
        if n in aliases:
            return aliases[n], 0.98
        if pycountry is not None and n:
            try:
                matches = pycountry.countries.search_fuzzy(n)
                cands = [m.alpha_3 for m in matches if getattr(m, "alpha_3", None)]
                cands = list(dict.fromkeys(cands))
                # Accept only a unique country; never guess between two Koreas/Congos.
                if len(cands) == 1:
                    return cands[0], 0.90
            except Exception:
                pass
        return None, 0.0

    return map_one, iso3_set


def gwno_count(value: object) -> int:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return 0
    if isinstance(value, (int, np.integer)):
        return 1
    if isinstance(value, float) and value.is_integer():
        return 1
    s = str(value).strip()
    if not s or s.lower() in {"nan", "none", "."}:
        return 0
    return len(re.findall(r"\d+", s))


def parse_location(location: object, gwno_loc: object, map_one):
    """Map UCDP location string to ordered ISO3s, using gwno_loc count to split.

    For multi-country locations, test all plausible separator segmentations and
    accept only an unambiguous best mapping. Country names containing commas or
    hyphens are therefore not blindly split.
    """
    if location is None or (isinstance(location, float) and np.isnan(location)):
        return [], ["<missing location>"]
    s = str(location).strip()
    k = gwno_count(gwno_loc)
    if k <= 1:
        iso, _ = map_one(s)
        if iso:
            return [iso], []
        # If GWNO is absent/incomplete, accept a multi-country string only
        # when a simple split yields an unambiguous full set of country matches.
        for pattern in [r"\s*;\s*", r"\s*\|\s*", r"\s*,\s*", r"\s*-\s*"]:
            parts = [p.strip() for p in re.split(pattern, s) if p.strip()]
            if len(parts) > 1:
                mapped = [map_one(p) for p in parts]
                if all(x[0] for x in mapped):
                    return [x[0] for x in mapped], []
        return [], [s]

    sep_positions = [i for i, ch in enumerate(s) if ch in "-,;"]
    candidates = []
    if len(sep_positions) >= k - 1 and len(sep_positions) <= 14:
        for cuts in itertools.combinations(sep_positions, k - 1):
            parts = []
            start = 0
            for c in cuts:
                parts.append(s[start:c].strip())
                start = c + 1
            parts.append(s[start:].strip())
            mapped = [map_one(p) for p in parts]
            if all(x[0] for x in mapped):
                score = sum(x[1] for x in mapped)
                codes = tuple(x[0] for x in mapped)
                candidates.append((score, codes, parts))
    if candidates:
        candidates.sort(key=lambda x: x[0], reverse=True)
        best = candidates[0]
        # If multiple top-scoring parses produce different country lists, flag ambiguity.
        tied = [c for c in candidates if abs(c[0] - best[0]) < 1e-9]
        if len({c[1] for c in tied}) == 1:
            return list(best[1]), []
        return [], [s + " <ambiguous split>"]

    # Last fallback: try simple separators, but only if count matches exactly.
    for pattern in [r"\s*-\s*", r"\s*,\s*", r"\s*;\s*"]:
        parts = [p.strip() for p in re.split(pattern, s) if p.strip()]
        if len(parts) == k:
            mapped = [map_one(p) for p in parts]
            if all(x[0] for x in mapped):
                return [x[0] for x in mapped], []
    return [], [s]


def download_ucdp(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading official UCDP conflict-level CSV: {UCDP_URL}")
    req = urllib.request.Request(UCDP_URL, headers={"User-Agent": "Mozilla/5.0 research-replication/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=90) as response:
            raw = response.read()
    except Exception as e:
        raise RuntimeError(
            "The UCDP CSV could not be downloaded from this environment. "
            "Download the conflict-level CSV manually from https://ucdp.uu.se/downloads/ "
            "(UCDP Conflict Termination Dataset version 4-2024 → conflict level → CSV) "
            f"and save it as {path}. Details: {e}"
        )
    if len(raw) < 1000:
        raise RuntimeError("Downloaded UCDP file is unexpectedly small; refusing to continue.")
    path.write_bytes(raw)
    print("UCDP SHA256:", hashlib.sha256(raw).hexdigest())


def read_csv_flexible(path: Path) -> pd.DataFrame:
    try:
        d = pd.read_csv(path, low_memory=False, encoding="utf-8-sig")
        if len(d.columns) == 1:
            d = pd.read_csv(path, low_memory=False, encoding="utf-8-sig", sep=";")
        return d
    except UnicodeDecodeError:
        d = pd.read_csv(path, low_memory=False, encoding="latin-1")
        if len(d.columns) == 1:
            d = pd.read_csv(path, low_memory=False, encoding="latin-1", sep=";")
        return d


def canonicalize_ucdp_columns(d: pd.DataFrame) -> pd.DataFrame:
    normcols = {re.sub(r"[^a-z0-9]", "", str(c).lower()): c for c in d.columns}
    required = {
        "c_epid": ["cepid"], "year": ["year"], "type_of_conflict": ["typeofconflict"],
        "start_date2": ["startdate2"], "start_prec2": ["startprec2"],
        "start_date": ["startdate"], "start_prec": ["startprec"],
        "location": ["location"], "gwno_loc": ["gwnoloc"],
    }
    rename = {}
    for target, aliases in required.items():
        found = next((normcols[a] for a in aliases if a in normcols), None)
        if found is None:
            raise ValueError(f"UCDP file lacks required column {target}. Available: {list(d.columns)}")
        rename[found] = target
    # Optional secondary participant fields.
    for target in ["side_a_2nd", "side_b_2nd", "gwno_a_2nd", "gwno_b_2nd", "c_ep_startyear", "c_ep_endyear"]:
        key = re.sub(r"[^a-z0-9]", "", target.lower())
        if key in normcols:
            rename[normcols[key]] = target
    return d.rename(columns=rename)


def parse_secondary_participants(value: object, map_one, panel_iso3: set[str]):
    """Parse comma-separated UCDP secondary troop contributors.

    The GWNO secondary-code field can be non-aligned with the textual list
    (for example, because historical or non-panel entities are represented).
    Parse each named state independently and retain only countries present in
    the inflation panel. Unknown names are reported for audit, not silently
    assigned to a panel country.
    """
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return [], []
    s = str(value).strip()
    if not s or s.lower() in {"nan", "none", "."}:
        return [], []
    tokens = [t.strip() for t in s.split(",") if t.strip()]
    mapped, bad = [], []
    for token in tokens:
        iso, _ = map_one(token)
        if iso is None:
            bad.append(token)
        elif iso in panel_iso3:
            mapped.append(iso)
    return sorted(set(mapped)), bad


def build_episode_metadata(ucdp: pd.DataFrame, map_one, panel_iso3: set[str]):
    d = canonicalize_ucdp_columns(ucdp)
    d["c_epid"] = d["c_epid"].astype(str).str.replace(r"\.0$", "", regex=True)
    d["year"] = pd.to_numeric(d["year"], errors="coerce")
    d["type_of_conflict"] = pd.to_numeric(d["type_of_conflict"], errors="coerce")
    d["start_prec2"] = pd.to_numeric(d["start_prec2"], errors="coerce")
    d["start_date2_parsed"] = pd.to_datetime(d["start_date2"], errors="coerce")
    d["start_date_parsed"] = pd.to_datetime(d["start_date"], errors="coerce")
    d["start_prec"] = pd.to_numeric(d["start_prec"], errors="coerce")

    records = []
    unmapped = []
    for epid, g in d.groupby("c_epid", sort=False):
        gd = g[g.start_date2_parsed.notna()].copy()
        date_vals = sorted(gd.start_date2_parsed.dropna().dt.normalize().unique())
        event_date = pd.Timestamp(date_vals[0]) if date_vals else pd.NaT
        date_conflict = len(date_vals) > 1
        precision_vals = sorted(pd.to_numeric(gd.start_prec2, errors="coerce").dropna().unique().tolist())
        precision_conflict = len(precision_vals) > 1
        gfirst = g[g.start_date_parsed.notna()].copy()
        first_date_vals = sorted(gfirst.start_date_parsed.dropna().dt.normalize().unique())
        first_battle_date = pd.Timestamp(first_date_vals[0]) if first_date_vals else pd.NaT
        first_date_conflict = len(first_date_vals) > 1
        first_precision_vals = sorted(pd.to_numeric(gfirst.start_prec, errors="coerce").dropna().unique().tolist())
        first_precision_conflict = len(first_precision_vals) > 1
        if len(gd):
            if pd.notna(event_date):
                onset_year = event_date.year
                onset_rows = gd[gd.year == onset_year]
                r = onset_rows.iloc[0] if len(onset_rows) else gd.iloc[0]
            else:
                r = gd.iloc[0]
        else:
            r = g.iloc[0]
        typ = int(r.type_of_conflict) if pd.notna(r.type_of_conflict) else np.nan
        loc = r.location if "location" in r.index else ""
        gwno = r.gwno_loc if "gwno_loc" in r.index else np.nan
        # Event treatment is coded from the onset-year row. For control screening,
        # however, flag an episode if it is type 2 or 4 in ANY conflict-year row.
        active_type24 = bool(g.type_of_conflict.isin([2, 4]).any())
        if typ in (2, 4):
            loc_codes, bad = parse_location(loc, gwno, map_one)
            if bad:
                unmapped.append({"c_epid": epid, "location": loc, "gwno_loc": gwno, "unmapped": "; ".join(bad), "type": typ})
        else:
            loc_codes, bad = [], []
        active_loc_codes = set()
        active_secondary_codes = set()
        secondary_onset_codes = set()
        if active_type24:
            g24 = g[g.type_of_conflict.isin([2, 4])]
            for ar in g24.itertuples(index=False):
                acodes, abad = parse_location(getattr(ar, "location", ""), getattr(ar, "gwno_loc", np.nan), map_one)
                active_loc_codes.update(acodes)
                if abad:
                    unmapped.append({"c_epid": epid, "location": getattr(ar, "location", ""),
                                     "gwno_loc": getattr(ar, "gwno_loc", np.nan),
                                     "unmapped": "; ".join(abad), "type": getattr(ar, "type_of_conflict", np.nan)})
                # Secondary states that provide troop support are not treated as
                # primary location countries, but are excluded from control pools.
                for side_col, gwno_col in [("side_a_2nd", "gwno_a_2nd"), ("side_b_2nd", "gwno_b_2nd")]:
                    side_val = getattr(ar, side_col, "")
                    if side_val is None or (isinstance(side_val, float) and np.isnan(side_val)) or str(side_val).strip() in {"", ".", "nan", "None"}:
                        continue
                    sec_codes, sec_bad = parse_secondary_participants(side_val, map_one, panel_iso3)
                    active_secondary_codes.update(sec_codes)
                    if sec_bad:
                        unmapped.append({"c_epid": epid, "location": str(side_val),
                                         "gwno_loc": getattr(ar, gwno_col, np.nan),
                                         "unmapped": "secondary participant: " + "; ".join(sec_bad),
                                         "type": getattr(ar, "type_of_conflict", np.nan)})
        # Secondary troop contributors recorded in the threshold-crossing year.
        if pd.notna(event_date) and len(gd):
            onset_rows = gd[gd.year == event_date.year]
            r_onset = onset_rows.iloc[0] if len(onset_rows) else r
        else:
            r_onset = r
        for side_col, gwno_col in [("side_a_2nd", "gwno_a_2nd"), ("side_b_2nd", "gwno_b_2nd")]:
            side_val = r_onset.get(side_col, "") if hasattr(r_onset, "get") else ""
            if side_val is None or (isinstance(side_val, float) and np.isnan(side_val)) or str(side_val).strip() in {"", ".", "nan", "None"}:
                continue
            sec_codes, sec_bad = parse_secondary_participants(side_val, map_one, panel_iso3)
            secondary_onset_codes.update(sec_codes)
            if sec_bad:
                unmapped.append({"c_epid": epid, "location": str(side_val), "gwno_loc": r_onset.get(gwno_col, np.nan),
                                 "unmapped": "secondary participant at onset: " + "; ".join(sec_bad), "type": typ})
        # Only years in which the episode is coded as type 2 or 4 are used for the main control screen.
        g24 = g[g.type_of_conflict.isin([2, 4])]
        active_years = sorted(set(pd.to_numeric(g24.year, errors="coerce").dropna().astype(int).tolist()))
        # State supporters are kept separately from the primary location countries.
        side_a_2nd = str(r.get("side_a_2nd", "")) if "side_a_2nd" in r.index else ""
        side_b_2nd = str(r.get("side_b_2nd", "")) if "side_b_2nd" in r.index else ""
        records.append({
            "episode_id": epid, "conflict_id": int(r.conflict_id), "c_epno": int(r.c_epno),
            "c_ep_startyear": int(r.c_ep_startyear) if pd.notna(r.c_ep_startyear) else np.nan,
            "event_date": event_date, "start_precision": r.start_prec2,
            "first_battle_date": first_battle_date,
            "first_battle_precision": (r.start_prec if pd.notna(r.start_prec) else (gfirst.start_prec.dropna().iloc[0] if len(gfirst.start_prec.dropna()) else np.nan)),
            "date_conflict": date_conflict, "precision_conflict": precision_conflict,
            "first_date_conflict": first_date_conflict, "first_precision_conflict": first_precision_conflict,
            "type_of_conflict": typ, "location": str(loc), "gwno_loc": str(gwno),
            "location_iso3_all": sorted(set(loc_codes)),
            "location_iso3_in_panel": sorted(set(x for x in loc_codes if x in panel_iso3)),
            "active_type24": active_type24,
            "active_location_iso3_in_panel": sorted(set(x for x in active_loc_codes if x in panel_iso3)),
            "secondary_iso3_in_panel": sorted(set(x for x in active_secondary_codes if x in panel_iso3)),
            "secondary_onset_iso3_in_panel": sorted(set(x for x in secondary_onset_codes if x in panel_iso3)),
            "active_country_iso3_in_panel": sorted(set(x for x in (active_loc_codes | active_secondary_codes) if x in panel_iso3)),
            "active_years": active_years, "date_conflict": date_conflict,
            "precision_conflict": precision_conflict,
            "side_a_2nd": side_a_2nd, "side_b_2nd": side_b_2nd,
        })
    episodes = pd.DataFrame(records)
    if unmapped:
        pd.DataFrame(unmapped).to_csv(OUT / "ucdp_unmapped_locations.csv", index=False)
    return d, episodes, unmapped


def panel_prepare(panel_path: Path):
    p = pd.read_stata(panel_path, convert_categoricals=False)
    needed = {"iso3", "country", "mdate", "hcpi"}
    miss = needed - set(p.columns)
    if miss:
        raise ValueError(f"Panel lacks columns: {sorted(miss)}")
    p["iso3"] = p.iso3.astype(str)
    p["mdate"] = pd.to_numeric(p.mdate, errors="coerce").astype("Int64")
    p = p.dropna(subset=["iso3", "mdate"]).copy()
    p["mdate"] = p.mdate.astype(int)
    p = p.sort_values(["iso3", "mdate"]).reset_index(drop=True)
    if p.duplicated(["iso3", "mdate"]).any():
        raise ValueError("Panel has duplicate iso3-month rows.")
    p["hcpi"] = pd.to_numeric(p.hcpi, errors="coerce")
    p["pi_m"] = 100 * np.log(p.hcpi / p.groupby("iso3").hcpi.shift(1))
    # Replication universe: original 144-country base (XKX and WSM removed),
    # then exclude the four audited exact-duplicate CPI trajectories in the UCDP pool.
    p = p.loc[~p.iso3.isin(BASELINE_UNIVERSE_EXCLUSIONS | DUPLICATE_CPI_COUNTRIES)].copy()
    return p


def overlap_years(window_months: set[int]) -> set[int]:
    return {int((m + 23520) // 12) for m in window_months}  # Stata mdate -> calendar year


def build_stacks(panel: pd.DataFrame, episodes: pd.DataFrame, all_episodes: pd.DataFrame,
                 date_source="threshold", treat_secondary_at_onset=False):
    # Select precisely defined onset episodes in the requested period and types.
    selected = episodes.copy()
    if date_source == "threshold":
        selected["event_date_used"] = selected.event_date
        selected["precision_used"] = selected.start_precision
        selected["date_conflict_used"] = selected.date_conflict
        selected["precision_conflict_used"] = selected.precision_conflict
    elif date_source == "first_conflict_fatality":
        # UCDP start_date is the first battle-related death in the conflict,
        # not a new first-death date for each later conflict episode.
        # It is therefore a valid timing sensitivity only for c_epno == 1.
        selected = selected.loc[pd.to_numeric(selected.c_epno, errors="coerce").eq(1)].copy()
        selected["event_date_used"] = selected.first_battle_date
        selected["precision_used"] = selected.first_battle_precision
        selected["date_conflict_used"] = selected.first_date_conflict
        selected["precision_conflict_used"] = selected.first_precision_conflict
    else:
        raise ValueError("date_source must be 'threshold' or 'first_conflict_fatality'")
    selected = selected.loc[
        selected.event_date_used.notna()
        & selected.precision_used.notna()
        & (selected.precision_used <= 3)
        & selected.type_of_conflict.isin([2, 4])
        & selected.event_date_used.dt.year.between(2000, 2023)
    ].copy()
    selected = selected.sort_values(["event_date_used", "episode_id"]).reset_index(drop=True)
    conflicting = selected.loc[selected.date_conflict_used | selected.precision_conflict_used]
    selected_clean = selected.loc[~(selected.date_conflict_used | selected.precision_conflict_used)].copy()

    # Active conflict country-years for ALL type-2/type-4 episodes, including
    # episodes that began before 2000; these are used only to screen controls.
    active_map = {}
    active_episode_rows = []
    for r in all_episodes.itertuples(index=False):
        if not r.active_type24:
            continue
        for iso in r.active_country_iso3_in_panel:
            for yr in r.active_years:
                active_map.setdefault(iso, []).append((str(r.episode_id), int(yr)))
                active_episode_rows.append((str(r.episode_id), iso, int(yr)))

    panel_by_iso = {iso: g.set_index("mdate").sort_index() for iso, g in panel.groupby("iso3", sort=False)}
    all_iso = sorted(panel.iso3.unique())
    stack_rows, audit_rows = [], []
    dropped_rows = []
    for r in conflicting.itertuples(index=False):
        reason = []
        if r.date_conflict_used: reason.append(f"inconsistent {date_source} date across conflict-year rows")
        if r.precision_conflict_used: reason.append(f"inconsistent {date_source} precision across conflict-year rows")
        dropped_rows.append({"episode_id": str(r.episode_id), "reason": "; ".join(reason), "location": r.location})

    for er in selected_clean.itertuples(index=False):
        event_date_used = er.event_date_used
        event_m = int(event_date_used.year * 12 + event_date_used.month - 1 - 23520)
        win = list(range(event_m - 12, event_m + 13))
        win_set = set(win)
        win_years = overlap_years(win_set)
        treated_all = list(er.location_iso3_in_panel)
        secondary_all = list(er.secondary_iso3_in_panel)
        if treat_secondary_at_onset:
            treated_all = sorted(set(treated_all).union(set(er.secondary_onset_iso3_in_panel)))
        if not treated_all:
            dropped_rows.append({"episode_id": er.episode_id, "reason": "No UCDP location country in CPI panel", "location": er.location})
            continue

        # Full balanced 25-month CPI path is required for every country in a stack.
        complete = []
        for iso in all_iso:
            g = panel_by_iso[iso]
            if all(m in g.index and pd.notna(g.loc[m, "pi_m"]) for m in win):
                complete.append(iso)
        treated_complete = sorted(set(treated_all).intersection(complete))
        treated_missing = sorted(set(treated_all) - set(treated_complete))
        if not treated_complete:
            dropped_rows.append({"episode_id": er.episode_id, "reason": "Treated countries lack complete 25-month CPI path", "location": er.location})
            continue

        # Exclude controls with another UCDP type-2/type-4 conflict episode active
        # in any calendar year touched by this event window.
        controls = []
        for iso in complete:
            if iso in treated_complete or iso in secondary_all:
                continue
            other_active = any((other_id != str(er.episode_id) and yr in win_years)
                               for other_id, yr in active_map.get(iso, []))
            if not other_active:
                controls.append(iso)
        if len(controls) < 5:
            dropped_rows.append({"episode_id": er.episode_id, "reason": f"Fewer than 5 eligible controls ({len(controls)})", "location": er.location})
            continue

        event_id_num = len(audit_rows) + 1
        for iso in treated_complete + controls:
            g = panel_by_iso[iso].loc[win].reset_index()
            is_treated = int(iso in treated_complete)
            for row in g.itertuples(index=False):
                m = int(row.mdate)
                stack_rows.append({
                    "event_id_num": event_id_num, "episode_id": str(er.episode_id),
                    "iso3": iso, "mdate": m, "h": m - event_m,
                    "treated": is_treated, "pi_m": float(row.pi_m),
                    "hcpi": float(row.hcpi), "type_of_conflict": int(er.type_of_conflict),
                    "event_date": event_date_used.strftime("%Y-%m-%d"),
                    "stack_country": f"{event_id_num}|{iso}",
                    "stack_month": f"{event_id_num}|{m}",
                })
        audit_rows.append({
            "event_id_num": event_id_num, "episode_id": str(er.episode_id),
            "conflict_id": int(er.conflict_id), "c_epno": int(er.c_epno),
            "event_date": event_date_used.strftime("%Y-%m-%d"),
            "event_month": event_m, "type_of_conflict": int(er.type_of_conflict),
            "location": er.location, "treated_iso3_all": ";".join(treated_all),
            "treated_iso3_used": ";".join(treated_complete),
            "secondary_participants_excluded_from_controls": ";".join(secondary_all),
            "date_source": date_source,
            "treatment_definition": "primary_plus_secondary_at_onset" if treat_secondary_at_onset else "primary_location_only",
            "treated_missing_cpi": ";".join(treated_missing),
            "n_treated": len(treated_complete), "n_secondary_excluded": len(secondary_all), "n_controls": len(controls),
            "n_stack_obs": (len(treated_complete) + len(controls)) * 25,
            "active_years": ";".join(map(str, er.active_years)),
        })

    stacks = pd.DataFrame(stack_rows)
    audit = pd.DataFrame(audit_rows)
    dropped = pd.DataFrame(dropped_rows)
    if len(stacks):
        stacks["stack_country_id"] = pd.factorize(stacks.stack_country, sort=True)[0] + 1
        stacks["stack_month_id"] = pd.factorize(stacks.stack_month, sort=True)[0] + 1
        stacks["cid_cluster"] = pd.factorize(stacks.iso3, sort=True)[0] + 1
    return stacks, audit, dropped, selected


def absorb_two_way(arr, group1, group2, trend=False, time_values=None, tol=1e-10, maxiter=500):
    A = np.asarray(arr, dtype=np.float64).copy()
    if A.ndim == 1:
        A = A[:, None]
    n = len(A)
    mats = []
    for g in [group1, group2]:
        code, _ = pd.factorize(np.asarray(g), sort=False)
        if (code < 0).any():
            raise ValueError("Missing fixed-effect group code")
        G = int(code.max() + 1)
        mem = csr_matrix((np.ones(n), (code, np.arange(n))), shape=(G, n))
        counts = np.asarray(mem.sum(axis=1)).ravel()
        mats.append((code, mem, counts))
    if trend:
        if time_values is None:
            raise ValueError("time_values required for trend absorption")
        code, mem, counts = mats[0]
        t = np.asarray(time_values, dtype=float)
        tc = t - (mem @ t)[code] / counts[code]
        tc = tc / 12.0
        den = np.asarray(mem @ (tc * tc)).ravel()
        den = np.maximum(den, 1e-12)
    for _ in range(maxiter):
        old = A.copy()
        for j, (code, mem, counts) in enumerate(mats):
            means = (mem @ A) / counts[:, None]
            A -= means[code]
            if trend and j == 0:
                slopes = (mem @ (A * tc[:, None])) / den[:, None]
                A -= slopes[code] * tc[:, None]
        if np.max(np.abs(A - old)) < tol:
            return A
    raise RuntimeError("Two-way fixed-effect absorption did not converge")


def cluster_meat(X, u, groups):
    code, _ = pd.factorize(np.asarray(groups), sort=False)
    G = int(code.max() + 1)
    n, k = X.shape
    S = np.zeros((G, k), dtype=float)
    np.add.at(S, code, X * u[:, None])
    meat = S.T @ S
    if G > 1 and n > k + 1:
        meat *= (G / (G - 1)) * ((n - 1) / (n - k))
    return meat


def fit_event_model(stacks: pd.DataFrame, add_stack_country_trends=False):
    hs = [h for h in range(-12, 13) if h != -1]
    names = [f"tau_{'m'+str(abs(h)) if h<0 else 'p'+str(h)}" for h in hs]
    Xraw = np.column_stack([(stacks.treated.to_numpy() * (stacks.h.to_numpy() == h)).astype(float) for h in hs])
    yraw = stacks.pi_m.to_numpy(float)
    resid = absorb_two_way(
        np.column_stack([yraw, Xraw]),
        stacks.stack_country_id.to_numpy(), stacks.stack_month_id.to_numpy(),
        trend=add_stack_country_trends, time_values=stacks.mdate.to_numpy(float)
    )
    y, X = resid[:, 0], resid[:, 1:]
    bread = np.linalg.pinv(X.T @ X, rcond=1e-12)
    beta = bread @ X.T @ y
    u = y - X @ beta
    cell = stacks.iso3.astype(str).to_numpy() + "|" + stacks.mdate.astype(str).to_numpy()
    meat = (cluster_meat(X, u, stacks.iso3.to_numpy())
            + cluster_meat(X, u, stacks.mdate.to_numpy())
            - cluster_meat(X, u, cell))
    V = bread @ meat @ bread
    V = (V + V.T) / 2
    se = np.sqrt(np.maximum(np.diag(V), 0))
    Gc = stacks.iso3.nunique()
    Gt = stacks.mdate.nunique()
    df = int(min(Gc - 1, Gt - 1))
    tval = np.divide(beta, se, out=np.full_like(beta, np.nan), where=se > 0)
    pval = 2 * st.t.sf(np.abs(tval), df)
    crit = st.t.ppf(.975, df)
    bonf = st.t.ppf(1 - .05 / (2 * len(hs)), df)
    rows = []
    for j, h in enumerate(hs):
        rows.append({"h": h, "regressor": names[j], "coef_pp_monthly_inflation": beta[j],
                     "se_two_way_cluster": se[j], "p": pval[j],
                     "ci95_low": beta[j] - crit * se[j], "ci95_high": beta[j] + crit * se[j],
                     "simul95_bonf_low": beta[j] - bonf * se[j],
                     "simul95_bonf_high": beta[j] + bonf * se[j]})

    def joint_test(hlist):
        idx = [hs.index(h) for h in hlist]
        b = beta[idx]
        vv = (V[np.ix_(idx, idx)] + V[np.ix_(idx, idx)].T) / 2
        eig, vec = np.linalg.eigh(vv)
        # Multiway cluster covariance estimates can be non-positive-semidefinite
        # in finite samples. Project to the PSD cone for the joint Wald diagnostic,
        # and report the correction transparently rather than allowing negative Wald stats.
        tol_eig = max(1e-12, np.max(np.abs(eig)) * 1e-10)
        rank = int(np.sum(eig > tol_eig))
        vv_psd = vec @ np.diag(np.maximum(eig, 0.0)) @ vec.T
        stat = float(b @ np.linalg.pinv(vv_psd, rcond=1e-10) @ b) if rank else np.nan
        q = len(idx)
        fstat = stat / rank if rank else np.nan
        return {"wald_chi2_psd": stat, "df_num": rank, "restrictions": q, "df_den": df,
                "f_stat_psd": fstat, "p_psd": float(st.f.sf(fstat, rank, df)) if rank else np.nan,
                "min_eigenvalue_raw": float(eig.min()), "negative_eigenvalues": int(np.sum(eig < -tol_eig)),
                "psd_projection_applied": bool(np.any(eig < -tol_eig))}

    cum_rows = []
    for H in range(13):
        idx = [hs.index(h) for h in range(0, H + 1)]
        w = np.zeros(len(hs)); w[idx] = 1
        est = float(w @ beta)
        serr = float(np.sqrt(max(w @ V @ w, 0)))
        tv = est / serr if serr > 0 else np.nan
        pp = float(2 * st.t.sf(abs(tv), df)) if np.isfinite(tv) else np.nan
        c95 = st.t.ppf(.975, df)
        cbonf = st.t.ppf(1 - .05 / (2 * 13), df)
        cum_rows.append({"H": H, "cumulative_log_price_response_pct": est,
                         "exact_price_level_response_pct": 100 * (np.exp(est / 100) - 1),
                         "se": serr, "p_pointwise": pp,
                         "ci95_low": est - c95 * serr, "ci95_high": est + c95 * serr,
                         "simul95_bonf_low": est - cbonf * serr,
                         "simul95_bonf_high": est + cbonf * serr})
    return {
        "coefficients": pd.DataFrame(rows), "cumulative": pd.DataFrame(cum_rows),
        "covariance": V, "beta": beta, "se": se, "horizons": hs, "names": names,
        "pre_test": joint_test(list(range(-12, -1))),
        "post_test": joint_test(list(range(0, 13))),
        "post_0_6_test": joint_test(list(range(0, 7))),
        "n": len(stacks), "countries": Gc, "calendar_months": Gt,
        "episodes": stacks.episode_id.nunique(), "trend": add_stack_country_trends,
    }


def episode_level_effects(stacks: pd.DataFrame):
    monthly_rows, raw_rows = [], []
    for epid, g in stacks.groupby("episode_id", sort=False):
        # Country-specific baseline normalization over h=-6,...,-2.
        base = g.loc[g.h.between(-6, -2)].groupby("iso3").pi_m.mean()
        gg = g.copy()
        gg["pi_norm"] = gg.pi_m - gg.iso3.map(base)
        for h in range(-12, 13):
            gh = gg[gg.h == h]
            gt = gh[gh.treated == 1].pi_norm
            gc = gh[gh.treated == 0].pi_norm
            grawt = gh[gh.treated == 1].pi_m
            grawc = gh[gh.treated == 0].pi_m
            if len(gt) and len(gc):
                monthly_rows.append({"episode_id": epid, "h": h,
                                     "treated_minus_control_normalized": float(gt.mean() - gc.mean()),
                                     "treated_mean_normalized": float(gt.mean()),
                                     "control_mean_normalized": float(gc.mean()),
                                     "n_treated": int(gh[gh.treated == 1].iso3.nunique()),
                                     "n_controls": int(gh[gh.treated == 0].iso3.nunique())})
                raw_rows.append({"episode_id": epid, "h": h,
                                 "treated_mean_raw": float(grawt.mean()),
                                 "control_mean_raw": float(grawc.mean()),
                                 "raw_difference": float(grawt.mean() - grawc.mean())})
    monthly = pd.DataFrame(monthly_rows)
    raw = pd.DataFrame(raw_rows)
    cumulative = []
    for epid, g in monthly.groupby("episode_id", sort=False):
        g = g.set_index("h").sort_index()
        for H in range(13):
            val = float(g.loc[0:H, "treated_minus_control_normalized"].sum())
            cumulative.append({"episode_id": epid, "H": H, "cumulative_response_pct": val,
                               "exact_price_level_response_pct": 100 * (np.exp(val / 100) - 1)})
    return monthly, raw, pd.DataFrame(cumulative)


def signflip_family(df: pd.DataFrame, value_col: str, index_col: str, B=49999, seed=20261116):
    wide = df.pivot(index="episode_id", columns=index_col, values=value_col).dropna(axis=0, how="any")
    A = wide.to_numpy(float)
    E, K = A.shape
    if E < 3:
        return {"episodes": E, "error": "Fewer than 3 complete episodes"}, pd.DataFrame()
    means = A.mean(axis=0)
    ses = A.std(axis=0, ddof=1) / np.sqrt(E)
    if np.any(ses <= 1e-12):
        return {"episodes": E, "error": "Zero dispersion at one or more horizons; sign-flip t statistics are undefined"}, pd.DataFrame()
    tobs = means / ses
    rng = np.random.default_rng(seed)
    # Chunk to avoid large memory use for the 49,999 x 22 x 13 array.
    max_t = np.empty(B)
    for start in range(0, B, 1000):
        n = min(1000, B - start)
        signs = rng.choice(np.array([-1.0, 1.0]), size=(n, E))
        vals = signs[:, :, None] * A[None, :, :]
        bm = vals.mean(axis=1)
        bs = vals.std(axis=1, ddof=1) / np.sqrt(E)
        bt = np.divide(bm, bs, out=np.zeros_like(bm), where=bs > 0)
        max_t[start:start+n] = np.max(np.abs(bt), axis=1)
    crit = float(np.quantile(max_t, .95))
    max_obs = float(np.max(np.abs(tobs)))
    p_fwer = float((1 + np.sum(max_t >= max_obs)) / (B + 1))
    rows = []
    for j, label in enumerate(wide.columns):
        # marginal sign-flip p-value with the same randomization draws
        rng2 = np.random.default_rng(seed + 100 + j)
        null_t = np.empty(B)
        vals = A[:, j]
        for start in range(0, B, 1000):
            n = min(1000, B - start)
            signs = rng2.choice(np.array([-1.0, 1.0]), size=(n, E))
            z = signs * vals[None, :]
            zm = z.mean(axis=1)
            zs = z.std(axis=1, ddof=1) / np.sqrt(E)
            null_t[start:start+n] = np.divide(zm, zs, out=np.zeros_like(zm), where=zs > 0)
        p = float((1 + np.sum(np.abs(null_t) >= abs(tobs[j]))) / (B + 1))
        rows.append({"horizon": label, "episodes": E, "mean": means[j], "se": ses[j],
                     "t": tobs[j], "p_signflip": p,
                     "simul95_low": means[j] - crit * ses[j],
                     "simul95_high": means[j] + crit * ses[j],
                     "familywise_p_max_t": p_fwer})
    return {"episodes": E, "B": B, "max_t_critical_95": crit, "familywise_p": p_fwer}, pd.DataFrame(rows)


def episode_mean_test(vals: pd.Series, B=49999, seed=20261116):
    x = vals.dropna().to_numpy(float)
    E = len(x)
    if E < 3:
        return {"episodes": E, "error": "Fewer than 3 episodes"}
    mean = float(x.mean()); se = float(x.std(ddof=1) / np.sqrt(E))
    if se <= 1e-12:
        return {"episodes": E, "error": "Zero dispersion across units; t and sign-flip inference are undefined"}
    t = mean / se
    rng = np.random.default_rng(seed)
    null_t = np.empty(B)
    for start in range(0, B, 1000):
        n = min(1000, B - start)
        signs = rng.choice(np.array([-1.0, 1.0]), size=(n, E))
        z = signs * x[None, :]
        zm = z.mean(axis=1); zs = z.std(axis=1, ddof=1) / np.sqrt(E)
        null_t[start:start+n] = np.divide(zm, zs, out=np.zeros_like(zm), where=zs > 0)
    p = float((1 + np.sum(np.abs(null_t) >= abs(t))) / (B + 1))
    crit = st.t.ppf(.975, E - 1)
    return {"episodes": E, "mean": mean, "median": float(np.median(x)), "se": se,
            "t": t, "df": E - 1, "p_t": float(2 * st.t.sf(abs(t), E - 1)),
            "p_signflip": p, "ci95_low": mean - crit * se, "ci95_high": mean + crit * se}


def make_figures(stacks, coef, cumulative, monthly, raw, episode_cum, cum_family_df, outdir: Path, family_label="episode-level descriptive"):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                         "axes.spines.top": False, "axes.spines.right": False})
    # Figure 1: raw, unnormalized treated and control monthly inflation paths.
    raw_sum = raw.groupby("h").agg(treat_mean=("treated_mean_raw", "mean"),
                                    control_mean=("control_mean_raw", "mean"),
                                    treat_sd=("treated_mean_raw", "std"),
                                    control_sd=("control_mean_raw", "std"),
                                    E=("episode_id", "nunique")).reset_index()
    for grp in ["treat", "control"]:
        raw_sum[f"{grp}_se"] = raw_sum[f"{grp}_sd"] / np.sqrt(raw_sum.E)
    fig, ax = plt.subplots(figsize=(8.3, 4.4))
    ax.errorbar(raw_sum.h, raw_sum.treat_mean, yerr=1.96*raw_sum.treat_se, fmt="o", linestyle="none", ms=3.5,
                capsize=2, label="Treated countries (raw)")
    ax.errorbar(raw_sum.h, raw_sum.control_mean, yerr=1.96*raw_sum.control_se, fmt="s", linestyle="none", ms=3.5,
                capsize=2, label="Eligible controls (raw)")
    ax.axvline(-1, color="0.45", ls=":", lw=1); ax.axvline(0, color="0.2", ls="--", lw=1)
    ax.set(xlabel="Months relative to UCDP threshold-crossing month", ylabel="Monthly headline inflation (%)",
           title="Raw inflation paths around UCDP conflict onsets")
    ax.legend(frameon=False, ncol=2, loc="best")
    fig.text(.5, .01, "Event-equal averages; no pre-event normalization or smoothing. Error bars: ±1.96 SE across episodes.",
             ha="center", fontsize=7.5)
    fig.tight_layout(rect=[0, .04, 1, 1]); fig.savefig(outdir/"ucdp_fig1_raw_paths.png", dpi=240); plt.close(fig)

    # Figure 2: stacked regression coefficients; points and bars, not a smoothed curve.
    c = coef.sort_values("h")
    fig, ax = plt.subplots(figsize=(8.3, 4.4))
    ax.errorbar(c.h, c.coef_pp_monthly_inflation,
                yerr=[c.coef_pp_monthly_inflation-c.ci95_low, c.ci95_high-c.coef_pp_monthly_inflation],
                fmt="o", ms=4, capsize=2, lw=1.0, label="Pointwise 95% CI")
    ax.vlines(c.h, c.simul95_bonf_low, c.simul95_bonf_high, lw=2, alpha=.35, label="Bonferroni simultaneous 95% CI")
    ax.axhline(0, color="0.35", ls="--", lw=1); ax.axvline(-1, color="0.45", ls=":", lw=1); ax.axvline(0, color="0.2", ls="--", lw=1)
    ax.set(xlabel="Months relative to UCDP threshold-crossing month", ylabel="Treated–control difference (pp monthly inflation)",
           title="Stacked event-study coefficients (reference month = −1)")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(outdir/"ucdp_fig2_event_study_coefficients.png", dpi=240); plt.close(fig)

    # Figure 3: episode-level 0-6 cumulative effects; shows dispersion transparently.
    d = episode_cum[episode_cum.H == 6].sort_values("cumulative_response_pct").reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(8.3, 5.0))
    ax.scatter(d.cumulative_response_pct, np.arange(1, len(d)+1), s=28)
    ax.axvline(0, color="0.35", ls="-", lw=1)
    ax.axvline(d.cumulative_response_pct.mean(), color="0.2", ls="--", lw=1.4, label=f"Equal-episode mean = {d.cumulative_response_pct.mean():.2f}%")
    ax.axvline(d.cumulative_response_pct.median(), color="0.5", ls=":", lw=1.4, label=f"Median = {d.cumulative_response_pct.median():.2f}%")
    ax.set(yticks=np.arange(1, len(d)+1), yticklabels=d.episode_id,
           xlabel="Cumulative normalized price response, months 0–6 (%)",
           ylabel="UCDP conflict episode", title="Dispersion across independent conflict episodes")
    ax.legend(frameon=False, loc="best", fontsize=8)
    fig.text(.5, .01, "One dot per episode; no ranking of true conflict-specific causal effects is implied.", ha="center", fontsize=7.5)
    fig.tight_layout(rect=[0, .03, 1, 1]); fig.savefig(outdir/"ucdp_fig3_episode_dispersion.png", dpi=240); plt.close(fig)

    # Figure 4: normalized paths; baseline window explicitly shaded.
    mn = monthly.groupby("h").agg(treat=("treated_mean_normalized", "mean"),
                                    control=("control_mean_normalized", "mean"),
                                    treat_sd=("treated_mean_normalized", "std"),
                                    control_sd=("control_mean_normalized", "std"),
                                    E=("episode_id", "nunique")).reset_index()
    mn["treat_se"] = mn.treat_sd / np.sqrt(mn.E)
    mn["control_se"] = mn.control_sd / np.sqrt(mn.E)
    fig, ax = plt.subplots(figsize=(8.3, 4.4))
    ax.axvspan(-6, -2, color="0.85", alpha=.55, label="Normalization window (−6 to −2)")
    ax.errorbar(mn.h, mn.treat, yerr=1.96*mn.treat_se, fmt="o", linestyle="none", ms=3.5, capsize=2, label="Treated")
    ax.errorbar(mn.h, mn.control, yerr=1.96*mn.control_se, fmt="s", linestyle="none", ms=3.5, capsize=2, label="Controls")
    ax.axhline(0, color="0.35", ls="--", lw=1); ax.axvline(0, color="0.2", ls="--", lw=1)
    ax.set(xlabel="Months relative to UCDP threshold-crossing month", ylabel="Monthly inflation minus own pre-event mean (pp)",
           title="Within-country normalized trajectories (descriptive)")
    ax.legend(frameon=False, ncol=2, fontsize=8)
    fig.text(.5, .01, "Pretrend holdout is h = −12,…,−7 and h = −1; the shaded normalization window is not used as the holdout.",
             ha="center", fontsize=7.3)
    fig.tight_layout(rect=[0, .04, 1, 1]); fig.savefig(outdir/"ucdp_fig4_normalized_paths.png", dpi=240); plt.close(fig)

    # Figure 5: equal-episode cumulative path with wild max-|t| simultaneous bands.
    cf = cum_family_df.sort_values("horizon").copy()
    fig, ax = plt.subplots(figsize=(8.3, 4.4))
    ax.errorbar(cf.horizon.astype(int), cf["mean"],
                yerr=[cf["mean"]-cf["simul95_low"], cf["simul95_high"]-cf["mean"]],
                fmt="o", linestyle="none", ms=3.5, capsize=2,
                label=f"{family_label} mean; simultaneous 95% max-|t| band")
    ax.axhline(0, color="0.35", ls="--", lw=1); ax.axvline(0, color="0.2", ls="--", lw=1)
    ax.set(xlabel="Months after UCDP threshold-crossing month", ylabel="Cumulative normalized price response (%)",
           title=f"Cumulative response: {family_label}")
    ax.legend(frameon=False, fontsize=8)
    note=("Episodes are grouped when event windows overlap or treated countries are shared. "
          "Bands use cluster-level Rademacher sign-flip max-|t| across horizons 0–12." if family_label=="dependence-cluster" else
          "Episode-level bands are descriptive only; the number of independent episodes may be smaller than the episode count.")
    fig.text(.5, .01, note, ha="center", fontsize=7.5)
    fig.tight_layout(rect=[0, .04, 1, 1]); fig.savefig(outdir/"ucdp_fig5_cumulative_dependence_cluster.png", dpi=240); plt.close(fig)



def assign_episode_dependence_clusters(event_audit: pd.DataFrame, window_months=12):
    """Conservative connected components: link episodes with overlapping event windows
    or any common treated country. Cluster-level sign flips are reported alongside
    episode-level sign flips because 22 episode labels need not imply 22 independent units.
    """
    if event_audit.empty:
        return pd.DataFrame(columns=["episode_id", "dependence_cluster"])
    rows=event_audit.reset_index(drop=True)
    n=len(rows); parent=list(range(n))
    def find(x):
        while parent[x]!=x:
            parent[x]=parent[parent[x]]; x=parent[x]
        return x
    def union(a,b):
        ra,rb=find(a),find(b)
        if ra!=rb: parent[rb]=ra
    countries=[]
    for r in rows.itertuples(index=False):
        countries.append(set(str(r.treated_iso3_used).split(';'))-{'','nan','None'})
    for i in range(n):
        for j in range(i+1,n):
            overlap=abs(int(rows.loc[i,'event_month'])-int(rows.loc[j,'event_month'])) <= 2*window_months
            shared=bool(countries[i].intersection(countries[j]))
            if overlap or shared: union(i,j)
    roots=[find(i) for i in range(n)]
    unique={r:k+1 for k,r in enumerate(sorted(set(roots)))}
    return pd.DataFrame({'episode_id':rows.episode_id.astype(str),
                         'dependence_cluster':[f'C{unique[r]:02d}' for r in roots]})


def cluster_level_signflip(episode_cum, monthly, pre_hold, cluster_map, B, seed):
    ec=episode_cum.merge(cluster_map,on='episode_id',how='inner')
    mc=monthly.merge(cluster_map,on='episode_id',how='inner')
    cum=ec.groupby(['dependence_cluster','H'],as_index=False).cumulative_response_pct.mean()
    cum_for=cum.rename(columns={'dependence_cluster':'episode_id','H':'H'})
    cum_family,cum_df=signflip_family(cum_for,'cumulative_response_pct','H',B,seed)
    mon=mc.groupby(['dependence_cluster','h'],as_index=False).treated_minus_control_normalized.mean()
    mon_for=mon.rename(columns={'dependence_cluster':'episode_id'})
    mon_family,mon_df=signflip_family(mon_for,'treated_minus_control_normalized','h',B,seed+1)
    ph=pre_hold.rename('pre_value').reset_index()
    ph=ph.merge(cluster_map,on='episode_id',how='inner').groupby('dependence_cluster').pre_value.mean()
    ep6=cum[cum.H==6].set_index('dependence_cluster').cumulative_response_pct
    ep12=cum[cum.H==12].set_index('dependence_cluster').cumulative_response_pct
    result={'dependence_clusters':int(cluster_map.dependence_cluster.nunique()),
            'episodes':int(cluster_map.episode_id.nunique()),
            'cumulative_0_6':episode_mean_test(ep6,B,seed+2),
            'cumulative_0_12':episode_mean_test(ep12,B,seed+3),
            'pre_event_holdout':episode_mean_test(ph,B,seed+4),
            'cumulative_familywise':cum_family,
            'monthly_post_familywise':mon_family}
    return result,cum_df,mon_df,cum,mon



def make_timing_sensitivity_figure(coef_frame: pd.DataFrame, outdir: Path):
    if coef_frame is None or coef_frame.empty:
        return
    labels={
        "threshold_primary":"Threshold date; primary countries",
        "first_conflict_fatality_first_episode":"Conflict first fatality (c_epno=1 only); primary countries",
        "threshold_primary_plus_secondary":"Threshold date; primary + secondary troop contributors"
    }
    order=[k for k in labels if k in set(coef_frame.specification)]
    if not order: return
    fig,axs=plt.subplots(1,len(order),figsize=(5.0*len(order),4.2),sharey=True)
    if len(order)==1: axs=[axs]
    offsets=np.linspace(-0.10,0.10,len(order)) if len(order)>1 else [0]
    for ax,key,off in zip(axs,order,offsets):
        z=coef_frame[coef_frame.specification==key].sort_values('h')
        y=z.coef_pp_monthly_inflation.to_numpy()
        lo=z.simul95_bonf_low.to_numpy(); hi=z.simul95_bonf_high.to_numpy()
        ax.errorbar(z.h.to_numpy()+off,y,yerr=[y-lo,hi-y],fmt='o',linestyle='none',ms=3.3,capsize=2)
        ax.axhline(0,color='0.4',ls=':',lw=.8); ax.axvline(0,color='0.25',ls='--',lw=.8)
        ax.set_title(labels[key],fontsize=8.5); ax.set_xlabel('Event time (months)'); ax.set_xticks([-12,-6,-1,0,6,12])
    axs[0].set_ylabel('Treated–control difference (pp monthly inflation)')
    fig.suptitle('Event-study sensitivity to onset date and treatment definition')
    fig.text(.5,.01,'Dots are unsmoothed event-time estimates; bars are Bonferroni simultaneous 95% intervals.',ha='center',fontsize=7.5)
    fig.tight_layout(rect=[0,.04,1,.94]); fig.savefig(outdir/'ucdp_fig6_timing_treatment_sensitivity.png',dpi=240); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", default=str(DEFAULT_PANEL))
    ap.add_argument("--ucdp", default=str(DEFAULT_UCDP))
    ap.add_argument("--no-download", action="store_true", help="Do not attempt automatic UCDP download")
    ap.add_argument("--bootstrap", type=int, default=49999)
    ap.add_argument("--seed", type=int, default=20261116)
    ap.add_argument("--with-trends", action="store_true", help="Also estimate stack-country linear-trend robustness")
    ap.add_argument("--no-sensitivities", action="store_true", help="Skip alternative onset-date and treatment-definition checks")
    ap.add_argument("--overrides", default=str(HERE / "ucdp_country_overrides.csv"),
                    help="Optional manual country-name crosswalk CSV")
    args = ap.parse_args()
    panel_path = Path(args.panel)
    ucdp_path = Path(args.ucdp)
    if not panel_path.exists():
        raise SystemExit(f"Panel file not found: {panel_path}")
    if not ucdp_path.exists():
        if args.no_download:
            raise SystemExit(f"UCDP CSV not found: {ucdp_path}")
        download_ucdp(ucdp_path)

    panel = panel_prepare(panel_path)
    overrides_path = Path(args.overrides)
    map_one, panel_iso3 = load_country_maps(panel, overrides_path)
    raw_ucdp = read_csv_flexible(ucdp_path)
    ucdp, episodes_all, unmapped = build_episode_metadata(raw_ucdp, map_one, panel_iso3)
    if unmapped:
        print(f"NOTE: {len(unmapped)} historical/non-panel UCDP names remain unmatched. The script separately checks eligible type-2/type-4 onsets in 2000–2023 before estimation; see ucdp_unmapped_locations.csv for the full audit.")
        unmapped_ids = {str(x["c_epid"]) for x in unmapped}
        relevant_unmapped = episodes_all.loc[
            episodes_all.episode_id.astype(str).isin(unmapped_ids)
            & episodes_all.active_type24
            & episodes_all.active_years.apply(lambda ys: any(1999 <= int(y) <= 2024 for y in ys))
        ]
        if len(relevant_unmapped):
            raise SystemExit(
                f"STOP: {len(relevant_unmapped)} type-2/type-4 episodes active in 1999-2024 have unresolved country mappings. "
                "Do not estimate with potentially contaminated controls. Review output/ucdp_unmapped_locations.csv, "
                "complete ucdp_country_overrides.csv, and rerun."
            )
    episodes_all.to_json(OUT/"ucdp_episode_metadata.json", orient="records", indent=2, date_format="iso")

    stacks, event_audit, dropped, selected = build_stacks(panel, episodes_all, episodes_all,
                                                           date_source="threshold", treat_secondary_at_onset=False)
    if event_audit.empty or stacks.empty:
        raise SystemExit("No usable UCDP event stacks were created. Inspect ucdp_unmapped_locations.csv and the episode audit.")
    event_audit.to_csv(OUT/"ucdp_event_sample_audit.csv", index=False)
    dropped.to_csv(OUT/"ucdp_dropped_episodes.csv", index=False)
    stacks.to_csv(OUT/"ucdp_stacked_panel.csv", index=False)
    # Export a compact Stata file for independent estimation.
    stacks.to_stata(OUT/"ucdp_stacked_panel.dta", write_index=False, version=118)

    print("Built UCDP stacks:", {"episodes": int(stacks.episode_id.nunique()),
                                  "treated_country_events": int(event_audit.n_treated.sum()),
                                  "observations": int(len(stacks)),
                                  "countries": int(stacks.iso3.nunique()),
                                  "calendar_months": int(stacks.mdate.nunique())})
    replication_check = {
        "expected_original": EXPECTED_REPLICATION,
        "reconstructed": {"episodes": int(stacks.episode_id.nunique()),
                          "treated_country_events": int(event_audit.n_treated.sum()),
                          "stack_observations": int(len(stacks))},
        "matches_original_counts": {
            "episodes": int(stacks.episode_id.nunique()) == EXPECTED_REPLICATION["episodes"],
            "treated_country_events": int(event_audit.n_treated.sum()) == EXPECTED_REPLICATION["treated_country_events"],
            "stack_observations": int(len(stacks)) == EXPECTED_REPLICATION["stack_observations"],
        },
        "interpretation": "A count mismatch is an audit finding, not a reason to force or fabricate the original sample."
    }
    (OUT/"ucdp_replication_count_check.json").write_text(json.dumps(replication_check, indent=2), encoding="utf-8")

    model = fit_event_model(stacks, add_stack_country_trends=False)
    model["coefficients"].to_csv(OUT/"ucdp_event_study_coefficients.csv", index=False)
    sensitivity_coef_frames=[model["coefficients"].assign(specification="threshold_primary")]
    model["cumulative"].to_csv(OUT/"ucdp_event_study_cumulative.csv", index=False)
    if args.with_trends:
        trend_model = fit_event_model(stacks, add_stack_country_trends=True)
        trend_model["coefficients"].to_csv(OUT/"ucdp_event_study_coefficients_country_trends.csv", index=False)
        trend_model["cumulative"].to_csv(OUT/"ucdp_event_study_cumulative_country_trends.csv", index=False)
    if not args.no_sensitivities:
        sensitivity_specs = [
            ("threshold_primary_plus_secondary", "threshold", True),
        ]
        sens_summaries=[]
        for label, date_source, treat_secondary in sensitivity_specs:
            ss, sa, sd, ssel = build_stacks(panel, episodes_all, episodes_all,
                                            date_source=date_source,
                                            treat_secondary_at_onset=treat_secondary)
            if ss.empty:
                sens_summaries.append({"specification":label,"episodes":0,"reason":"No usable stacks"})
                continue
            sm=fit_event_model(ss,add_stack_country_trends=False)
            sensitivity_coef_frames.append(sm["coefficients"].assign(specification=label))
            sm["coefficients"].to_csv(OUT/f"ucdp_{label}_coefficients.csv",index=False)
            sm["cumulative"].to_csv(OUT/f"ucdp_{label}_cumulative.csv",index=False)
            sa.to_csv(OUT/f"ucdp_{label}_sample_audit.csv",index=False)
            sd.to_csv(OUT/f"ucdp_{label}_dropped_episodes.csv",index=False)
            sens_summaries.append({"specification":label,"date_source":date_source,
                "treat_secondary_at_onset":treat_secondary,"episodes":int(ss.episode_id.nunique()),
                "treated_country_events":int(sa.n_treated.sum()),"observations":int(len(ss)),
                "pretrend_joint_test":sm["pre_test"],"post_joint_test":sm["post_test"]})
        (OUT/"ucdp_timing_treatment_sensitivity.json").write_text(json.dumps(sens_summaries,indent=2),encoding="utf-8")
    sens_coefs=pd.concat(sensitivity_coef_frames,ignore_index=True)
    sens_coefs.to_csv(OUT/"ucdp_event_study_timing_treatment_coefficients.csv",index=False)
    make_timing_sensitivity_figure(sens_coefs,OUT)
    monthly, raw, episode_cum = episode_level_effects(stacks)
    monthly.to_csv(OUT/"ucdp_episode_monthly_effects.csv", index=False)
    raw.to_csv(OUT/"ucdp_episode_raw_paths.csv", index=False)
    episode_cum.to_csv(OUT/"ucdp_episode_cumulative_effects.csv", index=False)

    # Equal-episode inference for the principal 0-6 and 0-12 cumulative estimands.
    ep6 = episode_cum.loc[episode_cum.H == 6].set_index("episode_id").cumulative_response_pct
    ep12 = episode_cum.loc[episode_cum.H == 12].set_index("episode_id").cumulative_response_pct
    infer = {"cumulative_0_6": episode_mean_test(ep6, args.bootstrap, args.seed),
             "cumulative_0_12": episode_mean_test(ep12, args.bootstrap, args.seed + 1)}
    # Family-wise max-|t| inference across cumulative horizons and monthly post-onset horizons.
    cum_family, cum_family_df = signflip_family(
        episode_cum.rename(columns={"cumulative_response_pct": "value"}), "value", "H", args.bootstrap, args.seed + 2)
    post_month = monthly[monthly.h.between(0, 12)].rename(columns={"treated_minus_control_normalized": "value"})
    month_family, month_family_df = signflip_family(post_month, "value", "h", args.bootstrap, args.seed + 3)
    pre_hold = monthly[monthly.h.isin(list(range(-12, -6)) + [-1])].groupby("episode_id").treated_minus_control_normalized.mean()
    infer["pre_event_holdout"] = episode_mean_test(pre_hold, args.bootstrap, args.seed + 4)
    infer["cumulative_familywise"] = cum_family
    infer["monthly_post_familywise"] = month_family
    cum_family_df.to_csv(OUT/"ucdp_cumulative_signflip_family.csv", index=False)
    month_family_df.to_csv(OUT/"ucdp_monthly_signflip_family.csv", index=False)
    (OUT/"ucdp_episode_level_inference.json").write_text(json.dumps(infer, indent=2), encoding="utf-8")
    # More conservative dependence-aware inference: episodes are linked when their
    # 25-month windows overlap or they share a treated country.
    cluster_map = assign_episode_dependence_clusters(event_audit, window_months=12)
    cluster_map.to_csv(OUT/"ucdp_episode_dependence_clusters.csv", index=False)
    cluster_infer, cluster_cum_df, cluster_month_df, cluster_cum, cluster_month = cluster_level_signflip(
        episode_cum, monthly, pre_hold, cluster_map, args.bootstrap, args.seed+20)
    cluster_cum_df.to_csv(OUT/"ucdp_cluster_cumulative_signflip_family.csv", index=False)
    cluster_month_df.to_csv(OUT/"ucdp_cluster_monthly_signflip_family.csv", index=False)
    (OUT/"ucdp_cluster_level_inference.json").write_text(json.dumps(cluster_infer, indent=2), encoding="utf-8")

    # Leave-one-episode-out influence analysis for the 0-6 cumulative response.
    loo = []
    vals = episode_cum[episode_cum.H == 6].set_index("episode_id").cumulative_response_pct
    for epid in vals.index:
        x = vals.drop(index=epid)
        loo.append({"episode_removed": epid, "mean_0_6": float(x.mean()),
                    "median_0_6": float(x.median()), "episodes_remaining": len(x)})
    pd.DataFrame(loo).to_csv(OUT/"ucdp_leave_one_episode_out.csv", index=False)

    fig_family_df = cluster_cum_df if not cluster_cum_df.empty else cum_family_df
    family_label = "dependence-cluster" if not cluster_cum_df.empty else "episode-level descriptive"
    make_figures(stacks, model["coefficients"], model["cumulative"], monthly, raw, episode_cum, fig_family_df, OUT, family_label)
    summary = {
        "source_file": str(ucdp_path.name), "source_url": UCDP_URL,
        "source_sha256": hashlib.sha256(ucdp_path.read_bytes()).hexdigest(),
        "codebook_url": CODEBOOK_URL,
        "panel_file": str(panel_path.name), "panel_countries_after_exclusions": int(panel.iso3.nunique()),
        "selected_onsets_before_stack_eligibility": int(len(selected)),
        "usable_episodes": int(stacks.episode_id.nunique()),
        "treated_country_events": int(event_audit.n_treated.sum()),
        "stack_observations": int(len(stacks)),
        "countries_in_stacks": int(stacks.iso3.nunique()),
        "calendar_months_in_stacks": int(stacks.mdate.nunique()),
        "event_study_pretrend_joint_test": model["pre_test"],
        "event_study_post_joint_test": model["post_test"],
        "event_study_post_0_6_joint_test": model["post_0_6_test"],
        "episode_level_inference": infer,
        "dependence_aware_cluster_inference": cluster_infer,
        "episode_dependence_clusters": int(cluster_map.dependence_cluster.nunique()),
        "replication_count_check": replication_check,
        "warning": "These are conditional event-time associations. UCDP threshold-crossing dates do not equal the first violent incident; no causal interpretation follows without additional identifying assumptions."
    }
    (OUT/"ucdp_event_study_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\nCompleted. Outputs written to:", OUT)
    print("Pre-event joint test:", model["pre_test"])
    print("Post-event joint test:", model["post_test"])
    print("Episode-level 0-6:", infer["cumulative_0_6"])
    print("IMPORTANT: inspect ucdp_replication_count_check.json before claiming exact replication.")


if __name__ == "__main__":
    main()
