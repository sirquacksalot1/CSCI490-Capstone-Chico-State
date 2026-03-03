"""
runsignup_2mile_splits_scraper.py

Goal:
  Build a CSV dataset of real race results with:
    - 2 Mile finish time
    - Mile 1 / Mile 2 split times (when available)

Same endpoints/auth/output schema as your 5K scraper (mile3 columns remain for compatibility).
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import os
import re
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import requests


BASE_URL = "https://api.runsignup.com/rest"
UPDATED_RESULT_SETS_PATH = "/v2/results/updated-result-sets.json"
GET_RESULTS_PATH = "/race/{race_id}/results/get-results"
GET_SPLITS_PATH = "/race/{race_id}/results/result-set-splits"
GET_RACE_PATH = "/race/{race_id}"


_TIME_RE = re.compile(r"^\s*(?:(\d+):)?(\d{1,2}):(\d{2})(?:\.\d+)?\s*$")

def time_str_to_seconds(s: str) -> Optional[int]:
    if s is None:
        return None
    s = str(s).strip()
    if not s or s.upper() in {"NONE", "N/A", "NA", "NULL"}:
        return None
    m = _TIME_RE.match(s)
    if not m:
        return None
    h = int(m.group(1)) if m.group(1) is not None else 0
    mm = int(m.group(2))
    ss = int(m.group(3))
    return h * 3600 + mm * 60 + ss


@dataclass
class RunSignupClient:
    api_key: str
    api_secret: str
    timeout_s: int = 30
    sleep_s: float = 0.15

    def _get(self, path: str, params: Dict[str, Any]) -> Dict[str, Any]:
        url = BASE_URL + path
        qp = dict(params)
        qp["api_key"] = self.api_key
        qp["api_secret"] = self.api_secret
        qp.setdefault("format", "json")

        resp = requests.get(url, params=qp, timeout=self.timeout_s)
        if resp.status_code != 200:
            raise RuntimeError(f"HTTP {resp.status_code} for {url}: {resp.text[:500]}")

        time.sleep(self.sleep_s)
        try:
            return resp.json()
        except Exception as e:
            raise RuntimeError(f"Failed to parse JSON from {url}: {e}") from e

    def get_updated_result_sets(
        self,
        page: int,
        num_per_page: int,
        modified_since_ts: Optional[int] = None,
        modified_until_ts: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        params: Dict[str, Any] = {"page": page, "num_per_page": num_per_page}
        if modified_since_ts is not None:
            params["modified_since_timestamp"] = modified_since_ts
        if modified_until_ts is not None:
            params["modified_until_timestamp"] = modified_until_ts
        data = self._get(UPDATED_RESULT_SETS_PATH, params)
        return data.get("result_sets", []) or []

    def get_race(self, race_id: int) -> Dict[str, Any]:
        return self._get(GET_RACE_PATH.format(race_id=race_id), params={})

    def get_event_results(
        self,
        race_id: int,
        event_id: int,
        individual_result_set_id: int,
        page: int = 1,
        results_per_page: int = 250,
        include_split_time_ms: bool = False,
    ) -> Dict[str, Any]:
        params = {
            "race_id": race_id,
            "event_id": event_id,
            "individual_result_set_id": individual_result_set_id,
            "page": page,
            "results_per_page": results_per_page,
            "include_split_time_ms": "T" if include_split_time_ms else "F",
        }
        return self._get(GET_RESULTS_PATH.format(race_id=race_id), params=params)

    def get_result_set_splits(self, race_id: int, event_id: int, individual_result_set_id: int) -> Dict[str, Any]:
        params = {"race_id": race_id, "event_id": event_id, "individual_result_set_id": individual_result_set_id}
        return self._get(GET_SPLITS_PATH.format(race_id=race_id), params=params)


def safe_lower(x: Any) -> str:
    return str(x or "").lower()

def _distance_to_miles(dist: Any, unit: str) -> Optional[float]:
    if dist is None:
        return None
    try:
        d = float(dist)
    except Exception:
        return None
    u = (unit or "").lower()
    if "km" in u:
        return d * 0.621371
    if "mi" in u or "mile" in u:
        return d
    return None

def looks_like_2mile_event(event: Dict[str, Any]) -> bool:
    name = safe_lower(event.get("name") or event.get("event_name"))
    event_type = safe_lower(event.get("event_type"))
    dist = event.get("distance") or event.get("event_distance") or event.get("length")
    unit = safe_lower(event.get("distance_unit") or event.get("unit") or event.get("event_distance_unit"))

    if any(bad in name for bad in ["kid", "kids", "children", "tot", "toddler"]):
        return False

    if ("2 mile" in name) or ("two mile" in name) or re.search(r"\b2\s*mi\b", name):
        return True
    if event_type in {"2 mile", "2mile"} or ("2mile" in event_type) or ("2 mile" in event_type):
        return True

    miles = _distance_to_miles(dist, unit)
    if miles is not None and abs(miles - 2.0) <= 0.12:
        return True

    return False

def extract_event_name(event: Dict[str, Any]) -> str:
    return str(event.get("name") or event.get("event_name") or "").strip()

def find_event_by_id(race_payload: Dict[str, Any], event_id: int) -> Optional[Dict[str, Any]]:
    candidates = []
    if isinstance(race_payload, dict):
        if "race" in race_payload and isinstance(race_payload["race"], dict):
            r = race_payload["race"]
            for key in ("events", "event"):
                if key in r and isinstance(r[key], list):
                    candidates.extend(r[key])
        for key in ("events", "event"):
            if key in race_payload and isinstance(race_payload[key], list):
                candidates.extend(race_payload[key])
    for e in candidates:
        try:
            if int(e.get("event_id") or e.get("id")) == int(event_id):
                return e
        except Exception:
            continue
    return None

def iter_results_from_get_results(payload: Dict[str, Any]) -> Iterable[Tuple[Dict[str, Any], Dict[str, Any]]]:
    sets = payload.get("individual_results_sets") or payload.get("individual_results_sets".replace("_", "")) or []
    if not isinstance(sets, list):
        return
    for rs in sets:
        results = rs.get("results", [])
        if isinstance(results, list):
            for row in results:
                if isinstance(row, dict):
                    yield rs, row

def parse_splits_response(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    if not isinstance(payload, dict):
        return []
    for key in ("splits", "result_set_splits", "result_sets_splits", "result_set_split", "result_set_splits"):
        v = payload.get(key)
        if isinstance(v, list) and all(isinstance(x, dict) for x in v):
            return v
    for v in payload.values():
        if isinstance(v, list) and v and all(isinstance(x, dict) for x in v):
            return v
    return []

def split_def_id(split_def: Dict[str, Any]) -> Optional[int]:
    for k in ("split_id", "result_set_split_id", "individual_result_set_split_id", "id"):
        if k in split_def:
            try:
                return int(split_def[k])
            except Exception:
                continue
    return None

def split_def_name(split_def: Dict[str, Any]) -> str:
    for k in ("name", "split_name", "label", "display_name"):
        if k in split_def and split_def[k]:
            return str(split_def[k])
    return ""

def split_def_distance(split_def: Dict[str, Any]) -> Tuple[Optional[float], str]:
    dist = None
    unit = ""
    for dk in ("distance", "split_distance", "dist"):
        if dk in split_def and split_def[dk] is not None:
            try:
                dist = float(split_def[dk])
                break
            except Exception:
                pass
    for uk in ("distance_unit", "unit", "split_distance_unit"):
        if uk in split_def and split_def[uk]:
            unit = str(split_def[uk]).lower()
            break
    return dist, unit

def map_mile_splits(split_defs: List[Dict[str, Any]], max_mile: int) -> Dict[int, int]:
    entries = []
    for sd in split_defs:
        sid = split_def_id(sd)
        if sid is None:
            continue
        name = split_def_name(sd).strip()
        dist, unit = split_def_distance(sd)
        entries.append((sid, name, dist, unit))

    mile_map: Dict[int, int] = {}

    for target_mile in range(1, max_mile + 1):
        best = None
        for sid, name, dist, unit in entries:
            if dist is None:
                continue
            miles = None
            if "km" in unit:
                miles = dist * 0.621371
            elif "mi" in unit or "mile" in unit:
                miles = dist
            else:
                continue
            err = abs(miles - float(target_mile))
            if err <= 0.12:
                if best is None or err < best[0]:
                    best = (err, sid)
        if best is not None:
            mile_map[target_mile] = best[1]

    if len(mile_map) < max_mile:
        for target_mile in range(1, max_mile + 1):
            if target_mile in mile_map:
                continue
            pattern = re.compile(rf"\b(mile|mi)\s*{target_mile}\b|\b{target_mile}\s*(mile|mi)\b", re.IGNORECASE)
            for sid, name, dist, unit in entries:
                if name and pattern.search(name):
                    mile_map[target_mile] = sid
                    break

    return mile_map

def extract_split_time(row: Dict[str, Any], split_id: int) -> Optional[str]:
    for key in (f"split-{split_id}", f"split_{split_id}"):
        if key in row:
            v = row.get(key)
            if v is None:
                return None
            return str(v).strip()
    return None


def unix_ts_days_ago(days: int) -> int:
    now = dt.datetime.utcnow()
    past = now - dt.timedelta(days=days)
    return int(past.timestamp())

def pick_finish_time(row: Dict[str, Any]) -> Tuple[Optional[str], Optional[int]]:
    chip = row.get("chip_time")
    clock = row.get("clock_time")
    raw = None
    if chip and str(chip).strip() and str(chip).strip().upper() != "NONE":
        raw = str(chip).strip()
    elif clock and str(clock).strip() and str(clock).strip().upper() != "NONE":
        raw = str(clock).strip()
    return raw, time_str_to_seconds(raw) if raw else (None, None)

def scrape(
    client: RunSignupClient,
    days_back: int,
    max_result_sets: int,
    results_per_page: int,
    max_pages_per_result_set: int,
    require_all_miles_present: bool,
    verbose: bool,
) -> List[Dict[str, Any]]:
    rows_out: List[Dict[str, Any]] = []
    modified_since = unix_ts_days_ago(days_back)

    page = 1
    num_per_page = min(5000, max_result_sets) if max_result_sets > 0 else 5000
    seen = 0

    MAX_MILE = 2

    while seen < max_result_sets:
        result_sets = client.get_updated_result_sets(page=page, num_per_page=num_per_page, modified_since_ts=modified_since)
        if not result_sets:
            break

        for rs in result_sets:
            if seen >= max_result_sets:
                break
            seen += 1

            race_id = int(rs["race_id"])
            event_id = int(rs["event_id"])
            result_set_id = int(rs["individual_result_set_id"])
            race_name = str(rs.get("race_name", "")).strip()
            result_set_name = str(rs.get("individual_result_set_name", "")).strip()

            try:
                race_payload = client.get_race(race_id)
                event = find_event_by_id(race_payload, event_id) or {}
            except Exception as e:
                if verbose:
                    print(f"[WARN] get_race failed for race_id={race_id}: {e}", file=sys.stderr)
                continue

            if not event or not looks_like_2mile_event(event):
                if verbose:
                    en = extract_event_name(event)
                    print(f"[SKIP] Not 2 Mile: race_id={race_id} event_id={event_id} event='{en}' race='{race_name}'",
                          file=sys.stderr)
                continue

            event_name = extract_event_name(event)

            try:
                splits_payload = client.get_result_set_splits(race_id, event_id, result_set_id)
                split_defs = parse_splits_response(splits_payload)
                mile_to_split_id = map_mile_splits(split_defs, max_mile=MAX_MILE)
            except Exception as e:
                if verbose:
                    print(f"[WARN] splits fetch/parse failed for race_id={race_id} event_id={event_id} rs_id={result_set_id}: {e}",
                          file=sys.stderr)
                continue

            if len(mile_to_split_id) == 0:
                if verbose:
                    print(f"[SKIP] No splits found: race_id={race_id} event_id={event_id} rs_id={result_set_id}",
                          file=sys.stderr)
                continue

            for p in range(1, max_pages_per_result_set + 1):
                try:
                    results_payload = client.get_event_results(
                        race_id=race_id,
                        event_id=event_id,
                        individual_result_set_id=result_set_id,
                        page=p,
                        results_per_page=results_per_page,
                        include_split_time_ms=False,
                    )
                except Exception as e:
                    if verbose:
                        print(f"[WARN] get_results failed race_id={race_id} event_id={event_id} rs_id={result_set_id} page={p}: {e}",
                              file=sys.stderr)
                    break

                any_rows = False
                for rs_meta, row in iter_results_from_get_results(results_payload):
                    any_rows = True

                    finish_raw, finish_s = pick_finish_time(row)
                    if finish_s is None:
                        continue

                    mile_raw: Dict[int, Optional[str]] = {1: None, 2: None, 3: None}
                    mile_s: Dict[int, Optional[int]] = {1: None, 2: None, 3: None}

                    for mile in range(1, MAX_MILE + 1):
                        sid = mile_to_split_id.get(mile)
                        if sid is None:
                            continue
                        raw = extract_split_time(row, sid)
                        mile_raw[mile] = raw
                        mile_s[mile] = time_str_to_seconds(raw) if raw else None

                    if require_all_miles_present:
                        if any(mile_s[m] is None for m in range(1, MAX_MILE + 1)):
                            continue

                    rows_out.append({
                        "race_id": race_id,
                        "event_id": event_id,
                        "individual_result_set_id": result_set_id,
                        "race_name": race_name,
                        "event_name": event_name,
                        "individual_result_set_name": result_set_name,
                        "result_id": row.get("result_id"),
                        "place": row.get("place"),
                        "gender": row.get("gender"),
                        "age": row.get("age"),
                        "finish_time_s": finish_s,
                        "mile1_s": mile_s.get(1),
                        "mile2_s": mile_s.get(2),
                        "mile3_s": mile_s.get(3),
                        "finish_time_raw": finish_raw,
                        "mile1_raw": mile_raw.get(1),
                        "mile2_raw": mile_raw.get(2),
                        "mile3_raw": mile_raw.get(3),
                    })

                if not any_rows:
                    break

        page += 1

    return rows_out


def write_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    header = [
        "race_id","event_id","individual_result_set_id",
        "race_name","event_name","individual_result_set_name",
        "result_id","place","gender","age",
        "finish_time_s","mile1_s","mile2_s","mile3_s",
        "finish_time_raw","mile1_raw","mile2_raw","mile3_raw",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main() -> int:
    parser = argparse.ArgumentParser(description="Scrape RunSignup 2 Mile results with Mile 1/2 splits into a CSV.")
    parser.add_argument("--days-back", type=int, default=30)
    parser.add_argument("--max-result-sets", type=int, default=200)
    parser.add_argument("--results-per-page", type=int, default=250)
    parser.add_argument("--max-pages-per-result-set", type=int, default=3)
    parser.add_argument("--require-all-miles-present", action="store_true",
                        help="Keep only rows that contain required miles for this race distance (here: Mile 1 & 2).")
    parser.add_argument("--out", type=str, default="runsignup_2mile_splits.csv")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    api_key = os.environ.get("RUNSIGNUP_API_KEY", "").strip()
    api_secret = os.environ.get("RUNSIGNUP_API_SECRET", "").strip()
    if not api_key or not api_secret:
        print(
            "ERROR: Missing API credentials.\n"
            "Set environment variables:\n"
            "  RUNSIGNUP_API_KEY\n"
            "  RUNSIGNUP_API_SECRET\n",
            file=sys.stderr,
        )
        return 2

    client = RunSignupClient(api_key=api_key, api_secret=api_secret)

    rows = scrape(
        client=client,
        days_back=args.days_back,
        max_result_sets=args.max_result_sets,
        results_per_page=args.results_per_page,
        max_pages_per_result_set=args.max_pages_per_result_set,
        require_all_miles_present=args.require_all_miles_present,
        verbose=args.verbose,
    )

    write_csv(args.out, rows)
    print(f"Wrote {len(rows)} rows to {args.out}")
    if len(rows) == 0:
        print(
            "Tip: Try increasing --days-back and --max-result-sets, or remove --require-all-miles-present.\n"
            "Not all races publish splits, even if they publish results.",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
