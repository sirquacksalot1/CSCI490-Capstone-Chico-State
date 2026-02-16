#!/usr/bin/env python3
"""
runsignup_5k_splits_scraper.py

Goal:
  Build a CSV dataset of real race results with:
    - 5K finish time
    - Mile 1 / Mile 2 / Mile 3 split times (when available)

Data sources (RunSignup API):
  - Updated public result sets:
      GET https://api.runsignup.com/rest/v2/results/updated-result-sets.json
  - Event results (includes dynamic keys like split-<id>):
      GET https://api.runsignup.com/rest/race/:race_id/results/get-results
  - Result-set splits (to map split IDs to mile markers):
      GET https://api.runsignup.com/rest/race/:race_id/results/result-set-splits
  - Race details (to identify which events are likely 5Ks):
      GET https://api.runsignup.com/rest/race/:race_id

Auth:
  - Supports OAuth2 Bearer tokens (preferred):
      Set RUNSIGNUP_ACCESS_TOKEN and it will send:
        Authorization: Bearer <token>
  - Optional legacy fallback (query params):
      RUNSIGNUP_API_KEY, RUNSIGNUP_API_SECRET

Example usage (OAuth2):
  export RUNSIGNUP_ACCESS_TOKEN="eyJ...."

  python runsignup_5k_splits_scraper.py \
    --days-back 14 \
    --max-result-sets 50 \
    --out results_5k_splits.csv

Example usage (legacy fallback):
  export RUNSIGNUP_API_KEY="xxx"
  export RUNSIGNUP_API_SECRET="yyy"

Notes:
  - Many result sets will not be 5K, and many 5Ks won't have mile splits.
  - This script uses heuristics to detect a 5K event:
      - event_name contains "5K" OR
      - event_type equals/contains "5K" OR
      - distance is close to 5.0 km (if provided)
  - Split mapping:
      - Results payload includes dynamic fields named like split-<id>
      - We call result-set-splits to determine which split corresponds to Mile 1/2/3
      - If distance metadata is missing, fall back to split name matching ("Mile 1", "1 Mile", etc.)

Output columns:
  race_id, event_id, individual_result_set_id,
  race_name, event_name, individual_result_set_name,
  result_id, place, gender, age,
  finish_time_s, mile1_s, mile2_s, mile3_s,
  finish_time_raw, mile1_raw, mile2_raw, mile3_raw
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


# ----------------------------
# Utilities: time parsing
# ----------------------------

_TIME_RE = re.compile(r"^\s*(?:(\d+):)?(\d{1,2}):(\d{2})(?:\.\d+)?\s*$")  # [H:]MM:SS(.ms)


def time_str_to_seconds(s: str) -> Optional[int]:
    """
    Convert time strings like:
      - "18:34" -> 1114
      - "1:02:03" -> 3723
      - "00:19:45.123" -> 1185 (ms ignored)
    Returns None for blanks or non-times like "NONE".
    """
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


# ----------------------------
# HTTP client (OAuth2 + legacy fallback)
# ----------------------------

@dataclass
class RunSignupClient:
    # OAuth2 (preferred)
    access_token: Optional[str] = None

    # Legacy fallback
    api_key: Optional[str] = None
    api_secret: Optional[str] = None

    timeout_s: int = 30
    sleep_s: float = 0.15  # gentle pacing

    def _get(self, path: str, params: Dict[str, Any]) -> Dict[str, Any]:
        url = BASE_URL + path

        qp = dict(params)
        qp.setdefault("format", "json")

        headers: Dict[str, str] = {}

        # Prefer OAuth2 Bearer token if provided
        if self.access_token:
            headers["Authorization"] = f"Bearer {self.access_token}"
        else:
            # Fallback to legacy key/secret if provided
            if not self.api_key or not self.api_secret:
                raise RuntimeError(
                    "Missing auth. Set RUNSIGNUP_ACCESS_TOKEN (OAuth2) "
                    "or RUNSIGNUP_API_KEY and RUNSIGNUP_API_SECRET (legacy)."
                )
            qp["api_key"] = self.api_key
            qp["api_secret"] = self.api_secret

        resp = requests.get(url, params=qp, headers=headers, timeout=self.timeout_s)
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
        params: Dict[str, Any] = {
            "page": page,
            "num_per_page": num_per_page,
        }
        if modified_since_ts is not None:
            params["modified_since_timestamp"] = modified_since_ts
        if modified_until_ts is not None:
            params["modified_until_timestamp"] = modified_until_ts

        data = self._get(UPDATED_RESULT_SETS_PATH, params)
        return data.get("result_sets", []) or []

    def get_race(self, race_id: int) -> Dict[str, Any]:
        # Get Race API; by default includes events (and past events) in the response docs
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

    def get_result_set_splits(
        self,
        race_id: int,
        event_id: int,
        individual_result_set_id: int,
    ) -> Dict[str, Any]:
        params = {
            "race_id": race_id,
            "event_id": event_id,
            "individual_result_set_id": individual_result_set_id,
        }
        return self._get(GET_SPLITS_PATH.format(race_id=race_id), params=params)


# ----------------------------
# Parsing helpers
# ----------------------------

def safe_lower(x: Any) -> str:
    return str(x or "").lower()


def looks_like_5k_event(event: Dict[str, Any]) -> bool:
    """
    Heuristics to decide if an event is a 5K.
    We try several common fields RunSignup uses (varies by endpoint/version).
    """
    name = safe_lower(event.get("name") or event.get("event_name"))
    event_type = safe_lower(event.get("event_type"))
    # Distance sometimes appears as a number + unit
    dist = event.get("distance") or event.get("event_distance") or event.get("length")
    unit = safe_lower(event.get("distance_unit") or event.get("unit") or event.get("event_distance_unit"))

    # Name/type checks
    if "5k" in name or event_type == "5k" or "5k" in event_type:
        # avoid obvious kid fun runs mislabeled in name
        if any(bad in name for bad in ["kid", "kids", "children", "tot", "toddler"]):
            return False
        return True

    # Distance check: near 5.0 km or 3.1 miles
    try:
        if dist is not None:
            d = float(dist)
            if "km" in unit:
                return abs(d - 5.0) <= 0.25
            if "mi" in unit or "mile" in unit:
                return abs(d - 3.1069) <= 0.2
    except Exception:
        pass

    return False


def extract_event_name(event: Dict[str, Any]) -> str:
    return str(event.get("name") or event.get("event_name") or "").strip()


def find_event_by_id(race_payload: Dict[str, Any], event_id: int) -> Optional[Dict[str, Any]]:
    """
    Race response structure can vary. Common patterns:
      - {"race": {"events": [{"event_id": ...}, ...]}}
      - {"race": {"events": [{"event": {...}}]}}   <-- wrapper
      - {"race": {"event": [...]}}
      - {"events": [...]}
    """
    candidates: List[Any] = []

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
        if not isinstance(e, dict):
            continue

        # unwrap {"event": {...}} pattern if present
        e2 = e.get("event") if isinstance(e.get("event"), dict) else e

        try:
            if int(e2.get("event_id") or e2.get("id")) == int(event_id):
                return e2
        except Exception:
            continue

    return None


def iter_results_from_get_results(payload: Dict[str, Any]) -> Iterable[Tuple[Dict[str, Any], Dict[str, Any]]]:
    """
    Get Event Results returns one or more "individual_results_sets", each containing "results".
    Yield (result_set_meta, result_row).
    """
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
    """
    Find a list-like field that contains dict splits.
    """
    if not isinstance(payload, dict):
        return []

    for key in ("splits", "result_set_splits", "result_sets_splits", "result_set_split"):
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


def map_mile_splits(split_defs: List[Dict[str, Any]]) -> Dict[int, int]:
    entries = []
    for sd in split_defs:
        sid = split_def_id(sd)
        if sid is None:
            continue
        name = split_def_name(sd).strip()
        dist, unit = split_def_distance(sd)
        entries.append((sid, name, dist, unit))

    mile_map: Dict[int, int] = {}

    for target_mile in (1, 2, 3):
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

    if len(mile_map) < 3:
        for target_mile in (1, 2, 3):
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


# ----------------------------
# Main scrape logic
# ----------------------------

def unix_ts_days_ago(days: int) -> int:
    now = dt.datetime.utcnow()
    past = now - dt.timedelta(days=days)
    return int(past.timestamp())


def pick_finish_time(row: Dict[str, Any]) -> Tuple[Optional[str], Optional[int]]:
    """
    Prefer chip_time if present, else clock_time.
    """
    chip = row.get("chip_time")
    clock = row.get("clock_time")
    raw = None
    if chip and str(chip).strip() and str(chip).strip().upper() != "NONE":
        raw = str(chip).strip()
    elif clock and str(clock).strip() and str(clock).strip().upper() != "NONE":
        raw = str(clock).strip()

    # FIX: avoid returning (None, None) as the seconds value
    return raw, (time_str_to_seconds(raw) if raw else None)


def scrape(
    client: RunSignupClient,
    days_back: int,
    max_result_sets: int,
    results_per_page: int,
    max_pages_per_result_set: int,
    require_all_three_miles: bool,
    verbose: bool,
) -> List[Dict[str, Any]]:
    rows_out: List[Dict[str, Any]] = []

    modified_since = unix_ts_days_ago(days_back)

    page = 1
    num_per_page = min(5000, max_result_sets) if max_result_sets > 0 else 5000
    seen = 0

    while seen < max_result_sets:
        result_sets = client.get_updated_result_sets(
            page=page,
            num_per_page=num_per_page,
            modified_since_ts=modified_since,
            modified_until_ts=None,
        )
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

                # DEBUG: show payload structure
                if verbose:
                    print(
                        f"[DEBUG] race_id={race_id} top_keys={list(race_payload.keys())[:30]}",
                        file=sys.stderr,
                    )
                    if "race" in race_payload and isinstance(race_payload["race"], dict):
                        print(
                            f"[DEBUG] race['keys']={list(race_payload['race'].keys())[:30]}",
                            file=sys.stderr,
                        )
                    print(
                        f"[DEBUG] race_payload snippet={str(race_payload)[:500]}",
                        file=sys.stderr,
                    )

                event = find_event_by_id(race_payload, event_id) or {}
            except Exception as e:
                if verbose:
                    print(f"[WARN] get_race failed for race_id={race_id}: {e}", file=sys.stderr)
                continue

            if not event or not looks_like_5k_event(event):
                if verbose:
                    en = extract_event_name(event)
                    print(
                        f"[SKIP] Not 5K: race_id={race_id} event_id={event_id} event='{en}' race='{race_name}'",
                        file=sys.stderr,
                    )
                continue

            event_name = extract_event_name(event)

            try:
                splits_payload = client.get_result_set_splits(race_id, event_id, result_set_id)
                split_defs = parse_splits_response(splits_payload)
                mile_to_split_id = map_mile_splits(split_defs)
            except Exception as e:
                if verbose:
                    print(
                        f"[WARN] splits fetch/parse failed for race_id={race_id} event_id={event_id} rs_id={result_set_id}: {e}",
                        file=sys.stderr,
                    )
                continue

            if len(mile_to_split_id) == 0:
                if verbose:
                    print(
                        f"[SKIP] No splits found: race_id={race_id} event_id={event_id} rs_id={result_set_id}",
                        file=sys.stderr,
                    )
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
                        print(
                            f"[WARN] get_results failed race_id={race_id} event_id={event_id} rs_id={result_set_id} page={p}: {e}",
                            file=sys.stderr,
                        )
                    break

                any_rows = False
                for rs_meta, row in iter_results_from_get_results(results_payload):
                    any_rows = True

                    finish_raw, finish_s = pick_finish_time(row)
                    if finish_s is None:
                        continue

                    mile_raw: Dict[int, Optional[str]] = {}
                    mile_s: Dict[int, Optional[int]] = {}

                    for mile in (1, 2, 3):
                        sid = mile_to_split_id.get(mile)
                        if sid is None:
                            mile_raw[mile] = None
                            mile_s[mile] = None
                            continue
                        raw = extract_split_time(row, sid)
                        mile_raw[mile] = raw
                        mile_s[mile] = time_str_to_seconds(raw) if raw else None

                    if require_all_three_miles and any(mile_s[m] is None for m in (1, 2, 3)):
                        continue

                    out = {
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
                    }
                    rows_out.append(out)

                if not any_rows:
                    break

        page += 1

    return rows_out


def write_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    header = [
        "race_id", "event_id", "individual_result_set_id",
        "race_name", "event_name", "individual_result_set_name",
        "result_id", "place", "gender", "age",
        "finish_time_s", "mile1_s", "mile2_s", "mile3_s",
        "finish_time_raw", "mile1_raw", "mile2_raw", "mile3_raw",
    ]

    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main() -> int:
    parser = argparse.ArgumentParser(description="Scrape RunSignup 5K results with mile splits into a CSV.")
    parser.add_argument("--days-back", type=int, default=30, help="Look for updated public result sets modified in the last N days.")
    parser.add_argument("--max-result-sets", type=int, default=200, help="Max number of updated result sets to inspect.")
    parser.add_argument("--results-per-page", type=int, default=250, help="Results per page when pulling an event result set.")
    parser.add_argument("--max-pages-per-result-set", type=int, default=3, help="Max pages to pull per result set (pagination).")
    parser.add_argument("--require-all-three-miles", action="store_true", help="Keep only rows that contain Mile 1, Mile 2, and Mile 3.")
    parser.add_argument("--out", type=str, default="runsignup_5k_splits.csv", help="Output CSV file path.")
    parser.add_argument("--verbose", action="store_true", help="Print progress + skips to stderr.")
    args = parser.parse_args()

    access_token = os.environ.get("RUNSIGNUP_ACCESS_TOKEN", "").strip()
    api_key = os.environ.get("RUNSIGNUP_API_KEY", "").strip()
    api_secret = os.environ.get("RUNSIGNUP_API_SECRET", "").strip()

    if not access_token and (not api_key or not api_secret):
        print(
            "ERROR: Missing API credentials.\n"
            "Set either:\n"
            "  RUNSIGNUP_ACCESS_TOKEN  (OAuth2 Bearer token)\n"
            "OR (legacy fallback):\n"
            "  RUNSIGNUP_API_KEY\n"
            "  RUNSIGNUP_API_SECRET\n",
            file=sys.stderr,
        )
        return 2

    client = RunSignupClient(
        access_token=access_token or None,
        api_key=api_key or None,
        api_secret=api_secret or None,
    )

    rows = scrape(
        client=client,
        days_back=args.days_back,
        max_result_sets=args.max_result_sets,
        results_per_page=args.results_per_page,
        max_pages_per_result_set=args.max_pages_per_result_set,
        require_all_three_miles=args.require_all_three_miles,
        verbose=args.verbose,
    )

    write_csv(args.out, rows)

    print(f"Wrote {len(rows)} rows to {args.out}")
    if len(rows) == 0:
        print(
            "Tip: Try increasing --days-back and --max-result-sets, or remove --require-all-three-miles.\n"
            "Not all races publish splits, even if they publish results.",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

