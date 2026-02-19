#!/usr/bin/env python3
# tune_helper.py – small CLI for fingerprints and cache hydration.

import argparse
import json
import os
import sys
import csv

import profile
from tune import buildFingerprint
from cache import (
    RESULT_FIELD_NAMES,
    getKlinesCached,
    profile_windows as profileWindows,
    hydrateResultsCache,
    mergeGlobalResults,
)


def _load_profile(path: str) -> dict:
    return profile.loadJson(path)


def _cmd_fingerprint(args) -> int:
    cfg = _load_profile(args.profile)
    fp = buildFingerprint(cfg)
    if args.out:
        with open(args.out, 'w') as fh:
            json.dump(fp, fh, indent=2)
    if args.compare:
        prev = profile.loadJson(args.compare)
        if fp == prev:
            return 0
        return 1
    return 0


def _cmd_hydrate(args) -> int:
    cfg = _load_profile(args.profile)
    tickersList = profile._requireTickers(cfg)
    primer, _tuner, holdout, _total = profileWindows(cfg)
    hydrateResultsCache(
        args.results,
        primer,
        holdout,
        getKlinesCached,
        tickers=tickersList,
        maxRows=None,
    )
    return 0


def _cmd_compare(args) -> int:
    a = profile.loadJson(args.a)
    b = profile.loadJson(args.b)
    return 0 if a == b else 1


def _read_global(global_path: str) -> tuple[list[dict], set[str]]:
    rows: list[dict] = []
    seen: set[str] = set()
    if not os.path.exists(global_path):
        return rows, seen
    with open(global_path) as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            rows.append(row)
            sh = row.get("specHash")
            if sh:
                seen.add(str(sh))
    return rows, seen


def _write_global(global_path: str, rows: list[dict]) -> None:
    os.makedirs(os.path.dirname(global_path), exist_ok=True)
    fieldnames = list(RESULT_FIELD_NAMES) + ["specHash"]
    tmp = f"{global_path}.tmp"
    with open(tmp, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow(row)
    os.replace(tmp, global_path)


def _backfill_cache_results(cache_root: str, global_path: str) -> None:
    base = os.path.join(cache_root, "results")
    rows, seen = _read_global(global_path)
    for dirpath, _dirs, files in os.walk(base):
        for name in files:
            if not name.endswith(".json"):
                continue
            spec_hash = os.path.splitext(name)[0]
            if spec_hash in seen:
                continue
            path = os.path.join(dirpath, name)
            try:
                data = profile.loadJson(path)
                row = data.get("row", {})
                if not isinstance(row, dict):
                    continue
                row_clean = {
                    k: row[k]
                    for k in RESULT_FIELD_NAMES
                    if k in row
                }
                if not row_clean:
                    continue
                row_clean["specHash"] = spec_hash
                rows.append(row_clean)
                seen.add(spec_hash)
            except Exception as exc:  # noqa: BLE001
                print(
                    "[tune] skipping cache result "
                    f"{os.path.basename(path)}: {exc}",
                    file=sys.stderr,
                )
                continue
    _write_global(global_path, rows)


def _cmd_backfill(args) -> int:
    root = os.path.abspath(args.root)
    runs_dir = os.path.join(root, "outputs", "tuning")
    cache_root = os.environ.get(
        "GRADBOT_CACHE_DIR", os.path.join(root, "cache")
    )
    global_path = os.path.join(cache_root, "results", "global_results.csv")
    if not os.path.isdir(runs_dir):
        return 0
    for name in os.listdir(runs_dir):
        run_dir = os.path.join(runs_dir, name)
        if not os.path.isdir(run_dir):
            continue
        fp_path = os.path.join(run_dir, "fingerprint.json")
        csv_path = os.path.join(run_dir, "results.csv")
        if not os.path.exists(fp_path) or not os.path.exists(csv_path):
            continue
        try:
            fp = profile.loadJson(fp_path)
            primer = int(fp.get("primerDays", 0))
            holdout = int(fp.get("holdoutDays", 0))
            mergeGlobalResults(
                csv_path,
                primer,
                holdout,
                getKlinesCached,
            )
            print(
                f"[tune] backfilled global results from "
                f"{os.path.basename(run_dir)}"
            )
        except Exception as exc:  # noqa: BLE001
            print(
                "[tune] skipping "
                f"{os.path.basename(run_dir)}: {exc}",
                file=sys.stderr,
            )
            continue
    _backfill_cache_results(cache_root, global_path)
    return 0


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="tune_helper",
        description="Fingerprint and cache hydration helpers",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    fp_parser = sub.add_parser("fingerprint", help="Build/compare fingerprint")
    fp_parser.add_argument(
        "--profile",
        required=True,
        help="Path to tuner profile JSON",
    )
    fp_parser.add_argument(
        "--out",
        help="Optional path to write the computed fingerprint JSON",
    )
    fp_parser.add_argument(
        "--compare",
        help="Compare against an existing fingerprint JSON",
    )

    hyd_parser = sub.add_parser(
        "hydrate", help="Hydrate cache/results from results.csv",
    )
    hyd_parser.add_argument(
        "--profile",
        required=True,
        help="Path to tuner profile JSON",
    )
    hyd_parser.add_argument(
        "--results",
        required=True,
        help="Path to tuner results CSV",
    )

    cmp_parser = sub.add_parser(
        "compare", help="Compare two fingerprint JSON files",
    )
    cmp_parser.add_argument(
        "--a",
        required=True,
        help="First fingerprint path",
    )
    cmp_parser.add_argument(
        "--b",
        required=True,
        help="Second fingerprint path",
    )

    bf_parser = sub.add_parser(
        "backfill", help="Merge legacy run results into global cache",
    )
    bf_parser.add_argument(
        "--root",
        required=True,
        help="Repo root containing outputs/tuning",
    )

    args = parser.parse_args(argv)
    if args.cmd == "fingerprint":
        return _cmd_fingerprint(args)
    if args.cmd == "hydrate":
        return _cmd_hydrate(args)
    if args.cmd == "compare":
        return _cmd_compare(args)
    if args.cmd == "backfill":
        return _cmd_backfill(args)
    return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
