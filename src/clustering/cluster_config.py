#!/usr/bin/env python3

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PeriodConfig:
    fast: int
    mid: int
    slow: int


@dataclass(frozen=True)
class ExploreConfig:
    returnBars: list[int]
    emaSlopeBars: int
    rsiBars: int
    volumeZBars: int
    derivativeClip: float
    includeRsi: bool


@dataclass(frozen=True)
class ClusterConfig:
    name: str
    ticker: str
    interval: str
    days: int
    anchorMs: int | None
    policyStartMs: int | None
    policyEndMs: int | None
    windowBars: int
    clusters: list[int]
    randomState: int
    fitFraction: float
    chartBars: int
    yearlyCharts: bool
    views: list[str]
    featureFamilies: list[str]
    clusterMethods: list[str]
    forwardBars: list[int]
    policyTarget: int
    periods: PeriodConfig
    periodCombos: list[PeriodConfig]
    explore: ExploreConfig
    engine: dict[str, object]


def _requireMap(raw: dict, key: str) -> dict:
    return raw[key]


def _requireList(raw: dict, key: str) -> list:
    return raw[key]


def loadConfig(path: str | Path) -> ClusterConfig:
    with open(Path(path), "r") as fh:
        raw = json.load(fh)

    periods = _requireMap(raw, "periods")
    combosRaw = raw.get("periodCombos", [periods])
    explore = _requireMap(raw, "explore")
    clustersRaw = raw["clusters"]
    clusters = (
        [int(item) for item in clustersRaw]
        if isinstance(clustersRaw, list)
        else [int(clustersRaw)]
    )
    forwardBars = [int(item) for item in _requireList(raw, "forwardBars")]

    return ClusterConfig(
        name=str(raw["name"]),
        ticker=str(raw["ticker"]).upper(),
        interval=str(raw["interval"]),
        days=int(raw["days"]),
        anchorMs=(
            int(raw["anchorMs"])
            if raw.get("anchorMs") is not None
            else None
        ),
        policyStartMs=(
            int(raw["policyStartMs"])
            if raw.get("policyStartMs") is not None
            else None
        ),
        policyEndMs=(
            int(raw["policyEndMs"])
            if raw.get("policyEndMs") is not None
            else None
        ),
        windowBars=int(raw["windowBars"]),
        clusters=clusters,
        randomState=int(raw["randomState"]),
        fitFraction=float(raw["fitFraction"]),
        chartBars=int(raw["chartBars"]),
        yearlyCharts=bool(raw["yearlyCharts"]),
        views=[str(item) for item in _requireList(raw, "views")],
        featureFamilies=[
            str(item) for item in _requireList(raw, "featureFamilies")
        ],
        clusterMethods=[
            str(item) for item in raw.get("clusterMethods", ["kmeans"])
        ],
        forwardBars=forwardBars,
        policyTarget=int(raw.get("policyTarget", forwardBars[-1])),
        periods=PeriodConfig(
            fast=int(periods["fast"]),
            mid=int(periods["mid"]),
            slow=int(periods["slow"]),
        ),
        periodCombos=[
            PeriodConfig(
                fast=int(item["fast"]),
                mid=int(item["mid"]),
                slow=int(item["slow"]),
            )
            for item in combosRaw
        ],
        explore=ExploreConfig(
            returnBars=[
                int(item) for item in _requireList(explore, "returnBars")
            ],
            emaSlopeBars=int(explore["emaSlopeBars"]),
            rsiBars=int(explore["rsiBars"]),
            volumeZBars=int(explore["volumeZBars"]),
            derivativeClip=float(explore["derivativeClip"]),
            includeRsi=bool(explore["includeRsi"]),
        ),
        engine=dict(_requireMap(raw, "engine")),
    )
