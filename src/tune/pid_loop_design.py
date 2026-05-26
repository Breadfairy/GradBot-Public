#!/usr/bin/env python3
"""Generate PID frequency and loop-design notes for peak lock."""

from __future__ import annotations

import argparse
import cmath
import csv
import json
import math
import os
from pathlib import Path
from typing import Any

from repo_paths import rootPath


###############################################################################
# Constants
###############################################################################

FREQ_POINTS = 240
MAX_PERIOD_DAYS = 365.0
PLANT_GAIN = 1.0


###############################################################################
# Profile Helpers
###############################################################################

def loadProfile(path: Path) -> dict[str, Any]:
    with open(path) as fh:
        return json.load(fh)


def scalarValue(config: dict[str, Any], key: str, default: float) -> float:
    raw = config.get(key, default)
    if isinstance(raw, list):
        raw = raw[0]
    return float(raw)


def scalarInt(config: dict[str, Any], key: str, default: int) -> int:
    raw = config.get(key, default)
    if isinstance(raw, list):
        raw = raw[0]
    return int(raw)


def scalarText(config: dict[str, Any], key: str, default: str) -> str:
    raw = config.get(key, default)
    if isinstance(raw, list):
        raw = raw[0]
    return str(raw)


def barsPerDay(interval: str) -> float:
    unit = interval[-1]
    size = float(interval[:-1])
    bars = 24.0
    if unit == "m":
        bars = 1440.0 / size
    elif unit == "h":
        bars = 24.0 / size
    elif unit == "d":
        bars = 1.0 / size
    return bars


###############################################################################
# Transfer Functions
###############################################################################

def safeDb(value: complex) -> float:
    mag = abs(value)
    return 20.0 * math.log10(max(mag, 1e-12))


def phaseDeg(value: complex) -> float:
    return math.degrees(cmath.phase(value))


def safeDiv(top: complex, bottom: complex) -> complex:
    if abs(bottom) < 1e-12:
        return complex(0.0, 0.0)
    return top / bottom


def zPoint(omega: float, sampleDays: float) -> complex:
    return cmath.exp(complex(0.0, omega * sampleDays))


def emaHighpass(zVal: complex, alpha: float) -> complex:
    beta = 1.0 - alpha
    zInv = safeDiv(1.0, zVal)
    return safeDiv(beta * (1.0 - zInv), 1.0 - (beta * zInv))


def pidDiscrete(
    zVal: complex,
    kp: float,
    ki: float,
    kd: float,
    decay: float,
) -> complex:
    zInv = safeDiv(1.0, zVal)
    prop = complex(kp, 0.0)
    integ = safeDiv(ki, 1.0 - (decay * zInv))
    deriv = kd * (1.0 - zInv)
    return prop + integ + deriv


def emaTauDays(beta: float, sampleDays: float) -> float:
    if beta <= 0.0:
        return sampleDays
    return -sampleDays / math.log(beta)


def highpassCont(omega: float, tauDays: float) -> complex:
    sVal = complex(0.0, omega)
    return safeDiv(tauDays * sVal, 1.0 + (tauDays * sVal))


def pidCont(
    omega: float,
    sampleDays: float,
    kp: float,
    ki: float,
    kd: float,
    decay: float,
) -> complex:
    sVal = complex(0.0, omega)
    prop = complex(kp, 0.0)
    integ = safeDiv(ki, (1.0 - decay) + (decay * sampleDays * sVal))
    deriv = kd * sampleDays * sVal
    return prop + integ + deriv


def plantCont(
    omega: float,
    tauDays: float,
    delayDays: float,
) -> complex:
    sVal = complex(0.0, omega)
    lag = safeDiv(PLANT_GAIN, 1.0 + (tauDays * sVal))
    delay = cmath.exp(-sVal * delayDays)
    return lag * delay


###############################################################################
# Report Helpers
###############################################################################

def unwrapPhase(values: list[float]) -> list[float]:
    unwrapped = []
    previous = 0.0
    offset = 0.0
    for i, value in enumerate(values):
        current = value
        if i > 0:
            delta = current - previous
            if delta > 180.0:
                offset -= 360.0
            elif delta < -180.0:
                offset += 360.0
        unwrapped.append(current + offset)
        previous = current
    return unwrapped


def marginEstimate(rows: list[dict[str, float]]) -> dict[str, float]:
    phases = [r["loopPhaseDeg"] for r in rows]
    unwrapped = unwrapPhase(phases)
    bestIdx = 0
    bestErr = float("inf")
    for i, r in enumerate(rows):
        err = abs(r["loopMag"] - 1.0)
        if err < bestErr:
            bestErr = err
            bestIdx = i
    return {
        "gainCrossPeriodDays": rows[bestIdx]["periodDays"],
        "gainCrossMagDb": rows[bestIdx]["loopMagDb"],
        "phaseMarginDeg": 180.0 + unwrapped[bestIdx],
    }


def frequencyRows(config: dict[str, Any]) -> tuple[list[dict[str, float]], dict]:
    interval = scalarText(config, "intervals", "1h")
    barDay = barsPerDay(interval)
    sampleDays = 1.0 / barDay
    maDays = scalarValue(config, "PEAK_LOCK_MA_DAYS", 30.0)
    kp = scalarValue(config, "PEAK_LOCK_KP", 6.0)
    ki = scalarValue(config, "PEAK_LOCK_KI", 0.0)
    kd = scalarValue(config, "PEAK_LOCK_KD", 0.0)
    decay = scalarValue(config, "PEAK_LOCK_INTEGRAL_DECAY", 0.985)
    entry = scalarValue(config, "PEAK_LOCK_ENTRY_THRESHOLD", 0.25)
    exitVal = scalarValue(config, "PEAK_LOCK_EXIT_THRESHOLD", 0.05)
    confirm = scalarInt(config, "PEAK_LOCK_CONFIRM_BARS", 6)
    maBars = max(2, int(round(maDays * barDay)))
    alpha = 2.0 / float(maBars + 1)
    beta = 1.0 - alpha
    tauMa = emaTauDays(beta, sampleDays)
    tauPlant = max(float(confirm) * sampleDays, sampleDays)
    delayPlant = max(0.5 * float(confirm) * sampleDays, sampleDays)
    minPeriod = max(2.0 * sampleDays, sampleDays * 3.0)
    logMin = math.log(minPeriod)
    logMax = math.log(MAX_PERIOD_DAYS)
    rows = []
    for i in range(FREQ_POINTS):
        frac = float(i) / float(FREQ_POINTS - 1)
        period = math.exp(logMax + ((logMin - logMax) * frac))
        omega = (2.0 * math.pi) / period
        zVal = zPoint(omega, sampleDays)
        hpZ = emaHighpass(zVal, alpha)
        pidZ = pidDiscrete(zVal, kp, ki, kd, decay)
        detZ = hpZ * pidZ
        hpS = highpassCont(omega, tauMa)
        pidS = pidCont(omega, sampleDays, kp, ki, kd, decay)
        plantS = plantCont(omega, tauPlant, delayPlant)
        loopS = hpS * pidS * plantS
        closedS = safeDiv(loopS, 1.0 + loopS)
        sensS = safeDiv(1.0, 1.0 + loopS)
        rows.append({
            "periodDays": period,
            "omegaRadDay": omega,
            "discHighDb": safeDb(hpZ),
            "discPidDb": safeDb(pidZ),
            "discDetectorDb": safeDb(detZ),
            "contHighDb": safeDb(hpS),
            "contPidDb": safeDb(pidS),
            "loopMag": abs(loopS),
            "loopMagDb": safeDb(loopS),
            "loopPhaseDeg": phaseDeg(loopS),
            "closedMagDb": safeDb(closedS),
            "sensitivityDb": safeDb(sensS),
        })
    params = {
        "interval": interval,
        "barsPerDay": barDay,
        "sampleDays": sampleDays,
        "maDays": maDays,
        "maBars": maBars,
        "alpha": alpha,
        "beta": beta,
        "tauMaDays": tauMa,
        "kp": kp,
        "ki": ki,
        "kd": kd,
        "decay": decay,
        "entryThreshold": entry,
        "exitThreshold": exitVal,
        "confirmBars": confirm,
        "confirmDays": float(confirm) * sampleDays,
        "plantTauDays": tauPlant,
        "plantDelayDays": delayPlant,
        "entryErrorPct": 100.0 * safeDiv(entry, kp).real,
        "exitErrorPct": 100.0 * safeDiv(exitVal, kp).real,
    }
    return rows, params


def atomicText(path: Path, text: str) -> None:
    tmpPath = path.with_suffix(path.suffix + ".tmp")
    with open(tmpPath, "w") as fh:
        fh.write(text)
    os.replace(tmpPath, path)


def atomicJson(path: Path, data: dict[str, Any]) -> None:
    tmpPath = path.with_suffix(path.suffix + ".tmp")
    with open(tmpPath, "w") as fh:
        json.dump(data, fh, indent=2)
        fh.write("\n")
    os.replace(tmpPath, path)


def writeCsv(path: Path, rows: list[dict[str, float]]) -> None:
    fieldNames = list(rows[0].keys())
    tmpPath = path.with_suffix(path.suffix + ".tmp")
    with open(tmpPath, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldNames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmpPath, path)


def reportText(params: dict[str, float], margins: dict[str, float]) -> str:
    lines = [
        "# PID Loop Design",
        "",
        "This report models the current peak-lock PID as a detector that "
        "controls wallet exposure, not market price.",
        "",
        "## Current Parameters",
        "",
        f"- interval: `{params['interval']}`",
        f"- bars/day: `{params['barsPerDay']:.6g}`",
        f"- sample time: `{params['sampleDays']:.6g}` days",
        f"- MA days: `{params['maDays']:.6g}`",
        f"- MA bars: `{params['maBars']}`",
        f"- EMA alpha: `{params['alpha']:.12g}`",
        f"- EMA continuous tau: `{params['tauMaDays']:.6g}` days",
        f"- Kp/Ki/Kd: `{params['kp']:.6g}`, "
        f"`{params['ki']:.6g}`, `{params['kd']:.6g}`",
        f"- integral decay: `{params['decay']:.6g}`",
        f"- entry/exit raw thresholds: "
        f"`{params['entryThreshold']:.6g}`, "
        f"`{params['exitThreshold']:.6g}`",
        f"- proportional-only entry error: "
        f"`{params['entryErrorPct']:.6g}%` above EMA",
        f"- proportional-only exit error: "
        f"`{params['exitErrorPct']:.6g}%` above EMA",
        f"- confirm delay: `{params['confirmDays']:.6g}` days",
        "",
        "## Exact Discrete Model",
        "",
        "The implementation updates an EMA baseline:",
        "",
        "```text",
        "m[k] = alpha * p[k] + (1 - alpha) * m[k-1]",
        "e[k] = (p[k] - m[k]) / m[k]",
        "I[k] = decay * I[k-1] + e[k]",
        "D[k] = e[k] - e[k-1]",
        "u[k] = Kp * e[k] + Ki * I[k] + Kd * D[k]",
        "```",
        "",
        "Linearized around the current price level:",
        "",
        "```text",
        "H(z) = ((1 - alpha) * (1 - z^-1))",
        "       / (1 - (1 - alpha) * z^-1)",
        "",
        "C(z) = Kp + Ki / (1 - decay * z^-1) + Kd * (1 - z^-1)",
        "",
        "G(z) = C(z) * H(z)",
        "```",
        "",
        "## Continuous Approximation",
        "",
        "The EMA pole maps to:",
        "",
        "```text",
        "tau_ma = -T / ln(1 - alpha)",
        "H(s) = tau_ma * s / (tau_ma * s + 1)",
        "C(s) = Kp + Ki / (1 - decay + decay * T * s) + Kd * T * s",
        "```",
        "",
        "For closed-loop inspection this script uses a local exposure plant:",
        "",
        "```text",
        "P(s) = exp(-theta * s) / (tau_p * s + 1)",
        "L(s) = C(s) * H(s) * P(s)",
        "T_cl(s) = L(s) / (1 + L(s))",
        "S(s) = 1 / (1 + L(s))",
        "```",
        "",
        "The plant is not a claim that PID controls price. It is a compact "
        "model for delayed exposure response after the detector changes state.",
        "",
        "## Margin Estimate",
        "",
        f"- nearest gain crossover period: "
        f"`{margins['gainCrossPeriodDays']:.6g}` days",
        f"- nearest crossover magnitude: "
        f"`{margins['gainCrossMagDb']:.6g}` dB",
        f"- approximate phase margin: "
        f"`{margins['phaseMarginDeg']:.6g}` deg",
        "",
        "## Tuning Meaning",
        "",
        "- `PEAK_LOCK_MA_DAYS` sets the high-pass corner. Larger values "
        "ignore faster swings and react to slower trend breaks.",
        "- `Kp` lowers the effective price/EMA threshold. With Ki=Kd=0, "
        "entry is approximately `entryThreshold / Kp`.",
        "- `Ki` adds memory. With decay below 1 it is leaky, not a pure "
        "integrator.",
        "- `Kd` amplifies fast changes in error and should be swept small.",
        "- `CONFIRM_BARS` adds dwell delay after the detector exits long.",
        "",
    ]
    return "\n".join(lines)


###############################################################################
# CLI
###############################################################################

def parseArgs(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="tune.pid_loop_design",
        description="Generate PID loop-design math artifacts.",
    )
    parser.add_argument("--profile", required=True)
    parser.add_argument("--out", required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parseArgs(argv)
    profilePath = rootPath(args.profile)
    outputDir = rootPath(args.out)
    config = loadProfile(profilePath)
    rows, params = frequencyRows(config)
    margins = marginEstimate(rows)
    outputDir.mkdir(parents=True, exist_ok=True)
    writeCsv(outputDir / "pid_frequency.csv", rows)
    atomicJson(outputDir / "pid_summary.json", {
        "profile": str(profilePath),
        "params": params,
        "margins": margins,
    })
    atomicText(outputDir / "pid_loop_design.md", reportText(params, margins))
    print(f"[pid] wrote {outputDir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
