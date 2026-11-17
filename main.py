"""GTA VI Twitch CCU forecast, v2: kernel-weighted Gaussian day-shape model.

For each launch hour h = 2..24 the program forecasts five quantiles of the
Twitch concurrent viewers in the release-day category, from the hour-1 print
and the past launches shipped in the case file.

Model (all in log space, ratios to the hour-1 print):
  - Each past launch contributes its release-day log-ratio curve
    l_h = ln(viewers_h / viewers_1).
  - A similarity kernel weights the launches: tier (comparable 1.0,
    early-access-contaminated 0.5, weak 0.3), platform class match
    (console-only vs multi-platform, mismatch 0.4), log hour-1 distance
    (Gaussian, bandwidth 0.85), and pre-release buzz ratio (Gaussian,
    bandwidth 2.0). No recency factor: the shape-to-scale relationship is
    structural, and leave-one-out CV over the kit's comparables found
    recency only adds noise.
  - Per hour, the forecast is a Gaussian in log-ratio space: the
    kernel-weighted mean and standard deviation of l_h across the launches,
    with quantiles mean + z_tau * sd * spread. The spread is calibrated by
    launch class (console-only 0.65, multi-platform 1.1): console-only
    launches are more homogeneous than the mixed pool suggests, while
    multi-platform launches are more dispersed.
  - If no launch carries weight for an hour, or the case has no usable
    comparables at all, that hour falls back to the house persistence
    baseline (median at the hour-1 print, spread widening like a random
    walk in log space).

The program is deterministic, standard library only, and reads no network.
"""

import json
import math
import os
import sys
from datetime import datetime
from pathlib import Path

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = {0.05: -1.645, 0.25: -0.674, 0.5: 0.0, 0.75: 0.674, 0.95: 1.645}
HOURS = list(range(2, 25))

# Kernel and calibration constants, tuned by leave-one-out CV over the
# kit's 37 usable past launches (all splits, practice-like included).
BW_H1 = 0.85
BW_BUZZ = 2.0
PLAT_MISMATCH = 0.4
SPREAD_CONSOLE = 0.65
SPREAD_MULTI = 1.1

# House persistence baseline (the kit starter): spread per sqrt hour.
BASE_SIGMA = 0.15


def parse_ts(text):
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def round2(x):
    """Round to 2 decimals, half up; a whole number comes back as an int."""
    scaled = x * 100
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 100
    return int(value) if value.is_integer() else value


def gauss(distance, bandwidth):
    if bandwidth is None or bandwidth <= 0:
        return 1.0
    return math.exp(-(distance * distance) / (2.0 * bandwidth * bandwidth))


def tier_weight(unlock_text):
    """Weight of a past launch from its tier note; None means it is not a
    launch at all (the target game's own pre-release history) and must be
    left out of the shape pool entirely."""
    text = unlock_text or ""
    if "target-history" in text:
        return None
    if "Tier: weak" in text:
        return 0.3
    if "early-access" in text.lower():
        return 0.5
    return 1.0


def is_console_only(platforms):
    """A launch is console-only when PC is not among its launch platforms."""
    return "PC" not in (platforms or "")


def target_console_only(notes):
    """Read the target game's platform class from the case notes. The notes
    name the launch platforms right after the release date; a PC mention
    there means a PC launch, while 'console only' or 'no PC' means the
    game is console-only even when a later PC date is mentioned."""
    text = (notes or "").lower()
    if "console only" in text or "no pc" in text:
        return True
    idx = text.find("releas")
    if idx >= 0:
        segment = text[idx:idx + 120]
        for stop in (";", ". ", "\n"):
            cut = segment.find(stop)
            if cut >= 0:
                segment = segment[:cut]
        if "pc" in segment:
            return False
    return True


# ------------------------------------------------------------ extraction ---

def extract_launch(record):
    """One past launch -> feature dict, or None when it cannot be used:
    the target's own history, or a launch without a complete positive
    release-day curve (hours 1..24)."""
    tier = tier_weight(record.get("unlock", ""))
    if tier is None:
        return None
    day_start = parse_ts(record["day_start"])
    curve = {}
    for row in record["hourly"]:
        hour = int((parse_ts(row["from"]) - day_start).total_seconds() // 3600) + 1
        if 1 <= hour <= 24:
            curve[hour] = row["viewers"]
    if len(curve) != 24 or any(v is None or v <= 0 for v in curve.values()):
        return None
    pre = [row["viewers"] for row in record["hourly"]
           if -24 <= (parse_ts(row["from"]) - day_start).total_seconds() / 3600 < 0]
    hour1 = curve[1]
    buzz = (sum(pre) / len(pre) / hour1) if pre else None
    return {
        "day_start": day_start,
        "console_only": is_console_only(record.get("platforms", "")),
        "tier": tier,
        "h1": hour1,
        "log_h1": math.log(hour1),
        "log_ratio": {h: math.log(curve[h] / hour1) for h in range(1, 25)},
        "log_buzz": math.log(buzz) if buzz and buzz > 0 else None,
    }


def target_features(case):
    """Features of the game being forecast, from the case file itself."""
    first = next((k for k in case["known"] if k["hour"] == 1), None)
    if first is None:
        raise ValueError("case has no hour-1 print")
    day_start = parse_ts(case["day_start"])
    pre = [row["viewers"] for row in (case.get("history_hourly") or [])
           if -24 <= (parse_ts(row["from"]) - day_start).total_seconds() / 3600 < 0]
    hour1 = max(1, first["viewers"])
    buzz = (sum(pre) / len(pre) / hour1) if pre else None
    return {
        "h1": hour1,
        "log_h1": math.log(hour1),
        "console_only": target_console_only(case.get("notes", "")),
        "log_buzz": math.log(buzz) if buzz and buzz > 0 else None,
    }


# ---------------------------------------------------------------- kernel ---

def kernel_weight(target, launch):
    """Similarity of one past launch to the target game."""
    weight = launch["tier"]
    if target["console_only"] != launch["console_only"]:
        weight *= PLAT_MISMATCH
    weight *= gauss(target["log_h1"] - launch["log_h1"], BW_H1)
    if target["log_buzz"] is not None and launch["log_buzz"] is not None:
        weight *= gauss(target["log_buzz"] - launch["log_buzz"], BW_BUZZ)
    return weight


def weighted_mean_sd(pairs):
    """Kernel-weighted mean and standard deviation of log-ratios.
    Returns (mean, sd) or (None, None) when nothing carries weight."""
    total = sum(w for _, w in pairs)
    if total <= 0 or not pairs:
        return None, None
    mean = sum(v * w for v, w in pairs) / total
    var = sum(w * (v - mean) ** 2 for v, w in pairs) / total
    return mean, math.sqrt(max(var, 0.0))


def baseline_quantiles(hour):
    """House persistence baseline in log-ratio space for one hour."""
    spread = BASE_SIGMA * math.sqrt(hour - 1)
    return {tau: Z[tau] * spread for tau in LEVELS}


def forecast_log_quantiles(target, pool):
    """Per hour h = 2..24: the five forecast log-ratio quantiles."""
    out = {}
    spread = SPREAD_CONSOLE if target["console_only"] else SPREAD_MULTI
    for hour in HOURS:
        pairs = []
        for launch in pool:
            weight = kernel_weight(target, launch)
            if weight > 0:
                pairs.append((launch["log_ratio"][hour], weight))
        mean, sd = weighted_mean_sd(pairs)
        if mean is None:
            out[hour] = baseline_quantiles(hour)
        else:
            out[hour] = {tau: mean + Z[tau] * sd * spread for tau in LEVELS}
    return out


# ----------------------------------------------------------------- driver ---

def forecast_case(case):
    """Forecast one case: 23 rows, hours 2..24, five quantiles each."""
    if case.get("levels") != LEVELS:
        raise ValueError("case asks for levels %r, not %r"
                         % (case.get("levels"), LEVELS))
    target = target_features(case)
    pool = []
    for record in case.get("comparables") or []:
        launch = extract_launch(record)
        if launch is not None:
            pool.append(launch)
    if not pool:
        log_q = {hour: baseline_quantiles(hour) for hour in HOURS}
    else:
        log_q = forecast_log_quantiles(target, pool)
    hour1 = target["h1"]
    rows = []
    for row in case["forecast"]:
        hour = row["hour"]
        quantiles = []
        for tau in LEVELS:
            value = hour1 * math.exp(log_q[hour][tau])
            value = min(max(value, 0.01), 1e9)
            quantiles.append(round2(value))
        # Rounding can only tie neighbours, never invert them, but guard
        # the non-decreasing order explicitly.
        for i in range(1, len(quantiles)):
            if quantiles[i] < quantiles[i - 1]:
                quantiles[i] = quantiles[i - 1]
        rows.append({"hour": hour, "q": quantiles})
    rows.sort(key=lambda r: r["hour"])
    return {"hours": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case)
        (OUTPUT_DIR / (path.stem + ".out")).write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
