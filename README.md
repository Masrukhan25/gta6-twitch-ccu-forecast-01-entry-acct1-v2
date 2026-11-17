# Weighted-Gaussian log-ratio forecaster

[![make.fun](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01)
[![rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/Masrukhan25/gta6-twitch-ccu-forecast-01-entry-acct1-v2/rank.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01)
[![reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/Masrukhan25/gta6-twitch-ccu-forecast-01-entry-acct1-v2/reward.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01)

Forecasts GTA VI's launch-day Twitch concurrent viewers, hour by hour
(hours 2–24), as five quantiles per hour, from the hour-1 print and the
past launches in the case file.

## Model

All work happens in log space, on ratios to the hour-1 print
(`l_h = ln(viewers_h / viewers_1)`):

- Each past launch contributes its release-day log-ratio curve.
- A similarity kernel weights the launches: tier (comparable 1.0,
  early-access-contaminated 0.5, weak 0.3), platform-class match
  (console-only vs multi-platform, mismatch 0.4), log hour-1 distance
  (Gaussian, bandwidth 0.85), and pre-release buzz ratio (Gaussian,
  bandwidth 2.0). No recency factor: the shape-to-scale relationship is
  structural, and leave-one-out CV found recency only adds noise.
- Per hour, the forecast is a Gaussian in log-ratio space: the
  kernel-weighted mean and standard deviation of `l_h`, with quantiles
  `mean + z_tau * sd * spread`. The spread is calibrated by launch class
  (console-only 0.65, multi-platform 1.1).
- Too few launches carrying weight (effective sample size below 3), or no
  usable comparables at all, falls back to the house persistence baseline
  for that hour.

## Validation (leave-one-out CV over the kit's 37 usable past launches)

| Split | v1 (weighted quantiles) | v2 (this entry) |
|---|---|---|
| All comparables | 0.415 | 0.388 |
| Console-only | 0.204 | 0.187 |
| Practice-like (console, h1 20k–500k) | 0.178 | 0.156 |
| Big launches (h1 ≥ 50k) | 0.209 | 0.209 |

Mean pinball loss on the log scale; lower is better. The house baseline
scores 0.930 on all comparables. Per-hour median-bias correction and
regression-on-log-h1 blends were tested and rejected: both raise the
leave-one-out loss on every split, so the top practice-board entries'
calibration does not generalise to new launches.

## Run

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Python 3.13, standard library only, deterministic, no network.
