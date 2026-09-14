"""Five-year rolling-history experiment for exact special-prize hits.

For every prediction date, only the previous five calendar years are available
to the model.  The experiment measures first-hit waiting time, inter-hit gaps,
cumulative hits and exact-prize profit for Top-1 through Top-10 tickets.

This is a diagnostic walk-forward experiment, not a final strategy-selection
protocol.  The Uniform baseline is included to quantify how unusual any hit
count is under a 1/100,000 exact-number probability.
"""

from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[2]
RAW_FILE = PROJECT_DIR / "data" / "raw" / "kqxsmb_all_prizes_2007_2026.csv"
OUTPUT_DIR = PROJECT_DIR / "artifacts" / "p3_strategies" / "special_prize_5y"

POSITIONS = ("ten_thousands", "thousands", "hundreds", "tens", "units")
DIGITS = np.asarray(list(itertools.product(range(10), repeat=5)), dtype=np.int8)
NUMBERS = np.asarray(["".join(map(str, row)) for row in DIGITS])
TOP_M = tuple(range(1, 11))
COST_PER_TICKET = 10_000
SPECIAL_PAYOUT = 25_000_000
WINDOWS = (30, 90, 180, 365, 730, 1825)


def load_special_prize() -> pd.DataFrame:
    data = pd.read_csv(
        RAW_FILE,
        dtype={"number": str, "prize_index": str},
        encoding="utf-8-sig",
    )
    data["date"] = pd.to_datetime(data["date"], errors="raise")
    data["number"] = data["number"].astype("string").str.strip()
    data = data.loc[data["prize"].eq("Đặc biệt")].copy()
    data = data.loc[data["number"].str.fullmatch(r"\d{5}").fillna(False)].copy()
    data = data.sort_values("date").drop_duplicates("date", keep="first").reset_index(drop=True)
    for index, position in enumerate(POSITIONS):
        data[position] = data["number"].str[index].astype(int)
    return data


def digit_probabilities(history: pd.DataFrame, window: int | None = None) -> np.ndarray:
    sample = history.tail(window) if window else history
    probabilities = []
    for position in POSITIONS:
        counts = sample[position].value_counts().reindex(range(10), fill_value=0).to_numpy(dtype=float)
        probabilities.append((counts + 1.0) / (counts.sum() + 10.0))
    return np.asarray(probabilities)


def markov_probabilities(history: pd.DataFrame) -> np.ndarray:
    probabilities = []
    for position in POSITIONS:
        if len(history) < 2:
            probabilities.append(np.full(10, 0.1))
            continue
        previous = history[position].iloc[:-1].to_numpy(dtype=int)
        current = history[position].iloc[1:].to_numpy(dtype=int)
        last_digit = int(history[position].iloc[-1])
        counts = np.ones(10, dtype=float)
        counts += np.bincount(current[previous == last_digit], minlength=10)
        probabilities.append(counts / counts.sum())
    return np.asarray(probabilities)


def top_candidates(probabilities: np.ndarray, actual: str) -> tuple[list[str], int, float]:
    scores = np.prod(probabilities[np.arange(5)[:, None], DIGITS.T], axis=0)
    scores = scores / scores.sum()
    actual_index = int(actual)
    actual_probability = float(scores[actual_index])
    true_rank = int(np.count_nonzero(scores > actual_probability) + 1)
    candidate_index = np.argpartition(-scores, 10)[:10]
    candidate_index = candidate_index[np.argsort(-scores[candidate_index], kind="stable")]
    return NUMBERS[candidate_index].tolist(), true_rank, actual_probability


def collect_daily(data: pd.DataFrame) -> pd.DataFrame:
    first_date = data["date"].min() + pd.DateOffset(years=5)
    target = data.loc[data["date"].ge(first_date)].copy()
    rows = []
    for _, current in target.iterrows():
        history = data.loc[
            data["date"].lt(current["date"])
            & data["date"].ge(current["date"] - pd.DateOffset(years=5))
        ]
        models = {
            "uniform_random": None,
            "rolling_frequency_w30": digit_probabilities(history, 30),
            "rolling_frequency_w90": digit_probabilities(history, 90),
            "rolling_frequency_w180": digit_probabilities(history, 180),
            "rolling_frequency_w365": digit_probabilities(history, 365),
            "rolling_frequency_w730": digit_probabilities(history, 730),
            "rolling_frequency_w1825": digit_probabilities(history, 1825),
            "markov_position_5y": markov_probabilities(history),
        }
        for model, probabilities in models.items():
            if model == "uniform_random":
                seed = int(current["date"].strftime("%Y%m%d"))
                order = np.random.default_rng(seed).permutation(len(NUMBERS))
                candidates = NUMBERS[order[:10]].tolist()
                true_rank = int(np.flatnonzero(order == int(current["number"]))[0] + 1)
                actual_probability = 1.0 / len(NUMBERS)
            else:
                candidates, true_rank, actual_probability = top_candidates(probabilities, current["number"])
            row = {
                "date": current["date"],
                "model": model,
                "actual": current["number"],
                "true_rank": true_rank,
                "actual_probability": actual_probability,
            }
            for m in TOP_M:
                row[f"top{m}"] = candidates[:m]
                row[f"top{m}_hit"] = int(current["number"] in candidates[:m])
            rows.append(row)
    return pd.DataFrame(rows)


def waiting_summary(daily: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model, m), frame in daily.groupby(["model", "m"], sort=True):
        frame = frame.sort_values("date").reset_index(drop=True)
        hits = frame.loc[frame[f"top{m}_hit"].eq(1), "date"]
        hit_dates = pd.to_datetime(hits).tolist()
        first_prediction_date = frame["date"].iloc[0]
        first_hit_date = hit_dates[0] if hit_dates else pd.NaT
        gaps = np.diff(np.asarray(hit_dates, dtype="datetime64[D]")).astype("timedelta64[D]").astype(int) if len(hit_dates) > 1 else np.asarray([], dtype=int)
        n_days = len(frame)
        n_hits = len(hit_dates)
        cost = n_days * m * COST_PER_TICKET
        payout = n_hits * SPECIAL_PAYOUT
        uniform_rate = m / 100_000
        rows.append(
            {
                "model": model,
                "m": m,
                "first_prediction_date": first_prediction_date,
                "last_prediction_date": frame["date"].iloc[-1],
                "n_prediction_days": n_days,
                "n_hits": n_hits,
                "hit_rate": n_hits / n_days,
                "uniform_expected_hits": n_days * uniform_rate,
                "uniform_probability_at_least_one_hit": 1.0 - (1.0 - uniform_rate) ** n_days,
                "hit_lift_vs_uniform": (n_hits / n_days) / uniform_rate if uniform_rate else np.nan,
                "first_hit_date": first_hit_date,
                "first_hit_wait_calendar_days": int((first_hit_date - first_prediction_date).days) if pd.notna(first_hit_date) else np.nan,
                "first_hit_after_n_predictions": int(frame.index[frame[f"top{m}_hit"].eq(1)][0] + 1) if n_hits else np.nan,
                "mean_inter_hit_days": float(gaps.mean()) if len(gaps) else np.nan,
                "median_inter_hit_days": float(np.median(gaps)) if len(gaps) else np.nan,
                "cost": cost,
                "payout": payout,
                "profit": payout - cost,
                "roi": (payout - cost) / cost,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    data = load_special_prize()
    print(f"Special-prize dates: {data.date.min().date()} -> {data.date.max().date()}")
    print(f"Five-year rolling evaluation starts: {(data.date.min() + pd.DateOffset(years=5)).date()}")
    daily = collect_daily(data)
    summary = waiting_summary(daily)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    daily.to_csv(OUTPUT_DIR / "special_prize_5y_daily.csv.gz", index=False, compression="gzip", encoding="utf-8-sig")
    summary.to_csv(OUTPUT_DIR / "special_prize_5y_waiting_summary.csv", index=False, encoding="utf-8-sig")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
