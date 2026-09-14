"""Fixed-cutoff training and five-year special-prize investment backtest.

Statistical models are updated prequentially using observations strictly before
each test date. CatBoost and XGBoost parameters are fitted only on data through
2020-12-31. The common test period is 2021-01-01 through 2025-12-31.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_DIR / "artifacts" / "p3_strategies" / "special_prize_train2020_test2021_2025"
MODEL_SCRIPT = PROJECT_DIR / "experiments" / "p2_models" / "06_special_prize_models.py"
BACKTEST_SCRIPT = PROJECT_DIR / "experiments" / "p3_strategies" / "03_special_prize_backtest.py"

TEST_START = pd.Timestamp("2021-01-01")
TEST_END = pd.Timestamp("2025-12-31")
TRAIN_END = pd.Timestamp("2020-12-31")
PHASE = "train_2007_2020_test_2021_2025"


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def waiting_summary(daily: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (phase, model, m), frame in daily.groupby(["phase", "model", "m"], sort=True):
        frame = frame.sort_values("date").reset_index(drop=True)
        hit_dates = pd.to_datetime(frame.loc[frame["db_hit_day"].eq(1), "date"]).tolist()
        prediction_dates = pd.to_datetime(frame["date"])
        first_hit = hit_dates[0] if len(hit_dates) >= 1 else pd.NaT
        second_hit = hit_dates[1] if len(hit_dates) >= 2 else pd.NaT
        first_wait = int((first_hit - prediction_dates.iloc[0]).days) if pd.notna(first_hit) else np.nan
        second_wait = int((second_hit - first_hit).days) if pd.notna(second_hit) else np.nan
        first_index = int(frame.index[frame["db_hit_day"].eq(1)][0] + 1) if len(hit_dates) >= 1 else np.nan
        second_index = int(frame.index[frame["db_hit_day"].eq(1)][1] + 1) if len(hit_dates) >= 2 else np.nan
        gaps = np.diff(np.asarray(hit_dates, dtype="datetime64[D]")).astype("timedelta64[D]").astype(int) if len(hit_dates) > 1 else np.asarray([], dtype=int)
        rows.append(
            {
                "phase": phase,
                "model": model,
                "m": m,
                "prediction_days": len(frame),
                "db_hit_days": int(frame["db_hit_day"].sum()),
                "first_hit_date": first_hit,
                "first_hit_wait_calendar_days": first_wait,
                "first_hit_after_n_predictions": first_index,
                "second_hit_date": second_hit,
                "second_hit_wait_after_first_calendar_days": second_wait,
                "second_hit_after_n_predictions": second_index,
                "mean_inter_hit_days": float(gaps.mean()) if len(gaps) else np.nan,
                "median_inter_hit_days": float(np.median(gaps)) if len(gaps) else np.nan,
                "all_hit_dates": ";".join(pd.Timestamp(date).strftime("%Y-%m-%d") for date in hit_dates),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    model_module = load_module(MODEL_SCRIPT, "special_prize_models_train2020")
    backtest_module = load_module(BACKTEST_SCRIPT, "special_prize_backtest_train2020")

    data = model_module.load_special_prize()
    print(f"Training history: {data.date.min().date()} -> {TRAIN_END.date()}", flush=True)
    print(f"Test period: {TEST_START.date()} -> {TEST_END.date()}", flush=True)

    predictions = model_module.collect_predictions(
        data=data,
        phase=PHASE,
        start=TEST_START,
        end=TEST_END,
        history_end=TRAIN_END,
    )
    model_summary = model_module.summarize(predictions)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    prediction_path = OUTPUT_DIR / "special_prize_predictions.csv.gz"
    predictions.to_csv(prediction_path, index=False, compression="gzip", encoding="utf-8-sig")

    backtest_module.PREDICTION_FILE = prediction_path
    backtest_module.OUTPUT_DIR = OUTPUT_DIR
    daily, prizes = backtest_module.run()
    daily_summary = backtest_module.summarize_daily(daily)
    prize_summary = backtest_module.summarize_prizes(prizes, daily)
    wait_summary = waiting_summary(daily)

    model_summary.to_csv(OUTPUT_DIR / "special_prize_model_summary.csv", index=False, encoding="utf-8-sig")
    daily.to_csv(OUTPUT_DIR / "special_prize_daily_results.csv.gz", index=False, compression="gzip", encoding="utf-8-sig")
    prizes.to_csv(OUTPUT_DIR / "special_prize_prize_hits.csv.gz", index=False, compression="gzip", encoding="utf-8-sig")
    daily_summary.to_csv(OUTPUT_DIR / "special_prize_strategy_summary.csv", index=False, encoding="utf-8-sig")
    prize_summary.to_csv(OUTPUT_DIR / "special_prize_by_prize_summary.csv", index=False, encoding="utf-8-sig")
    wait_summary.to_csv(OUTPUT_DIR / "special_prize_waiting_summary.csv", index=False, encoding="utf-8-sig")

    print("Model summary:", flush=True)
    print(model_summary.to_string(index=False), flush=True)
    print("Waiting summary:", flush=True)
    print(wait_summary.to_string(index=False), flush=True)
    print("Strategy summary:", flush=True)
    print(daily_summary.to_string(index=False), flush=True)
    print("By-prize summary:", flush=True)
    print(prize_summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
