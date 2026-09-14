"""Dedicated models for the exact five-digit Northern special prize.

The target is exactly one five-digit number per day from the row whose prize is
"Đặc biệt".  This stream is intentionally separate from the 27-result daily
pool used by the other P2 experiments.

Protocol:
- development/train: 2007-01-01 through 2022-12-31
- validation: 2023-01-01 through 2024-12-31
- final test: 2025-01-01 onward

All stateful models use only observations strictly before the prediction date.
Boosted models are fitted once per phase using only the corresponding history
boundary.  The script writes only the top 100 candidates per day, while exact
probability metrics are computed from the full 100,000-number distribution.
"""

from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[2]
RAW_FILE = PROJECT_DIR / "data" / "raw" / "kqxsmb_all_prizes_2007_2026.csv"
OUTPUT_DIR = PROJECT_DIR / "artifacts" / "p2_models" / "special_prize"

POSITION_NAMES = (
    "ten_thousands",
    "thousands",
    "hundreds",
    "tens",
    "units",
)
DIGITS = np.asarray(list(itertools.product(range(10), repeat=5)), dtype=np.int8)
NUMBERS = np.asarray(["".join(map(str, row)) for row in DIGITS])
TOP_K = (1, 5, 10, 20, 50, 100)
PHASES = {
    "validation_2023_2024": (pd.Timestamp("2023-01-01"), pd.Timestamp("2024-12-31"), pd.Timestamp("2022-12-31")),
    "final_test_2025_2026": (pd.Timestamp("2025-01-01"), pd.Timestamp("2026-12-31"), pd.Timestamp("2024-12-31")),
}
ROLLING_WINDOWS = (30, 90, 180, 365, 730)


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
    if data.empty:
        raise ValueError("Không tìm thấy dòng Đặc biệt hợp lệ")
    if data["date"].duplicated().any():
        raise ValueError("Dữ liệu Đặc biệt có nhiều dòng trong cùng một ngày")
    for index, position in enumerate(POSITION_NAMES):
        data[position] = data["number"].str[index].astype(int)
    return data


def smoothed_digit_probabilities(history: pd.DataFrame, window: int | None = None) -> np.ndarray:
    sample = history.tail(window) if window else history
    if sample.empty:
        return np.full((5, 10), 0.1)
    probabilities = []
    for position in POSITION_NAMES:
        counts = sample[position].value_counts().reindex(range(10), fill_value=0).to_numpy(dtype=float)
        probabilities.append((counts + 1.0) / (counts.sum() + 10.0))
    return np.asarray(probabilities)


def markov_digit_probabilities(history: pd.DataFrame) -> np.ndarray:
    if len(history) < 2:
        return np.full((5, 10), 0.1)
    probabilities = []
    for position in POSITION_NAMES:
        previous = history[position].iloc[:-1].to_numpy(dtype=int)
        current = history[position].iloc[1:].to_numpy(dtype=int)
        last_digit = int(history[position].iloc[-1])
        counts = np.ones(10, dtype=float)
        counts += np.bincount(current[previous == last_digit], minlength=10)
        probabilities.append(counts / counts.sum())
    return np.asarray(probabilities)


def feature_frame(data: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for lag in (1, 2, 3, 7, 14, 30):
        part = data[list(POSITION_NAMES)].shift(lag)
        part.columns = [f"{column}_lag{lag}" for column in POSITION_NAMES]
        parts.append(part)
    dates = data["date"]
    parts.append(
        pd.DataFrame(
            {
                "dow_sin": np.sin(2 * np.pi * dates.dt.dayofweek / 7),
                "dow_cos": np.cos(2 * np.pi * dates.dt.dayofweek / 7),
                "month_sin": np.sin(2 * np.pi * dates.dt.month / 12),
                "month_cos": np.cos(2 * np.pi * dates.dt.month / 12),
            },
            index=data.index,
        )
    )
    return pd.concat(parts, axis=1)


def fit_boosted_models(data: pd.DataFrame, history_end: pd.Timestamp, model_name: str) -> list:
    try:
        if model_name == "catboost_position":
            from catboost import CatBoostClassifier

            make_model = lambda: CatBoostClassifier(
                iterations=180,
                depth=5,
                learning_rate=0.03,
                loss_function="MultiClass",
                l2_leaf_reg=5,
                random_seed=42,
                thread_count=-1,
                verbose=False,
                allow_writing_files=False,
            )
        else:
            from xgboost import XGBClassifier

            make_model = lambda: XGBClassifier(
                n_estimators=180,
                max_depth=3,
                learning_rate=0.03,
                subsample=0.85,
                colsample_bytree=0.85,
                min_child_weight=5,
                objective="multi:softprob",
                num_class=10,
                eval_metric="mlogloss",
                tree_method="hist",
                n_jobs=-1,
                random_state=42,
                verbosity=0,
            )
    except ImportError as error:
        print(f"Bo qua {model_name}: {error}")
        return []

    features = feature_frame(data)
    train = data["date"].le(history_end) & features.notna().all(axis=1)
    models = []
    for position in POSITION_NAMES:
        model = make_model()
        model.fit(features.loc[train], data.loc[train, position].astype(int))
        models.append(model)
    return models


def boosted_probabilities(models: list, features: pd.DataFrame, row_index: int) -> np.ndarray:
    probabilities = []
    row = features.iloc[[row_index]]
    for model in models:
        raw = model.predict_proba(row)[0]
        aligned = np.zeros(10, dtype=float)
        for class_index, class_value in enumerate(model.classes_.astype(int)):
            aligned[class_value] = raw[class_index]
        probabilities.append(aligned)
    return np.asarray(probabilities)


def rank_candidates(probabilities: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    scores = np.prod(probabilities[np.arange(5)[:, None], DIGITS.T], axis=0)
    scores = scores / scores.sum()
    order = np.argsort(-scores, kind="stable")
    return order, scores


def model_probabilities(name: str, history: pd.DataFrame) -> np.ndarray:
    if name == "expanding_frequency":
        return smoothed_digit_probabilities(history)
    if name.startswith("rolling_frequency_w"):
        return smoothed_digit_probabilities(history, int(name.rsplit("w", 1)[1]))
    if name == "markov_position":
        return markov_digit_probabilities(history)
    raise ValueError(name)


def collect_predictions(
    data: pd.DataFrame,
    phase: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    history_end: pd.Timestamp,
) -> pd.DataFrame:
    target = data.loc[data["date"].between(start, end)].copy()
    target_indices = target.index.to_list()
    model_names = ["uniform_random", "expanding_frequency"]
    model_names.extend(f"rolling_frequency_w{window}" for window in ROLLING_WINDOWS)
    model_names.append("markov_position")
    boosted_models = {}
    for model_name in ("catboost_position", "xgboost_position"):
        models = fit_boosted_models(data, history_end, model_name)
        if models:
            boosted_models[model_name] = models
            model_names.append(model_name)

    features = feature_frame(data)
    rows = []
    for target_index in target_indices:
        current = data.loc[target_index]
        history = data.loc[data["date"].lt(current["date"])]
        for model_name in model_names:
            if model_name == "uniform_random":
                seed = int(current["date"].strftime("%Y%m%d"))
                order = np.random.default_rng(seed).permutation(len(NUMBERS))
                scores = np.full(len(NUMBERS), 1.0 / len(NUMBERS))
            elif model_name in boosted_models:
                probabilities = boosted_probabilities(boosted_models[model_name], features, target_index)
                order, scores = rank_candidates(probabilities)
            else:
                probabilities = model_probabilities(model_name, history)
                order, scores = rank_candidates(probabilities)
            actual_index = int(current["number"])
            actual_probability = float(scores[actual_index])
            true_rank = int(np.flatnonzero(order == actual_index)[0] + 1)
            brier_score = float(np.square(scores).sum() - 2.0 * actual_probability + 1.0)
            row = {
                "date": current["date"],
                "phase": phase,
                "model": model_name,
                "actual": current["number"],
                "actual_probability": actual_probability,
                "true_rank": true_rank,
                "log_loss": float(-np.log(max(actual_probability, 1e-15))),
                "brier_score": brier_score,
            }
            for top_k in TOP_K:
                row[f"top{top_k}"] = " ".join(NUMBERS[order[:top_k]])
                row[f"top{top_k}_hit"] = int(actual_index in order[:top_k])
            row["top100"] = " ".join(NUMBERS[order[:100]])
            rows.append(row)
    return pd.DataFrame(rows)


def summarize(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (phase, model), frame in predictions.groupby(["phase", "model"], sort=True):
        row = {
            "phase": phase,
            "model": model,
            "n_days": len(frame),
            "exact_hit_days": int(frame["top1_hit"].sum()),
            "exact_hit_rate_top1": float(frame["top1_hit"].mean()),
            "mean_true_rank": float(frame["true_rank"].mean()),
            "median_true_rank": float(frame["true_rank"].median()),
            "mean_log_loss": float(frame["log_loss"].mean()),
            "mean_brier_score": float(frame["brier_score"].mean()),
        }
        for top_k in TOP_K:
            row[f"hit_rate_top{top_k}"] = float(frame[f"top{top_k}_hit"].mean())
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    data = load_special_prize()
    prediction_frames = []
    for phase, (start, end, history_end) in PHASES.items():
        print(f"Running special-prize models: {phase}", flush=True)
        prediction_frames.append(collect_predictions(data, phase, start, end, history_end))
    predictions = pd.concat(prediction_frames, ignore_index=True)
    summary = summarize(predictions)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(
        OUTPUT_DIR / "special_prize_predictions.csv.gz",
        index=False,
        compression="gzip",
        encoding="utf-8-sig",
    )
    summary.to_csv(OUTPUT_DIR / "special_prize_model_summary.csv", index=False, encoding="utf-8-sig")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
