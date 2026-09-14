"""New lottery digit analysis: pooled, prize-separated, and overlap statistics.

This script is intentionally self-contained.  It reads the prize-level raw CSV,
computes the planned tests, writes one result file per analysis group, and
creates a small set of balanced summary figures.
"""

from __future__ import annotations

import argparse
import json
import math
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, chisquare


PRIZE_ORDER = [
    "Đặc biệt", "Giải nhất", "Giải nhì", "Giải ba",
    "Giải tư", "Giải năm", "Giải sáu", "Giải bảy",
]
PRIZE_RULES = {
    "Đặc biệt": (5, "exact"),
    "Giải nhất": (5, "exact"),
    "Giải nhì": (5, "exact"),
    "Giải ba": (5, "exact"),
    "Giải tư": (4, "suffix"),
    "Giải năm": (4, "suffix"),
    "Giải sáu": (3, "suffix"),
    "Giải bảy": (2, "suffix"),
}


def bh_adjust(pvalues: pd.Series) -> pd.Series:
    """Benjamini-Hochberg FDR correction, preserving the original order."""
    p = pvalues.astype(float).to_numpy()
    finite = np.isfinite(p)
    result = np.full(len(p), np.nan, dtype=float)
    if not finite.any():
        return pd.Series(result, index=pvalues.index)
    valid = p[finite]
    n = len(valid)
    order = np.argsort(valid)
    ranked = valid[order]
    adjusted = np.minimum.accumulate((ranked * n / np.arange(1, n + 1))[::-1])[::-1]
    result[finite] = np.clip(adjusted[np.argsort(order)], 0.0, 1.0)
    return pd.Series(result, index=pvalues.index)


def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig", dtype={"number": str})
    required = {"date", "province", "prize", "prize_index", "number"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Thiếu cột bắt buộc: {sorted(missing)}")
    df["date"] = pd.to_datetime(df["date"], errors="raise")
    raw_number = df["number"].astype(str).str.replace(r"\.0$", "", regex=True)
    if (~raw_number.str.fullmatch(r"\d{2,5}")).any():
        raise ValueError("Cột number phải chứa từ 2 đến 5 chữ số.")
    df["number_raw"] = raw_number
    # Position 1 is the units digit, followed by tens, hundreds, thousands,
    # and ten-thousands.  This right-aligned convention is essential because
    # the traditional draw contains 2-, 3-, 4-, and 5-digit prize numbers.
    digits = df["number_raw"].apply(
        lambda x: pd.Series(
            [int(x[::-1][i]) if i < len(x) else np.nan for i in range(5)],
            index=[f"d{i}" for i in range(1, 6)],
        )
    )
    df = pd.concat([df, digits], axis=1)
    df["year"] = df["date"].dt.year
    return df.sort_values(["date", "prize", "prize_index"]).reset_index(drop=True)


def scope_frames(df: pd.DataFrame):
    yield "pooled", df
    for prize in PRIZE_ORDER:
        part = df[df["prize"] == prize]
        if not part.empty:
            yield prize, part


def trim_contingency(table: pd.DataFrame) -> pd.DataFrame:
    """Remove empty rows/columns before scipy's expected-count calculation."""
    return table.loc[(table.sum(axis=1) > 0), (table.sum(axis=0) > 0)]


def digit_distribution(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for scope, part in scope_frames(df):
        for pos in range(1, 6):
            counts = part[f"d{pos}"].dropna().astype(int).value_counts().reindex(range(10), fill_value=0)
            total = counts.sum()
            if total:
                stat, pvalue = chisquare(counts.to_numpy())
            else:
                stat, pvalue = np.nan, np.nan
            expected_count = total / 10 if total else np.nan
            for digit, count in counts.items():
                rows.append({
                    "scope": scope, "position": pos, "digit": digit,
                    "count": int(count), "expected_count": expected_count,
                    "proportion": count / total if total else np.nan,
                    "expected_proportion": 0.10,
                    "deviation": count / total - 0.10 if total else np.nan,
                    "standardized_residual": (count - expected_count) / math.sqrt(expected_count) if total else np.nan,
                    "chi2": stat, "p_value": pvalue,
                })
    out = pd.DataFrame(rows)
    tests = out.drop_duplicates(["scope", "position"]).copy()
    cohen = (
        out.groupby(["scope", "position"], as_index=False)["deviation"]
        .apply(lambda x: math.sqrt(np.sum(x.dropna() ** 2 / 0.10)))
        .rename(columns={"deviation": "cohen_w"})
    )
    tests = tests.merge(cohen, on=["scope", "position"], how="left")
    tests["q_value"] = bh_adjust(tests["p_value"])
    tests["reject_fdr_05"] = tests["q_value"] < 0.05
    out = out.drop(columns=["chi2", "p_value"]).merge(
        tests[["scope", "position", "chi2", "p_value", "q_value", "reject_fdr_05"]],
        on=["scope", "position"], how="left",
    )
    return out


def suffix_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """Test suffixes only on observations whose original number has enough digits."""
    rows = []
    for scope, part in scope_frames(df):
        for length in range(1, 6):
            eligible = part[part["number_raw"].str.len() >= length]
            suffix = eligible["number_raw"].str[-length:]
            support = 10 ** length
            counts = suffix.value_counts().reindex(
                [str(i).zfill(length) for i in range(support)], fill_value=0
            )
            expected_count = len(eligible) / support
            # Pearson's chi-square approximation is unreliable when expected
            # cell counts are small.  Keep the empirical distribution, but do
            # not manufacture a p-value for sparse 4/5-digit supports.
            if expected_count >= 5:
                stat, pvalue = chisquare(counts.to_numpy())
                valid = True
            else:
                stat, pvalue, valid = np.nan, np.nan, False
            rows.append({
                "scope": scope, "suffix_length": length,
                "n_observations": len(eligible), "support_size": support,
                "chi2": stat, "p_value": pvalue,
                "expected_count": expected_count, "valid_chi_square": valid,
            })
    out = pd.DataFrame(rows)
    out["q_value"] = bh_adjust(out["p_value"])
    out["reject_fdr_05"] = out["q_value"] < 0.05
    return out


def position_independence(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for scope, part in scope_frames(df):
        for left, right in combinations(range(1, 6), 2):
            table = pd.crosstab(part[f"d{left}"], part[f"d{right}"]).reindex(
                index=range(10), columns=range(10), fill_value=0
            )
            table = trim_contingency(table)
            if table.empty or min(table.shape) < 2:
                rows.append({
                    "scope": scope, "position_left": left, "position_right": right,
                    "chi2": np.nan, "dof": np.nan, "p_value": np.nan,
                    "cramers_v": np.nan, "n_observations": int(table.to_numpy().sum()),
                })
                continue
            chi2, pvalue, dof, _ = chi2_contingency(table, correction=False)
            n = table.to_numpy().sum()
            denominator = n * min(table.shape[0] - 1, table.shape[1] - 1)
            v = math.sqrt(chi2 / denominator) if denominator else np.nan
            rows.append({
                "scope": scope, "position_left": left, "position_right": right,
                "chi2": chi2, "dof": dof, "p_value": pvalue,
                "cramers_v": v, "n_observations": int(n),
            })
    out = pd.DataFrame(rows)
    out["q_value"] = bh_adjust(out["p_value"])
    out["reject_fdr_05"] = out["q_value"] < 0.05
    return out


def temporal_dependence(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for scope, part in scope_frames(df):
        group_cols = ["prize", "prize_index"] if scope == "pooled" else ["prize_index"]
        for pos in range(1, 6):
            transitions = []
            for _, stream in part.groupby(group_cols, dropna=False):
                stream = stream.sort_values("date")
                current = stream[f"d{pos}"].astype(float)
                transitions.append(pd.DataFrame({"current": current, "previous": current.shift(1)}))
            values = pd.concat(transitions, ignore_index=True).dropna()
            table = pd.crosstab(values["previous"], values["current"]).reindex(
                index=range(10), columns=range(10), fill_value=0
            )
            table = trim_contingency(table)
            if table.empty or min(table.shape) < 2:
                rows.append({
                    "scope": scope, "position": pos, "lag": 1,
                    "chi2": np.nan, "dof": np.nan, "p_value": np.nan,
                    "cramers_v": np.nan, "digit_autocorrelation": np.nan,
                    "n_transitions": int(table.to_numpy().sum()),
                })
                continue
            chi2, pvalue, dof, _ = chi2_contingency(table, correction=False)
            n = table.to_numpy().sum()
            denominator = n * min(table.shape[0] - 1, table.shape[1] - 1)
            v = math.sqrt(chi2 / denominator) if denominator else np.nan
            if values["current"].nunique() < 2 or values["previous"].nunique() < 2:
                autocorr = np.nan
            else:
                autocorr = values["current"].corr(values["previous"])
            rows.append({
                "scope": scope, "position": pos, "lag": 1,
                "chi2": chi2, "dof": dof, "p_value": pvalue,
                "cramers_v": v, "digit_autocorrelation": autocorr,
                "n_transitions": int(n),
            })
    out = pd.DataFrame(rows)
    out["q_value"] = bh_adjust(out["p_value"])
    out["reject_fdr_05"] = out["q_value"] < 0.05
    return out


def build_daily_multi_win(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate overlapping prize rules for each drawn 5-digit number.

    The automatic DB -> DB khuyến khích pair is deliberately excluded because
    the project does not model the ticket's special symbol. Each result keeps
    its native length; no short prize is zero-padded to five digits here.
    """
    daily_rows = []
    for date, day in df.groupby("date", sort=True):
        targets = {}
        for prize, (length, kind) in PRIZE_RULES.items():
            values = day.loc[day["prize"] == prize, "number_raw"]
            targets[prize] = {x if kind == "exact" else x[-length:] for x in values}
        eligible = [p for p in PRIZE_ORDER if p in targets]
        ticket_rows = []
        for number in day["number_raw"]:
            hits = []
            for prize in eligible:
                length, kind = PRIZE_RULES[prize]
                key = number if kind == "exact" else number[-length:]
                if key in targets[prize]:
                    hits.append(prize)
            ticket_rows.append((number, hits))
        hit_counts = [len(hits) for _, hits in ticket_rows]
        max_hits = max(hit_counts, default=0)
        max_payout = max(
            sum({"Đặc biệt": 25_000_000, "Giải nhất": 10_000_000, "Giải nhì": 5_000_000,
                 "Giải ba": 1_000_000, "Giải tư": 400_000, "Giải năm": 200_000,
                 "Giải sáu": 100_000, "Giải bảy": 40_000}[p] for p in hits)
            for _, hits in ticket_rows
        ) if ticket_rows else 0
        daily_rows.append({
            "date": date.date(), "year": date.year,
            "n_drawn_numbers": len(day), "max_simultaneous_prizes": max_hits,
            "max_total_payout": max_payout,
            "has_2plus_prizes": max_hits >= 2,
            "has_3plus_prizes": max_hits >= 3,
            "has_4plus_prizes": max_hits >= 4,
        })
    daily = pd.DataFrame(daily_rows)
    yearly = daily.groupby("year").agg(
        total_days=("date", "size"),
        days_with_2plus_prizes=("has_2plus_prizes", "sum"),
        days_with_3plus_prizes=("has_3plus_prizes", "sum"),
        days_with_4plus_prizes=("has_4plus_prizes", "sum"),
        max_simultaneous_prizes_observed=("max_simultaneous_prizes", "max"),
    ).reset_index()
    for n in (2, 3, 4):
        yearly[f"rate_days_with_{n}plus_prizes"] = yearly[f"days_with_{n}plus_prizes"] / yearly["total_days"]
    return daily, yearly


def iter_digit_streams(part: pd.DataFrame, pooled: bool):
    """Yield independent prize/index streams in chronological order."""
    keys = ["prize", "prize_index"] if pooled else ["prize_index"]
    for _, stream in part.groupby(keys, dropna=False):
        yield stream.sort_values("date")


def markov_transition_tests(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Test lag-1 Markov dependence and save detailed transition cells."""
    summary, detail = [], []
    for scope, part in scope_frames(df):
        pooled = scope == "pooled"
        for pos in range(1, 6):
            previous_all, current_all, gaps_all = [], [], []
            for stream in iter_digit_streams(part, pooled):
                current = stream[f"d{pos}"].astype(float)
                previous = current.shift(1)
                gaps = stream["date"].diff().dt.days
                valid = current.notna() & previous.notna()
                previous_all.extend(previous[valid])
                current_all.extend(current[valid])
                gaps_all.extend(gaps[valid])
            for transition_scope, mask in (
                ("all_observed_draws", np.ones(len(current_all), dtype=bool)),
                ("consecutive_days", np.asarray(gaps_all) == 1),
            ):
                previous = pd.Series(previous_all)[mask]
                current = pd.Series(current_all)[mask]
                table = pd.crosstab(previous, current).reindex(index=range(10), columns=range(10), fill_value=0)
                table = trim_contingency(table)
                if table.empty or min(table.shape) < 2:
                    chi2 = pvalue = v = np.nan
                    dof = 0
                else:
                    chi2, pvalue, dof, _ = chi2_contingency(table, correction=False)
                    n = int(table.to_numpy().sum())
                    denom = n * min(table.shape[0] - 1, table.shape[1] - 1)
                    v = math.sqrt(chi2 / denom) if denom else np.nan
                summary.append({
                    "scope": scope, "transition_scope": transition_scope,
                    "position": pos, "n_transitions": int(len(current)),
                    "chi2": chi2, "dof": dof, "p_value": pvalue, "cramers_v": v,
                })
                if len(current):
                    raw = pd.DataFrame({"previous_digit": previous.astype(int), "next_digit": current.astype(int)})
                    counts = pd.crosstab(raw.previous_digit, raw.next_digit).reindex(index=range(10), columns=range(10), fill_value=0)
                    probabilities = counts.div(counts.sum(axis=1).replace(0, np.nan), axis=0)
                    for old in range(10):
                        for new in range(10):
                            detail.append({
                                "scope": scope, "transition_scope": transition_scope,
                                "position": pos, "previous_digit": old, "next_digit": new,
                                "count": int(counts.loc[old, new]),
                                "probability": probabilities.loc[old, new],
                                "deviation_from_uniform": probabilities.loc[old, new] - 0.10,
                            })
    summary = pd.DataFrame(summary)
    summary["q_value"] = summary.groupby("transition_scope")["p_value"].transform(lambda x: bh_adjust(x))
    summary["reject_fdr_05"] = summary["q_value"] < 0.05
    return summary, pd.DataFrame(detail)


def lag_dependence_tests(df: pd.DataFrame, max_lag: int = 30) -> pd.DataFrame:
    """Test observation lags 1..30, plus a calendar-consecutive variant."""
    rows = []
    for scope, part in scope_frames(df):
        pooled = scope == "pooled"
        for pos in range(1, 6):
            for lag in range(1, max_lag + 1):
                previous_all, current_all, gap_all = [], [], []
                for stream in iter_digit_streams(part, pooled):
                    current = stream[f"d{pos}"].astype(float)
                    previous = current.shift(lag)
                    gap = stream["date"].diff(lag)
                    valid = current.notna() & previous.notna()
                    previous_all.extend(previous[valid])
                    current_all.extend(current[valid])
                    gap_all.extend(gap[valid].dt.days)
                for lag_scope, mask in (
                    ("observed_lag", np.ones(len(current_all), dtype=bool)),
                    ("calendar_lag", np.asarray(gap_all) == lag),
                ):
                    previous = pd.Series(previous_all)[mask]
                    current = pd.Series(current_all)[mask]
                    table = pd.crosstab(previous, current).reindex(index=range(10), columns=range(10), fill_value=0)
                    table = trim_contingency(table)
                    if table.empty or min(table.shape) < 2:
                        chi2 = pvalue = v = np.nan
                        dof = 0
                    else:
                        chi2, pvalue, dof, _ = chi2_contingency(table, correction=False)
                        n = int(table.to_numpy().sum())
                        denom = n * min(table.shape[0] - 1, table.shape[1] - 1)
                        v = math.sqrt(chi2 / denom) if denom else np.nan
                    rows.append({
                        "scope": scope, "lag_scope": lag_scope, "position": pos,
                        "lag": lag, "n_transitions": int(len(current)),
                        "chi2": chi2, "dof": dof, "p_value": pvalue, "cramers_v": v,
                    })
    out = pd.DataFrame(rows)
    out["q_value"] = out.groupby("lag_scope")["p_value"].transform(lambda x: bh_adjust(x))
    out["reject_fdr_05"] = out["q_value"] < 0.05
    return out


def temporal_stability(df: pd.DataFrame, max_lag: int = 30) -> pd.DataFrame:
    """Re-test all lag/position combinations in three independent periods."""
    periods = [(2007, 2012), (2013, 2018), (2019, 2026)]
    rows = []
    for start, end in periods:
        part = df[df.year.between(start, end)]
        result = lag_dependence_tests(part, max_lag=max_lag)
        result = result[result["lag_scope"] == "observed_lag"].copy()
        result["period"] = f"{start}-{end}"
        rows.append(result)
    out = pd.concat(rows, ignore_index=True)
    out["q_value_within_period"] = out.groupby("period")["p_value"].transform(lambda x: bh_adjust(x))
    out["reject_fdr_05_within_period"] = out["q_value_within_period"] < 0.05
    return out


def make_figures(digit, position, temporal, yearly, out_dir: Path):
    for section in ("01_digit_distribution", "02_position_independence",
                    "03_markov_by_position", "04_temporal_randomness",
                    "04b_temporal_stability"):
        (out_dir / section / "figures").mkdir(parents=True, exist_ok=True)

    pooled = digit[digit.scope == "pooled"]
    fig, axes = plt.subplots(2, 3, figsize=(14, 7), sharex=True, sharey=True)
    for ax, pos in zip(axes.flat, range(1, 6)):
        part = pooled[pooled.position == pos]
        ax.bar(part.digit, part.proportion, color="#2f6690", width=0.72)
        ax.axhline(0.1, color="#d1495b", linewidth=1.2, linestyle="--")
        ax.set_title(f"Vị trí {pos}")
        ax.set_ylim(0.0, 0.2)
        ax.set_xticks(range(10))
        ax.grid(axis="y", alpha=0.25)
    axes.flat[-1].axis("off")
    fig.suptitle("Phân phối chữ số theo vị trí — gộp toàn bộ giải", fontsize=14)
    fig.supxlabel("Chữ số", y=0.04)
    fig.supylabel("Tỷ lệ quan sát", x=0.03)
    fig.tight_layout(rect=(0.04, 0.05, 1, 0.94))
    fig.savefig(out_dir / "01_digit_distribution" / "figures" / "01_pooled_digit_distribution.png", dpi=160)
    plt.close(fig)


def make_extended_figures(markov, transitions, lags, stability, digit, position, temporal, yearly, out_dir: Path):
    """Create the Markov/lag/stability views from the old analysis."""
    markov_dir = out_dir / "03_markov_by_position" / "figures"
    lag_dir = out_dir / "04_temporal_randomness" / "figures"
    stability_dir = out_dir / "04b_temporal_stability" / "figures"

    pooled = transitions[
        (transitions.scope == "pooled") &
        (transitions.transition_scope == "all_observed_draws")
    ]
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), sharex=True, sharey=True)
    axes = axes.flat
    max_dev = max(0.01, pooled["deviation_from_uniform"].abs().max())
    for pos in range(1, 6):
        ax = axes[pos - 1]
        part = pooled[pooled.position == pos]
        matrix = part.pivot(index="previous_digit", columns="next_digit", values="deviation_from_uniform").reindex(index=range(10), columns=range(10)) * 100
        image = ax.imshow(matrix, cmap="coolwarm", vmin=-max_dev * 100, vmax=max_dev * 100)
        ax.set_title(f"Vị trí {pos}")
        ax.set_xticks(range(10)); ax.set_yticks(range(10))
        ax.set_xlabel("Sau"); ax.set_ylabel("Trước")
    axes[-1].axis("off")
    fig.suptitle("Ma trận chuyển tiếp Markov — độ lệch so với 10%", fontsize=14)
    cbar_ax = fig.add_axes([0.91, 0.18, 0.018, 0.62])
    fig.colorbar(image, cax=cbar_ax, label="Độ lệch (điểm phần trăm)")
    fig.subplots_adjust(top=0.86, right=0.88, wspace=0.24, hspace=0.38)
    fig.savefig(markov_dir / "01_transition_deviations.png", dpi=160)
    plt.close(fig)

    pooled_lags = lags[(lags.scope == "pooled") & (lags.lag_scope == "observed_lag")]
    matrix = pooled_lags.pivot(index="position", columns="lag", values="cramers_v").reindex(index=range(1, 6))
    fig, ax = plt.subplots(figsize=(14, 5))
    image = ax.imshow(matrix, aspect="auto", cmap="viridis", vmin=0)
    ax.set_title("Mức phụ thuộc theo độ trễ 1–30 kỳ — Cramér's V")
    ax.set_xlabel("Độ trễ (kỳ)"); ax.set_ylabel("Vị trí")
    ax.set_xticks(range(0, 30, 2), range(1, 31, 2)); ax.set_yticks(range(5), range(1, 6))
    fig.colorbar(image, ax=ax, label="Cramér's V")
    fig.tight_layout()
    fig.savefig(lag_dir / "04_lag_dependence_cramers_v.png", dpi=160)
    plt.close(fig)

    pooled_stability = stability[stability.scope == "pooled"]
    pivot = pooled_stability.pivot_table(index="position", columns="period", values="cramers_v", aggfunc="max")
    fig, ax = plt.subplots(figsize=(10, 7))
    image = ax.imshow(pivot.values, aspect="auto", cmap="viridis", vmin=0)
    ax.set_title("Độ ổn định của phụ thuộc theo thời gian")
    ax.set_xlabel("Giai đoạn"); ax.set_ylabel("Vị trí")
    ax.set_xticks(range(len(pivot.columns)), pivot.columns)
    ax.set_yticks(range(len(pivot.index)), [f"Vị trí {p}" for p in pivot.index])
    fig.colorbar(image, ax=ax, label="Cramér's V")
    fig.tight_layout()
    fig.savefig(stability_dir / "05_temporal_stability.png", dpi=160)
    plt.close(fig)

    pooled_pos = position[position.scope == "pooled"].pivot(
        index="position_left", columns="position_right", values="cramers_v"
    ).reindex(index=range(1, 6), columns=range(1, 6))
    fig, ax = plt.subplots(figsize=(7, 6))
    image = ax.imshow(pooled_pos, cmap="viridis", vmin=0, vmax=max(0.05, pooled_pos.max().max()))
    for i in range(5):
        for j in range(5):
            value = pooled_pos.iloc[i, j]
            if not np.isnan(value):
                ax.text(j, i, f"{value:.3f}", ha="center", va="center", color="white" if value > pooled_pos.max().max() / 2 else "black")
    ax.set_xticks(range(5), [f"Vị trí {x}" for x in range(1, 6)])
    ax.set_yticks(range(5), [f"Vị trí {x}" for x in range(1, 6)])
    ax.set_title("Mức liên hệ giữa các vị trí — Cramér's V")
    fig.colorbar(image, ax=ax, label="Cramér's V")
    fig.tight_layout()
    fig.savefig(out_dir / "02_position_independence" / "figures" / "02_pooled_position_dependence.png", dpi=160)
    plt.close(fig)

    fig, (ax, ax_rare) = plt.subplots(1, 2, figsize=(14, 5.5), width_ratios=[2.2, 1], sharex=True)
    x = np.arange(len(yearly))
    ax.bar(x, yearly["rate_days_with_2plus_prizes"] * 100, color="#2f6690", width=0.68,
           label="Từ 2 giải trở lên")
    ax.set_xticks(x, yearly.year.astype(str), rotation=45, ha="right")
    ax.set_ylabel("Tỷ lệ ngày (%)")
    ax.set_title("Từ 2 giải trở lên")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    width = 0.34
    ax_rare.bar(x - width / 2, yearly["rate_days_with_3plus_prizes"] * 100, width,
                color="#ed9b40", label="Từ 3 giải")
    ax_rare.bar(x + width / 2, yearly["rate_days_with_4plus_prizes"] * 100, width,
                color="#6a994e", label="Từ 4 giải")
    ax_rare.set_title("Các trường hợp hiếm")
    ax_rare.set_xticks(x[::2], yearly.year.astype(str).iloc[::2], rotation=45, ha="right")
    ax_rare.set_ylabel("Tỷ lệ ngày (%)")
    ax_rare.grid(axis="y", alpha=0.25)
    ax_rare.legend(frameon=False)
    fig.suptitle("Tỷ lệ ngày có thể trúng đồng thời nhiều giải\n(đã loại cặp tự động ĐB–khuyến khích ĐB)", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_dir / "04_temporal_randomness" / "figures" / "03_multi_win_rates_by_year.png", dpi=160)
    plt.close(fig)

    # One compact diagnostic figure: only pooled tests, with FDR threshold.
    families = [
        ("Digit uniformity", digit[digit.scope == "pooled"].drop_duplicates(["scope", "position"]), "q_value"),
        ("Position independence", position[position.scope == "pooled"], "q_value"),
        ("Temporal dependence", temporal[temporal.scope == "pooled"], "q_value"),
    ]
    labels, values, colors = [], [], []
    for name, part, col in families:
        for idx, row in part.reset_index(drop=True).iterrows():
            labels.append(f"{name}\n{idx + 1}")
            values.append(-math.log10(max(float(row[col]), 1e-300)))
            colors.append("#d1495b" if row[col] < 0.05 else "#5fa8d3")
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.bar(range(len(values)), values, color=colors)
    ax.axhline(-math.log10(0.05), color="#333333", linestyle="--", linewidth=1, label="FDR q = 0.05")
    ax.set_xticks(range(len(values)), labels, rotation=70, ha="right", fontsize=8)
    ax.set_ylabel("−log10(q-value)")
    ax.set_title("Các kiểm định gộp — điểm nổi bật theo FDR")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_dir / "04b_temporal_stability" / "figures" / "04_pooled_test_significance.png", dpi=160)
    plt.close(fig)


def write_report(df, digit, suffix, position, temporal, daily, yearly, markov, lags, stability, out_dir: Path):
    def n_reject(frame):
        return int(frame["reject_fdr_05"].sum())
    digit_tests = digit.drop_duplicates(["scope", "position"])
    suffix_tests = int(suffix["valid_chi_square"].sum())
    report = f"""# Thống kê và kiểm định chữ số mới

## Phạm vi

- Dữ liệu: `{len(df):,}` dòng giải, `{df['date'].nunique():,}` ngày, từ `{df.date.min().date()}` đến `{df.date.max().date()}`.
- Phân tích gồm hai lớp: **gộp toàn bộ giải** và **tách theo từng giải**.
- Mức ý nghĩa: kiểm định hai phía với hiệu chỉnh nhiều kiểm định Benjamini–Hochberg (FDR), ngưỡng q < 0.05.
- Với mỗi độ dài suffix, chỉ các kết quả có đủ số chữ số gốc mới được đưa vào; không zero-pad giải 2/3/4 chữ số thành 5 chữ số.
- Với đuôi 4/5 chữ số, các ô có kỳ vọng thấp được giữ ở dạng tần suất mô tả và không gán p-value Chi-square nếu không đủ điều kiện xấp xỉ.
- Thống kê trúng nhiều giải loại trừ rule tự động: vé trùng Giải đặc biệt không được tính đồng thời Giải phụ đặc biệt và Giải khuyến khích đặc biệt.

## Kết quả kiểm định

| Nhóm | Số kiểm định | Số bác bỏ H0 sau FDR |
|---|---:|---:|
| Phân phối chữ số | {len(digit_tests)} | {n_reject(digit_tests)} |
| Phân phối đuôi số | {int(suffix["valid_chi_square"].sum())} hợp lệ / {len(suffix)} dòng | {n_reject(suffix)} |
| Độc lập giữa vị trí | {len(position)} | {n_reject(position)} |
| Markov lag 1 | {len(markov)} | {n_reject(markov)} |
| Phụ thuộc độ trễ 1–30 | {len(lags)} | {n_reject(lags)} |
| Ổn định theo giai đoạn | {len(stability)} | {int(stability['reject_fdr_05_within_period'].sum())} |

## Trúng đồng thời nhiều giải

- Tỷ lệ ngày có ít nhất 2 giải có thể cùng trúng: **{daily.has_2plus_prizes.mean():.2%}**.
- Tỷ lệ ngày có ít nhất 3 giải có thể cùng trúng: **{daily.has_3plus_prizes.mean():.2%}**.
- Tỷ lệ ngày có ít nhất 4 giải có thể cùng trúng: **{daily.has_4plus_prizes.mean():.2%}**.
- Kiểm tra ổn định theo 3 giai đoạn: lưu toàn bộ kết quả tại `04b_temporal_stability/temporal_stability_by_period.csv`.

Các bảng chi tiết và biểu đồ nằm trong cùng thư mục kết quả.
"""
    (out_dir / "statistical_report.md").write_text(report, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("data/raw/kqxsmb_all_prizes_2007_2026.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/statistics_new"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    sections = {
        "01": args.output_dir / "01_digit_distribution",
        "02": args.output_dir / "02_position_independence",
        "03": args.output_dir / "03_markov_by_position",
        "04": args.output_dir / "04_temporal_randomness",
        "04b": args.output_dir / "04b_temporal_stability",
    }
    for folder in sections.values():
        folder.mkdir(parents=True, exist_ok=True)
    df = load_data(args.input)
    digit = digit_distribution(df)
    suffix = suffix_distribution(df)
    position = position_independence(df)
    temporal = temporal_dependence(df)
    daily, yearly = build_daily_multi_win(df)
    markov, transitions = markov_transition_tests(df)
    lags = lag_dependence_tests(df)
    stability = temporal_stability(df)

    digit.to_csv(sections["01"] / "digit_distribution.csv", index=False, encoding="utf-8-sig")
    suffix.to_csv(sections["01"] / "suffix_distribution.csv", index=False, encoding="utf-8-sig")
    position.to_csv(sections["02"] / "position_independence.csv", index=False, encoding="utf-8-sig")
    temporal.to_csv(sections["04"] / "temporal_dependence_lag1.csv", index=False, encoding="utf-8-sig")
    markov.to_csv(sections["03"] / "markov_tests_by_position.csv", index=False, encoding="utf-8-sig")
    transitions.to_csv(sections["03"] / "markov_transition_probabilities.csv", index=False, encoding="utf-8-sig")
    lags.to_csv(sections["04"] / "temporal_dependence_lags_01_30.csv", index=False, encoding="utf-8-sig")
    daily.to_csv(sections["04"] / "multi_win_daily.csv", index=False, encoding="utf-8-sig")
    yearly.to_csv(sections["04"] / "multi_win_yearly.csv", index=False, encoding="utf-8-sig")
    stability.to_csv(sections["04b"] / "temporal_stability_by_period.csv", index=False, encoding="utf-8-sig")

    metadata = {
        "input": str(args.input), "n_rows": int(len(df)), "n_dates": int(df.date.nunique()),
        "date_min": str(df.date.min().date()), "date_max": str(df.date.max().date()),
        "prizes": PRIZE_ORDER, "fdr_alpha": 0.05,
        "multi_win_excluded_rule": "DB exact match -> DB special-prize/khuyen-khich automatic pair",
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    make_figures(digit, position, temporal, yearly, args.output_dir)
    make_extended_figures(markov, transitions, lags, stability, digit, position, temporal, yearly, args.output_dir)
    write_report(df, digit, suffix, position, temporal, daily, yearly, markov, lags, stability, args.output_dir)

    print(f"Rows: {len(df):,}; dates: {df.date.nunique():,}; range: {df.date.min().date()} -> {df.date.max().date()}")
    print(f"2+ prizes: {daily.has_2plus_prizes.mean():.2%}; 3+: {daily.has_3plus_prizes.mean():.2%}; 4+: {daily.has_4plus_prizes.mean():.2%}")
    print(f"Results written to: {args.output_dir}")


if __name__ == "__main__":
    main()
