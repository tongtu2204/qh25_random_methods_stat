"""Backtest exact special-prize tickets and their incidental lower-prize hits.

The primary result is exact "Đặc biệt" performance.  The same selected
five-digit tickets are also checked against every raw prize row on the same
day, so lower-prize hit rates and payouts are reported separately.

No model or m is selected from the final period.  The script only evaluates
all requested m values; selection must be made on validation first.
"""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path

import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[2]
RAW_FILE = PROJECT_DIR / "data" / "raw" / "kqxsmb_all_prizes_2007_2026.csv"
PREDICTION_FILE = PROJECT_DIR / "artifacts" / "p2_models" / "special_prize" / "special_prize_predictions.csv.gz"
OUTPUT_DIR = PROJECT_DIR / "artifacts" / "p3_strategies" / "special_prize"

COST_PER_TICKET = 10_000
TOP_M = tuple(range(1, 11))
PAYOUT = {
    "Đặc biệt": 25_000_000,
    "Phụ ĐB": 25_000_000,
    "Khuyến khích ĐB": 40_000,
    "Giải nhất": 10_000_000,
    "Giải nhì": 5_000_000,
    "Giải ba": 1_000_000,
    "Giải tư": 400_000,
    "Giải năm": 200_000,
    "Giải sáu": 100_000,
    "Giải bảy": 40_000,
}


def prize_slug(value: str) -> str:
    value = unicodedata.normalize("NFKD", str(value)).encode("ascii", "ignore").decode("ascii")
    value = value.lower().replace("d", "d")
    return re.sub(r"[^a-z0-9]+", "_", value).strip("_")


def load_raw() -> pd.DataFrame:
    data = pd.read_csv(
        RAW_FILE,
        dtype={"number": str, "prize_index": str},
        encoding="utf-8-sig",
    )
    data["date"] = pd.to_datetime(data["date"], errors="raise")
    data["number"] = data["number"].astype("string").str.strip()
    valid = data["number"].str.fullmatch(r"\d+").fillna(False)
    data = data.loc[valid].copy()
    return data.sort_values(["date", "prize", "prize_index"]).reset_index(drop=True)


def load_predictions() -> pd.DataFrame:
    predictions = pd.read_csv(
        PREDICTION_FILE,
        dtype={"actual": str},
        parse_dates=["date"],
        compression="gzip",
    )
    predictions["actual"] = predictions["actual"].str.zfill(5)
    return predictions.sort_values(["phase", "model", "date"]).reset_index(drop=True)


def match_tickets(tickets: list[str], day: pd.DataFrame, prize_names: list[str]) -> dict:
    stats = {
        prize: {"hit_day": 0, "event_count": 0, "ticket_count": 0, "payout": 0}
        for prize in prize_names
    }
    for ticket in tickets:
        ticket_prizes = set()
        for _, result in day.iterrows():
            prize = str(result["prize"])
            number = str(result["number"]).strip()
            if not number.isdigit() or not number:
                continue
            if ticket[-len(number):] == number:
                ticket_prizes.add(prize)
                stats[prize]["event_count"] += 1
                stats[prize]["payout"] += PAYOUT.get(prize, 0)
        for prize in ticket_prizes:
            stats[prize]["ticket_count"] += 1
    for prize in prize_names:
        stats[prize]["hit_day"] = int(stats[prize]["event_count"] > 0)
    return stats


def run() -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = load_raw()
    predictions = load_predictions()
    prize_names = sorted(raw["prize"].dropna().astype(str).unique())
    daily_rows = []
    prize_rows = []

    for (phase, model), model_predictions in predictions.groupby(["phase", "model"], sort=True):
        for _, prediction in model_predictions.iterrows():
            day = raw.loc[raw["date"].eq(prediction["date"])]
            ranked = str(prediction["top100"]).split()
            for m in TOP_M:
                tickets = ranked[:m]
                stats = match_tickets(tickets, day, prize_names)
                total_payout = sum(item["payout"] for item in stats.values())
                db = stats.get("Đặc biệt", {"event_count": 0, "payout": 0})
                cost = m * COST_PER_TICKET
                daily_rows.append(
                    {
                        "date": prediction["date"],
                        "phase": phase,
                        "model": model,
                        "m": m,
                        "n_tickets": m,
                        "db_hit_day": int(db["event_count"] > 0),
                        "db_hit_tickets": db["event_count"],
                        "db_payout": db["payout"],
                        "other_prize_payout": total_payout - db["payout"],
                        "total_payout": total_payout,
                        "cost": cost,
                        "db_only_profit": db["payout"] - cost,
                        "all_prizes_profit": total_payout - cost,
                    }
                )
                for prize, item in stats.items():
                    prize_rows.append(
                        {
                            "date": prediction["date"],
                            "phase": phase,
                            "model": model,
                            "m": m,
                            "prize": prize,
                            "hit_day": item["hit_day"],
                            "event_count": item["event_count"],
                            "ticket_count": item["ticket_count"],
                            "payout": item["payout"],
                        }
                    )

    daily = pd.DataFrame(daily_rows)
    prizes = pd.DataFrame(prize_rows)
    return daily, prizes


def summarize_daily(daily: pd.DataFrame) -> pd.DataFrame:
    summary = (
        daily.groupby(["phase", "model", "m"], as_index=False)
        .agg(
            days=("date", "nunique"),
            total_tickets=("n_tickets", "sum"),
            db_hit_days=("db_hit_day", "sum"),
            db_hit_tickets=("db_hit_tickets", "sum"),
            db_payout=("db_payout", "sum"),
            other_prize_payout=("other_prize_payout", "sum"),
            total_payout=("total_payout", "sum"),
            cost=("cost", "sum"),
            db_only_profit=("db_only_profit", "sum"),
            all_prizes_profit=("all_prizes_profit", "sum"),
        )
    )
    summary["db_hit_rate"] = summary["db_hit_days"] / summary["days"]
    summary["db_only_roi"] = summary["db_only_profit"] / summary["cost"]
    summary["all_prizes_roi"] = summary["all_prizes_profit"] / summary["cost"]
    summary["other_prize_share_of_payout"] = summary["other_prize_payout"] / summary["total_payout"].replace(0, pd.NA)
    return summary


def summarize_prizes(prizes: pd.DataFrame, daily: pd.DataFrame) -> pd.DataFrame:
    total_tickets = daily.groupby(["phase", "model", "m"], as_index=False)["n_tickets"].sum()
    total_tickets = total_tickets.rename(columns={"n_tickets": "total_tickets"})
    summary = (
        prizes.groupby(["phase", "model", "m", "prize"], as_index=False)
        .agg(
            days=("date", "nunique"),
            hit_days=("hit_day", "sum"),
            event_count=("event_count", "sum"),
            ticket_count=("ticket_count", "sum"),
            payout=("payout", "sum"),
        )
    )
    n_days = daily.groupby(["phase", "model", "m"], as_index=False)["date"].nunique()
    n_days = n_days.rename(columns={"date": "total_days"})
    summary = summary.merge(n_days, on=["phase", "model", "m"], how="left")
    summary = summary.merge(total_tickets, on=["phase", "model", "m"], how="left")
    summary["hit_rate_per_day"] = summary["hit_days"] / summary["total_days"]
    summary["event_rate_per_day"] = summary["event_count"] / summary["total_days"]
    summary["ticket_hit_rate"] = summary["ticket_count"] / summary["total_tickets"]
    return summary


def main() -> None:
    daily, prizes = run()
    daily_summary = summarize_daily(daily)
    prize_summary = summarize_prizes(prizes, daily)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    daily.to_csv(OUTPUT_DIR / "special_prize_daily_results.csv.gz", index=False, compression="gzip", encoding="utf-8-sig")
    prizes.to_csv(OUTPUT_DIR / "special_prize_prize_hits.csv.gz", index=False, compression="gzip", encoding="utf-8-sig")
    daily_summary.to_csv(OUTPUT_DIR / "special_prize_strategy_summary.csv", index=False, encoding="utf-8-sig")
    prize_summary.to_csv(OUTPUT_DIR / "special_prize_by_prize_summary.csv", index=False, encoding="utf-8-sig")
    print("Strategy summary:")
    print(daily_summary.to_string(index=False))
    print("\nBy-prize summary:")
    print(prize_summary.to_string(index=False))


if __name__ == "__main__":
    main()
