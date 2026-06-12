from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a tiny synthetic dataset for smoke tests.")
    parser.add_argument("--output-dir", default="data/raw/DEMO")
    parser.add_argument("--stocks", type=int, default=8)
    parser.add_argument("--days", type=int, default=90)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    dates = pd.bdate_range("2020-01-01", periods=args.days)
    tickers = [f"STK{i:03d}" for i in range(args.stocks)]

    rows = []
    for idx, ticker in enumerate(tickers):
        price = 50 + idx * 3 + np.cumsum(rng.normal(0.03, 0.8, size=len(dates)))
        price = np.maximum(price, 2.0)
        for date, close in zip(dates, price):
            spread = abs(rng.normal(0.0, 0.01))
            open_ = close * (1 + rng.normal(0.0, 0.005))
            high = max(open_, close) * (1 + spread)
            low = min(open_, close) * (1 - spread)
            volume = rng.integers(500_000, 5_000_000)
            rows.append([date, ticker, open_, high, low, close, volume])
    pd.DataFrame(rows, columns=["date", "ticker", "open", "high", "low", "close", "volume"]).to_csv(
        out / "prices.csv", index=False
    )

    news_rows = []
    words = ["strong profit beat", "weak loss risk", "neutral update"]
    for date in dates:
        for ticker in tickers:
            if rng.random() < 0.4:
                text = rng.choice(words)
                news_rows.append([date, ticker, text])
    pd.DataFrame(news_rows, columns=["date", "ticker", "text"]).to_csv(out / "news.csv", index=False)

    sectors = ["Tech", "Finance", "Health"]
    styles = ["large_value", "mid_balanced", "small_growth"]
    pd.DataFrame(
        {
            "ticker": tickers,
            "industry": [sectors[i % len(sectors)] for i in range(len(tickers))],
            "style": [styles[i % len(styles)] for i in range(len(tickers))],
            "market_cap": rng.uniform(1e9, 1e11, size=len(tickers)),
            "book_to_market": rng.uniform(0.1, 1.5, size=len(tickers)),
        }
    ).to_csv(out / "metadata.csv", index=False)

    macro = pd.DataFrame({"date": dates})
    for col in ["GDP", "CPI", "UR", "NFP", "ICI", "CCI", "MPMI", "M1", "M2", "TY10", "FSI", "GPR", "EPU", "SPMI"]:
        macro[col] = rng.normal(0.0, 1.0, size=len(dates)).cumsum()
    macro.to_csv(out / "macro.csv", index=False)
    print(f"Demo raw data written to {out}")


if __name__ == "__main__":
    main()
