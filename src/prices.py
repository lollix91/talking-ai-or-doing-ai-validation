"""Scarica i prezzi End-of-Day giornalieri (OHLCV) per il campione.

Copre l'intera finestra dello studio (config.START_DATE..END_DATE), cosi'
serve sia alla volatilita' di P1 (ultimo anno) sia alle analisi di
valutazione longitudinali di P2. Output in Parquet formato long.
"""
from __future__ import annotations

import pandas as pd
from tqdm import tqdm

import config
from src.eodhd_client import EODHDClient
from src.sample import load_sample_tickers

PRICES_OUT = config.PROCESSED_DIR / "prices_eod.parquet"

FIELDS = ["date", "open", "high", "low", "close", "adjusted_close", "volume"]


def download_prices(
    tickers: list[str] | None = None, client: EODHDClient | None = None
) -> pd.DataFrame:
    client = client or EODHDClient()
    tickers = tickers or load_sample_tickers()

    frames: list[pd.DataFrame] = []
    for ticker in tqdm(tickers, desc="Prezzi EOD"):
        cache = config.RAW_DIR / "eod" / f"{ticker}.json"
        data = client.eod(
            ticker, config.START_DATE, config.END_DATE, cache_path=cache
        )
        if not data or not isinstance(data, list):
            continue
        df = pd.DataFrame.from_records(data)
        if df.empty:
            continue
        df["ticker"] = ticker
        frames.append(df)

    if not frames:
        result = pd.DataFrame(columns=["ticker"] + FIELDS)
    else:
        result = pd.concat(frames, ignore_index=True)
        result["date"] = pd.to_datetime(result["date"], errors="coerce")
        for c in ["open", "high", "low", "close", "adjusted_close", "volume"]:
            if c in result.columns:
                result[c] = pd.to_numeric(result[c], errors="coerce")
        result = result[["ticker"] + [c for c in FIELDS if c in result.columns]]
        result = result.sort_values(["ticker", "date"]).reset_index(drop=True)

    result.to_parquet(PRICES_OUT, index=False)
    print(
        f"[PRICES] {result['ticker'].nunique() if not result.empty else 0} ticker, "
        f"{len(result)} righe -> {PRICES_OUT}"
    )
    return result


def main() -> None:
    download_prices()


if __name__ == "__main__":
    main()
