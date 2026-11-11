"""Scarica la mappatura ufficiale SEC CIK <-> Ticker e la salva in Parquet.

Fonte: https://www.sec.gov/files/company_tickers.json
Serve per collegare i dati SEC/EDGAR (CIK) ai ticker usati su EODHD.
"""
from __future__ import annotations

import pandas as pd
import requests

import config

OUTPUT = config.PROCESSED_DIR / "sec_cik_ticker.parquet"


def download_sec_mapping() -> pd.DataFrame:
    resp = requests.get(
        config.SEC_CIK_TICKER_URL,
        headers={"User-Agent": config.SEC_USER_AGENT},
        timeout=config.REQUEST_TIMEOUT_SEC,
    )
    resp.raise_for_status()
    raw = resp.json()

    # Il JSON e' un dict indicizzato per posizione: {"0": {...}, "1": {...}}
    records = list(raw.values())
    df = pd.DataFrame.from_records(records)
    # Colonne attese: cik_str, ticker, title
    df = df.rename(columns={"cik_str": "cik", "title": "company_name"})
    # CIK a 10 cifre con zero-padding (formato standard EDGAR).
    df["cik"] = df["cik"].astype(int)
    df["cik_padded"] = df["cik"].astype(str).str.zfill(10)
    df["ticker"] = df["ticker"].astype(str).str.upper()
    df = df[["cik", "cik_padded", "ticker", "company_name"]]

    # Salva anche il JSON grezzo per riproducibilita'.
    (config.RAW_DIR / "sec_company_tickers.json").write_bytes(resp.content)
    df.to_parquet(OUTPUT, index=False)
    return df


def main() -> None:
    df = download_sec_mapping()
    print(f"[SEC] Salvate {len(df)} righe -> {OUTPUT}")


if __name__ == "__main__":
    main()
