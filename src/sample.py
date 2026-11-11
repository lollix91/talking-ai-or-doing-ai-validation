"""Costruisce il campione point-in-time dei costituenti dell'S&P 500.

Usa l'endpoint fundamentals dell'indice (GSPC.INDX) con
filter=HistoricalTickerComponents, che restituisce per ogni titolo le date di
ingresso/uscita dall'indice. Da qui:
  1. salva lo storico completo di membership (per analisi point-in-time);
  2. estrae la lista dei ticker da scaricare = tutti quelli che sono stati
     nell'indice in un qualsiasi momento della finestra dello studio.

Cosi' si evita il survivorship bias (fondamentale per un lavoro Q1).
"""
from __future__ import annotations

import pandas as pd

import config
from src.eodhd_client import EODHDClient

MEMBERSHIP_OUTPUT = config.PROCESSED_DIR / "sp500_constituents_pit.parquet"
SAMPLE_OUTPUT = config.PROCESSED_DIR / "sample_tickers.parquet"


def _to_eodhd_ticker(code: str, exchange: str | None) -> str:
    exch = (exchange or config.US_EXCHANGE_SUFFIX).strip()
    return f"{code}.{exch}"


def build_sample(client: EODHDClient | None = None) -> pd.DataFrame:
    client = client or EODHDClient()
    data = client.index_filter(config.INDEX_CODE, "HistoricalTickerComponents")
    if not data:
        raise RuntimeError(
            "Nessun dato HistoricalTickerComponents restituito per "
            f"{config.INDEX_CODE}. Verifica che il tuo piano EODHD includa i "
            "fundamentals degli indici."
        )

    # Il filtro puo' restituire una lista oppure un dict indicizzato.
    records = list(data.values()) if isinstance(data, dict) else list(data)
    df = pd.DataFrame.from_records(records)

    # Normalizza i nomi colonna (robusto a piccole variazioni dell'API).
    rename = {
        "Code": "code",
        "Name": "company_name",
        "StartDate": "start_date",
        "EndDate": "end_date",
        "IsActiveNow": "is_active_now",
        "IsDelisted": "is_delisted",
        "Exchange": "exchange",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})

    df["start_date"] = pd.to_datetime(df.get("start_date"), errors="coerce")
    df["end_date"] = pd.to_datetime(df.get("end_date"), errors="coerce")

    window_start = pd.Timestamp(config.START_DATE)
    window_end = pd.Timestamp(config.END_DATE)

    # Un titolo appartiene al campione se la sua permanenza nell'indice si
    # sovrappone alla finestra dello studio: start <= window_end e
    # (end e' nullo -> tuttora dentro, oppure end >= window_start).
    overlaps = (df["start_date"].fillna(window_start) <= window_end) & (
        df["end_date"].isna() | (df["end_date"] >= window_start)
    )
    df["in_study_window"] = overlaps

    df["ticker"] = [
        _to_eodhd_ticker(c, df.get("exchange", pd.Series()).get(i))
        if "exchange" in df.columns
        else _to_eodhd_ticker(c, None)
        for i, c in df["code"].items()
    ]

    df.to_parquet(MEMBERSHIP_OUTPUT, index=False)

    sample = (
        df.loc[df["in_study_window"], ["code", "ticker", "company_name"]]
        .drop_duplicates(subset=["ticker"])
        .sort_values("ticker")
        .reset_index(drop=True)
    )
    sample.to_parquet(SAMPLE_OUTPUT, index=False)
    return sample


def load_sample_tickers() -> list[str]:
    df = pd.read_parquet(SAMPLE_OUTPUT)
    return df["ticker"].tolist()


def main() -> None:
    sample = build_sample()
    print(
        f"[SAMPLE] Membership storica -> {MEMBERSHIP_OUTPUT}\n"
        f"[SAMPLE] {len(sample)} ticker nel campione (finestra "
        f"{config.START_DATE}..{config.END_DATE}) -> {SAMPLE_OUTPUT}"
    )


if __name__ == "__main__":
    main()
