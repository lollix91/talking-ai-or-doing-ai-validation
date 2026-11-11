"""Fase 2: feature engineering dai Parquet grezzi -> panel analysis-ready.

Costruisce il panel (azienda x anno) del Progetto 2:

  panel_p2.parquet   -- adozione IA e performance longitudinali
                      (efficienza, investimento, valutazione)

Tutte le metriche sono allineate all'anno fiscale di riferimento del bilancio
(usa la data del Income Statement, non la data di filing).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import config

P2_OUT = config.PROCESSED_DIR / "panel_p2.parquet"


def _fiscal_year(df: pd.DataFrame) -> pd.DataFrame:
    """Aggiunge colonna 'fiscal_year' estratta dalla data del bilancio."""
    df = df.copy()
    df["fiscal_year"] = pd.to_datetime(df["date"]).dt.year
    return df


def _index_days() -> pd.DataFrame:
    """Giorni di appartenenza all'S&P 500 dentro la finestra di studio."""
    pit = pd.read_parquet(config.PROCESSED_DIR / "sp500_constituents_pit.parquet")
    for ticker, (start, end) in config.INDEX_MEMBERSHIP_CORRECTIONS.items():
        m = pit["ticker"] == ticker
        pit.loc[m, "start_date"] = pd.Timestamp(start)
        pit.loc[m, "end_date"] = pd.NaT if end is None else pd.Timestamp(end)
    lo, hi = pd.Timestamp(config.START_DATE), pd.Timestamp(config.END_DATE)
    start = pd.to_datetime(pit["start_date"]).fillna(lo).clip(lower=lo)
    end = pd.to_datetime(pit["end_date"]).fillna(hi).clip(upper=hi)
    pit = pit.assign(index_days=(end - start).dt.days.clip(lower=0))
    return pit.groupby("ticker", as_index=False)["index_days"].sum()


# ---------------------------------------------------------------------------
# Progetto 2: efficienza, investimento, valutazione (longitudinale)
# ---------------------------------------------------------------------------
def build_panel_p2() -> pd.DataFrame:
    income = pd.read_parquet(
        config.PROCESSED_DIR / "fundamentals_income_yearly.parquet"
    )
    cashflow = pd.read_parquet(
        config.PROCESSED_DIR / "fundamentals_cashflow_yearly.parquet"
    )
    general = pd.read_parquet(config.PROCESSED_DIR / "fundamentals_general.parquet")
    highlights = pd.read_parquet(
        config.PROCESSED_DIR / "fundamentals_highlights.parquet"
    )
    prices = pd.read_parquet(config.PROCESSED_DIR / "prices_eod.parquet")
    shares = pd.read_parquet(
        config.PROCESSED_DIR / "fundamentals_shares_yearly.parquet"
    )
    eps = pd.read_parquet(config.PROCESSED_DIR / "fundamentals_eps_yearly.parquet")
    balance = pd.read_parquet(config.PROCESSED_DIR / "fundamentals_balance_yearly.parquet")

    # Bilanci duplicati per lo stesso anno di calendario (cambio di fine anno
    # fiscale o righe ripetute): si tiene la chiusura piu' recente.
    income = _fiscal_year(income).dropna(subset=["date"])
    income = income.sort_values("date").drop_duplicates(["ticker", "fiscal_year"], keep="last")
    cashflow = _fiscal_year(cashflow).dropna(subset=["date"])
    cashflow = cashflow.sort_values("date").drop_duplicates(["ticker", "fiscal_year"], keep="last")
    balance = _fiscal_year(balance).dropna(subset=["date"])
    balance = balance.sort_values("date").drop_duplicates(["ticker", "fiscal_year"], keep="last")

    # --- base: income + cashflow -----------------------------------------
    # 'date' dell'Income Statement = fine anno fiscale (period_end): serve
    # per allineare i prezzi EOD alla data di bilancio.
    p2 = income[
        [
            "ticker",
            "date",
            "fiscal_year",
            "totalRevenue",
            "researchDevelopment",
            "operatingIncome",
            "operating_margin",
            "netIncome",
        ]
    ].rename(columns={"date": "period_end"}).copy()

    cf = cashflow[["ticker", "fiscal_year", "capitalExpenditures"]].copy()
    p2 = p2.merge(cf, on=["ticker", "fiscal_year"], how="left")
    bs = balance[
        ["ticker", "fiscal_year", "totalAssets", "totalLiab",
         "totalStockholderEquity", "shortLongTermDebtTotal", "cash"]
    ].copy()
    p2 = p2.merge(bs, on=["ticker", "fiscal_year"], how="left")

    # --- R&D mancante: non-reporting = 0 + dummy (convenzione Compustat) ---
    p2["rd_missing"] = p2["researchDevelopment"].isna().astype(int)
    p2["researchDevelopment"] = p2["researchDevelopment"].fillna(0.0)

    # --- intensita' di investimento --------------------------------------
    for col in ["researchDevelopment", "capitalExpenditures"]:
        ratio = col + "_intensity"
        p2[ratio] = np.where(
            p2["totalRevenue"] > 0, p2[col] / p2["totalRevenue"], np.nan
        )

    # --- settore / industria ---------------------------------------------
    gen = general[["ticker", "cik", "sector", "industry"]].copy()
    p2 = p2.merge(gen, on="ticker", how="left")

    # --- valutazione di mercato: P/E e market cap storici veri ------------
    # Ricostruzione corretta (non piu' approssimazione):
    #   market_cap_t = shares_outstanding_t * close_fine_anno_fiscale
    #   pe_ratio_t   = close_fine_anno_fiscale / eps_actual_t
    # shares ed EPS storici dal nodo outstandingShares.annual / Earnings.Annual
    # dei fundamentals EODHD; prezzo = ultimo close <= period_end.
    hl = highlights[["ticker", "pe_ratio", "market_cap"]].drop_duplicates("ticker")
    p2 = p2.merge(hl, on="ticker", how="left")

    shares = shares.copy()
    shares["fiscal_year"] = pd.to_datetime(shares["date"]).dt.year
    shares = (
        shares[["ticker", "fiscal_year", "shares_outstanding"]]
        .drop_duplicates(["ticker", "fiscal_year"], keep="last")
    )
    p2 = p2.merge(shares, on=["ticker", "fiscal_year"], how="left")

    eps = eps.copy()
    eps["fiscal_year"] = pd.to_datetime(eps["date"]).dt.year
    eps = (
        eps[["ticker", "fiscal_year", "eps_actual"]]
        .drop_duplicates(["ticker", "fiscal_year"], keep="last")
    )
    p2 = p2.merge(eps, on=["ticker", "fiscal_year"], how="left")

    # Ultimo close *adjusted* <= period_end. NB: shares/EPS di EODHD sono
    # split-adjusted, quindi il prezzo deve essere adjusted_close (non il
    # close nominale), altrimenti market cap e P/E sono gonfiati per gli
    # anni precedenti agli split. Tolleranza 10 giorni per evitare match
    # con prezzi troppo vecchi (es. ticker delisted).
    prices = prices.copy()
    prices["date"] = pd.to_datetime(prices["date"])
    p2 = p2.sort_values("period_end")
    px = prices.sort_values("date")[["ticker", "date", "adjusted_close"]]
    p2 = pd.merge_asof(
        p2,
        px,
        left_on="period_end",
        right_on="date",
        by="ticker",
        direction="backward",
        tolerance=pd.Timedelta(days=10),
    ).rename(columns={"adjusted_close": "adj_close_fye"}).drop(columns=["date"])

    p2["market_cap_historical"] = np.where(
        p2["shares_outstanding"] > 0,
        p2["shares_outstanding"] * p2["adj_close_fye"],
        np.nan,
    )
    # P/E definito solo per utili positivi (P/E negativo non interpretabile).
    p2["pe_ratio_historical"] = np.where(
        (p2["adj_close_fye"] > 0) & (p2["eps_actual"] > 0),
        p2["adj_close_fye"] / p2["eps_actual"],
        np.nan,
    )

    p2 = p2.rename(
        columns={
            "totalRevenue": "revenue",
            "researchDevelopment": "rd_expense",
            "operatingIncome": "operating_income",
            "netIncome": "net_income",
            "capitalExpenditures": "capex",
            "researchDevelopment_intensity": "rd_intensity",
            "capitalExpenditures_intensity": "capex_intensity",
            "totalAssets": "total_assets",
            "totalLiab": "total_liab",
            "totalStockholderEquity": "equity",
            "shortLongTermDebtTotal": "total_debt",
        }
    )

    # --- valutazione relativa, redditivita' e controlli --------------------
    # Tobin's Q approssimato: (market value equity + passivita') / attivo.
    pos_assets = p2["total_assets"] > 0
    p2["tobin_q"] = np.where(
        pos_assets, (p2["market_cap_historical"] + p2["total_liab"]) / p2["total_assets"], np.nan
    )
    p2["roa"] = np.where(pos_assets, p2["net_income"] / p2["total_assets"], np.nan)
    p2["leverage"] = np.where(pos_assets, p2["total_debt"].fillna(0) / p2["total_assets"], np.nan)
    p2["log_assets"] = np.log(p2["total_assets"].where(pos_assets))
    p2["log_market_cap"] = np.log(p2["market_cap_historical"].where(p2["market_cap_historical"] > 0))

    # Crescita dei ricavi su anni fiscali consecutivi.
    p2 = p2.sort_values(["ticker", "fiscal_year"])
    prev = p2.groupby("ticker")[["fiscal_year", "revenue"]].shift(1)
    p2["revenue_growth"] = np.where(
        (prev["fiscal_year"] == p2["fiscal_year"] - 1) & (prev["revenue"] > 0),
        p2["revenue"] / prev["revenue"] - 1,
        np.nan,
    )

    # Giorni di appartenenza all'S&P 500 nella finestra di studio: le
    # spin-off entrano nell'indice per 1-3 giorni (prassi S&P) e non sono
    # veri costituenti; il filtro e' applicato in analysis/prep.py.
    p2 = p2.merge(_index_days(), on="ticker", how="left")

    # Colonne finali ordinate.
    final_cols = [
        "ticker",
        "cik",
        "fiscal_year",
        "period_end",
        "sector",
        "industry",
        "revenue",
        "operating_income",
        "operating_margin",
        "rd_expense",
        "rd_missing",
        "rd_intensity",
        "capex",
        "capex_intensity",
        "shares_outstanding",
        "eps_actual",
        "adj_close_fye",
        "market_cap_historical",
        "pe_ratio",
        "pe_ratio_historical",
        "market_cap",
        "net_income",
        "total_assets",
        "total_liab",
        "equity",
        "total_debt",
        "tobin_q",
        "roa",
        "leverage",
        "log_assets",
        "log_market_cap",
        "revenue_growth",
        "index_days",
    ]
    p2 = p2[[c for c in final_cols if c in p2.columns]]
    p2 = p2.sort_values(["ticker", "fiscal_year"]).reset_index(drop=True)
    p2.to_parquet(P2_OUT, index=False)
    print(f"[P2] Panel {p2.shape} -> {P2_OUT}")
    print(f"[P2] Colonne: {list(p2.columns)}")
    return p2


def main() -> None:
    build_panel_p2()


if __name__ == "__main__":
    main()
