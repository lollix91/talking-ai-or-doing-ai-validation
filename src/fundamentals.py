"""Scarica i Fundamentals EODHD per ogni ticker del campione ed estrae le
tabelle necessarie ai due progetti, salvandole in Parquet (formato long).

Output:
  - fundamentals_income_yearly.parquet   (R&D, SG&A, operatingIncome, revenue, ...)
  - fundamentals_cashflow_yearly.parquet (capitalExpenditures, ...)
  - fundamentals_general.parquet         (settore, industria, dipendenti, CIK)
  - fundamentals_highlights.parquet      (PERatio, MarketCapitalization, ...)
  - esg_scores.parquet                   (GovernanceScore + altri score ESG)

Nota: EODHD usa la chiave (con refuso) "quartely" per i dati trimestrali e
"yearly" per gli annuali; il codice gestisce entrambe le grafie.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from tqdm import tqdm

import config
from src.eodhd_client import EODHDClient
from src.sample import load_sample_tickers

INCOME_OUT = config.PROCESSED_DIR / "fundamentals_income_yearly.parquet"
CASHFLOW_OUT = config.PROCESSED_DIR / "fundamentals_cashflow_yearly.parquet"
BALANCE_OUT = config.PROCESSED_DIR / "fundamentals_balance_yearly.parquet"
GENERAL_OUT = config.PROCESSED_DIR / "fundamentals_general.parquet"
HIGHLIGHTS_OUT = config.PROCESSED_DIR / "fundamentals_highlights.parquet"
ESG_OUT = config.PROCESSED_DIR / "esg_scores.parquet"
SHARES_OUT = config.PROCESSED_DIR / "fundamentals_shares_yearly.parquet"
EPS_OUT = config.PROCESSED_DIR / "fundamentals_eps_yearly.parquet"

INCOME_FIELDS = [
    "date",
    "totalRevenue",
    "researchDevelopment",
    "sellingGeneralAdministrative",
    "operatingIncome",
    "netIncome",
    "grossProfit",
]
BALANCE_FIELDS = [
    "date",
    "totalAssets",
    "totalLiab",
    "totalStockholderEquity",
    "shortLongTermDebtTotal",
    "cash",
]
CASHFLOW_FIELDS = [
    "date",
    "capitalExpenditures",
    "investments",
    "totalCashFromOperatingActivities",
]
GENERAL_FIELDS = {
    "CIK": "cik",
    "Sector": "sector",
    "Industry": "industry",
    "FullTimeEmployees": "full_time_employees",
    "IPODate": "ipo_date",
    "Name": "company_name",
    "CountryISO": "country_iso",
}
HIGHLIGHTS_FIELDS = {
    "MarketCapitalization": "market_cap",
    "PERatio": "pe_ratio",
    "EarningsShare": "eps",
    "DividendYield": "dividend_yield",
    "ProfitMargin": "profit_margin",
    "OperatingMarginTTM": "operating_margin_ttm",
    "RevenuePerShareTTM": "revenue_per_share_ttm",
}
ESG_FIELDS = {
    "RatingDate": "rating_date",
    "TotalEsg": "total_esg",
    "EnvironmentScore": "environment_score",
    "SocialScore": "social_score",
    "GovernanceScore": "governance_score",
    "TotalEsgPercentile": "total_esg_percentile",
    "GovernanceScorePercentile": "governance_score_percentile",
    "ControversyLevel": "controversy_level",
}


def _statement_yearly(financials: dict, name: str) -> dict:
    node = financials.get(name, {}) if isinstance(financials, dict) else {}
    # gestisce sia "yearly" sia eventuale refuso
    return node.get("yearly") or node.get("Yearly") or {}


def _extract_statement(ticker: str, yearly: dict, fields: list[str]) -> list[dict]:
    rows = []
    for _date, item in (yearly or {}).items():
        if not isinstance(item, dict):
            continue
        row = {"ticker": ticker}
        for f in fields:
            row[f] = item.get(f)
        rows.append(row)
    return rows


def _to_num(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def _load_cached(ticker: str) -> dict | None:
    """Legge il JSON grezzo gia' scaricato (nessuna chiamata API)."""
    cache = config.RAW_DIR / "fundamentals" / f"{ticker}.json"
    if not cache.exists():
        return None
    try:
        data = json.loads(cache.read_text(encoding="utf-8"))
    except Exception:
        return None
    return data if isinstance(data, dict) else None


def _extract_shares_yearly(ticker: str, data: dict) -> list[dict]:
    """Shares outstanding annuali dal nodo outstandingShares.annual."""
    node = data.get("outstandingShares", {})
    annual = node.get("annual", {}) if isinstance(node, dict) else {}
    rows = []
    items = annual.values() if isinstance(annual, dict) else annual
    for item in items or []:
        if not isinstance(item, dict):
            continue
        rows.append(
            {
                "ticker": ticker,
                "date": item.get("dateFormatted") or item.get("date"),
                "shares_outstanding": item.get("shares"),
            }
        )
    return rows


def _extract_eps_yearly(ticker: str, data: dict) -> list[dict]:
    """EPS annuali effettivi dal nodo Earnings.Annual."""
    node = data.get("Earnings", {})
    annual = node.get("Annual", {}) if isinstance(node, dict) else {}
    rows = []
    for item in (annual or {}).values():
        if not isinstance(item, dict):
            continue
        rows.append(
            {
                "ticker": ticker,
                "date": item.get("date"),
                "eps_actual": item.get("epsActual"),
            }
        )
    return rows


def download_fundamentals(
    tickers: list[str] | None = None,
    client: EODHDClient | None = None,
    use_cache: bool = True,
) -> None:
    client = client or EODHDClient()
    tickers = tickers or load_sample_tickers()

    income_rows: list[dict] = []
    cashflow_rows: list[dict] = []
    balance_rows: list[dict] = []
    general_rows: list[dict] = []
    highlights_rows: list[dict] = []
    esg_rows: list[dict] = []
    shares_rows: list[dict] = []
    eps_rows: list[dict] = []

    for ticker in tqdm(tickers, desc="Fundamentals"):
        cache = config.RAW_DIR / "fundamentals" / f"{ticker}.json"
        data = _load_cached(ticker) if use_cache else None
        if data is None:
            data = client.fundamentals(ticker, cache_path=cache)
        if not data or not isinstance(data, dict):
            continue

        financials = data.get("Financials", {})
        income_rows += _extract_statement(
            ticker, _statement_yearly(financials, "Income_Statement"), INCOME_FIELDS
        )
        cashflow_rows += _extract_statement(
            ticker, _statement_yearly(financials, "Cash_Flow"), CASHFLOW_FIELDS
        )
        balance_rows += _extract_statement(
            ticker, _statement_yearly(financials, "Balance_Sheet"), BALANCE_FIELDS
        )

        general = data.get("General", {}) or {}
        g_row = {"ticker": ticker}
        for src_key, dst in GENERAL_FIELDS.items():
            g_row[dst] = general.get(src_key)
        general_rows.append(g_row)

        highlights = data.get("Highlights", {}) or {}
        h_row = {"ticker": ticker}
        for src_key, dst in HIGHLIGHTS_FIELDS.items():
            h_row[dst] = highlights.get(src_key)
        highlights_rows.append(h_row)

        esg = data.get("ESGScores", {}) or {}
        if esg:
            e_row = {"ticker": ticker}
            for src_key, dst in ESG_FIELDS.items():
                e_row[dst] = esg.get(src_key)
            esg_rows.append(e_row)

        shares_rows += _extract_shares_yearly(ticker, data)
        eps_rows += _extract_eps_yearly(ticker, data)

    # -- Income -------------------------------------------------------------
    income = pd.DataFrame(income_rows)
    if not income.empty:
        income["date"] = pd.to_datetime(income["date"], errors="coerce")
        income = _to_num(income, [c for c in INCOME_FIELDS if c != "date"])
        # Margine operativo derivato (utile per P1/P2).
        income["operating_margin"] = income["operatingIncome"] / income["totalRevenue"]
    income.to_parquet(INCOME_OUT, index=False)

    # -- Cash flow ----------------------------------------------------------
    cashflow = pd.DataFrame(cashflow_rows)
    if not cashflow.empty:
        cashflow["date"] = pd.to_datetime(cashflow["date"], errors="coerce")
        cashflow = _to_num(cashflow, [c for c in CASHFLOW_FIELDS if c != "date"])
    cashflow.to_parquet(CASHFLOW_OUT, index=False)

    # -- Balance sheet ------------------------------------------------------
    balance = pd.DataFrame(balance_rows)
    if not balance.empty:
        balance["date"] = pd.to_datetime(balance["date"], errors="coerce")
        balance = _to_num(balance, [c for c in BALANCE_FIELDS if c != "date"])
    balance.to_parquet(BALANCE_OUT, index=False)

    # -- General ------------------------------------------------------------
    general = pd.DataFrame(general_rows)
    if not general.empty:
        general = _to_num(general, ["full_time_employees"])
    general.to_parquet(GENERAL_OUT, index=False)

    # -- Highlights ---------------------------------------------------------
    highlights = pd.DataFrame(highlights_rows)
    if not highlights.empty:
        highlights = _to_num(
            highlights, [v for v in HIGHLIGHTS_FIELDS.values()]
        )
    highlights.to_parquet(HIGHLIGHTS_OUT, index=False)

    # -- ESG ----------------------------------------------------------------
    esg = pd.DataFrame(esg_rows)
    esg.to_parquet(ESG_OUT, index=False)

    # -- Shares outstanding storico ------------------------------------------
    shares = pd.DataFrame(shares_rows)
    if not shares.empty:
        shares["date"] = pd.to_datetime(shares["date"], errors="coerce")
        shares = _to_num(shares, ["shares_outstanding"])
        shares = shares.dropna(subset=["date"]).drop_duplicates(
            ["ticker", "date"], keep="last"
        )
    shares.to_parquet(SHARES_OUT, index=False)

    # -- EPS storico ----------------------------------------------------------
    eps = pd.DataFrame(eps_rows)
    if not eps.empty:
        eps["date"] = pd.to_datetime(eps["date"], errors="coerce")
        eps = _to_num(eps, ["eps_actual"])
        eps = eps.dropna(subset=["date"]).drop_duplicates(
            ["ticker", "date"], keep="last"
        )
    eps.to_parquet(EPS_OUT, index=False)

    print(
        f"[FUND] income={len(income)} cashflow={len(cashflow)} balance={len(balance)} "
        f"general={len(general)} highlights={len(highlights)} esg={len(esg)} "
        f"shares={len(shares)} eps={len(eps)}"
    )
    print(f"[FUND] Output in {config.PROCESSED_DIR}")


def main() -> None:
    download_fundamentals()


if __name__ == "__main__":
    main()
