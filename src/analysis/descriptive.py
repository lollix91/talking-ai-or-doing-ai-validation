"""Statistiche descrittive del Progetto 2 (Fase 4).

Produce:
  - table1_p2.csv          summary statistics (N, mean, sd, min, quartili, max)
  - corr_p2.png            heatmap delle correlazioni
  - dist_p2_*.png          istogrammi delle variabili chiave
  - coverage_by_year.png   copertura del campione per anno
  - coverage_by_sector.csv copertura per settore
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.analysis import plots, prep

P2_VARS = [
    "operating_margin",
    "roa",
    "rd_intensity",
    "capex_intensity",
    "tobin_q",
    "log_market_cap",
    "pe_ratio_historical",
    "ai_score",
    "ai_score_fixed",
    "log_revenue",
    "log_assets",
    "leverage",
    "revenue_growth",
]


def _summary(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    cols = [c for c in cols if c in df.columns]
    desc = df[cols].describe().T
    desc = desc.rename(
        columns={
            "count": "N",
            "mean": "mean",
            "std": "sd",
            "min": "min",
            "25%": "q25",
            "50%": "median",
            "75%": "q75",
            "max": "max",
        }
    )
    return desc.round(4)


def _corr_heatmap(df: pd.DataFrame, cols: list[str], name: str, title: str) -> None:
    cols = [c for c in cols if c in df.columns]
    corr = df[cols].corr()
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, vmin=-1, vmax=1, ax=ax
    )
    ax.set_title(title)
    plots.save_fig(fig, name)


def _distributions(df: pd.DataFrame, cols: list[str], prefix: str) -> None:
    for c in [c for c in cols if c in df.columns]:
        s = df[c].dropna()
        if s.empty:
            continue
        fig, ax = plt.subplots(figsize=(5, 3.2))
        sns.histplot(s, bins=40, kde=True, ax=ax)
        ax.set_title(f"Distribution: {c}")
        plots.save_fig(fig, f"dist_{prefix}_{c}")


def run() -> None:
    p2 = prep.load_clean_p2()

    # --- Tabella 1 ---------------------------------------------------------
    plots.save_table(_summary(p2, P2_VARS), "table1_p2")
    print("[DESC] Tabella 1 salvata (P2).")

    # --- Correlazioni ------------------------------------------------------
    _corr_heatmap(p2, P2_VARS, "corr_p2", "Correlations — Project 2")

    # --- Distribuzioni -----------------------------------------------------
    _distributions(p2, P2_VARS, "p2")

    # --- Copertura per anno ------------------------------------------------
    cov = (
        p2.groupby("fiscal_year")["ticker"].nunique().rename("n_ticker").reset_index()
    )
    fig, ax = plt.subplots(figsize=(7, 3.5))
    sns.barplot(data=cov, x="fiscal_year", y="n_ticker", color="steelblue", ax=ax)
    ax.set_title("Sample coverage by year (P2)")
    ax.set_xlabel("fiscal year")
    ax.set_ylabel("n. firms")
    plt.xticks(rotation=45)
    plots.save_fig(fig, "coverage_by_year")

    # --- Copertura per settore --------------------------------------------
    if "sector" in p2.columns:
        sec = (
            p2.drop_duplicates("ticker")["sector"]
            .value_counts(dropna=False)
            .rename_axis("sector")
            .reset_index(name="n_ticker")
        )
        plots.save_table(sec, "coverage_by_sector", index=False)

    print(f"[DESC] Figure e tabelle descrittive -> {plots.FIG_DIR} / {plots.TAB_DIR}")


def main() -> None:
    run()


if __name__ == "__main__":
    main()
