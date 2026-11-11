"""Fase A — audit di coerenza tra dati grezzi, elaborati e panel di analisi.

Tabelle prodotte (outputs/tables/):
  - audit_attrition.csv        A1: da costituenti S&P 500 al campione di stima
  - audit_10k_matching.csv     A2: diagnostica dell'abbinamento 10-K <-> firm-year
  - audit_section_coverage.csv A3: copertura del parsing Item 1 / 1A / 7 per anno
  - audit_dictionary_drift.csv A4: score completo vs dizionario fisso per anno
Figura: outputs/figures/audit_dictionary_drift.png
"""
from __future__ import annotations

import pandas as pd

import config
from src.analysis import plots, prep

P = config.PROCESSED_DIR


def _attrition(clean: pd.DataFrame) -> pd.DataFrame:
    pit = pd.read_parquet(P / "sp500_constituents_pit.parquet")
    sample = pd.read_parquet(P / "sample_tickers.parquet")
    income = pd.read_parquet(P / "fundamentals_income_yearly.parquet")
    panel = pd.read_parquet(P / "panel_p2.parquet")
    idx = pd.read_parquet(P / "edgar_10k_index.parquet")

    members = panel.drop_duplicates("ticker")
    members = members[members["index_days"].fillna(0) >= config.MIN_INDEX_DAYS]
    win = panel[panel["ticker"].isin(members["ticker"])]
    win = win[win["fiscal_year"].between(config.STUDY_YEAR_MIN, config.STUDY_YEAR_MAX)]
    core = ["operating_margin", "log_revenue", "ai_score"]
    rows = [
        ("S&P 500 constituents in window (EODHD PIT list)", pit.loc[pit["in_study_window"], "ticker"].nunique(), None),
        ("Sample tickers", sample["ticker"].nunique(), None),
        ("  with EODHD annual financials", income["ticker"].nunique(), None),
        (f"  with index membership >= {config.MIN_INDEX_DAYS} days", members["ticker"].nunique(), None),
        (f"  with firm-years FY{config.STUDY_YEAR_MIN}-{config.STUDY_YEAR_MAX}", win["ticker"].nunique(), len(win)),
        ("  one ticker per issuer (dual share classes removed)", clean["ticker"].nunique(), len(clean)),
        ("  with at least one 10-K on EDGAR", clean.loc[clean["ticker"].isin(idx["ticker"]), "ticker"].nunique(),
         int(clean["ticker"].isin(idx["ticker"]).sum())),
        ("  firm-years matched to an original 10-K", clean.loc[clean["ai_score"].notna(), "ticker"].nunique(),
         int(clean["ai_score"].notna().sum())),
        ("  firm-years with complete core variables", clean.dropna(subset=core)["ticker"].nunique(),
         len(clean.dropna(subset=core))),
    ]
    return pd.DataFrame(rows, columns=["step", "firms", "firm_years"])


def _matching(clean: pd.DataFrame) -> pd.DataFrame:
    ai = pd.read_parquet(P / "ai_adoption_scores.parquet")
    orig = ai[ai["form"] == "10-K"]
    sel = prep.select_filings(ai)
    m = clean.dropna(subset=["ai_score"]).copy()
    gap = (pd.to_datetime(m["report_date"]) - pd.to_datetime(m["period_end"])).dt.days.abs()
    unmatched = clean[clean["ai_score"].isna()]
    rows = [
        ("10-K documents scored", len(ai)),
        ("  of which 10-K/A amendments (excluded)", int((ai["form"] == "10-K/A").sum())),
        ("  median words: original 10-K", int(orig["total_words"].median())),
        ("  median words: 10-K/A", int(ai.loc[ai["form"] == "10-K/A", "total_words"].median())),
        ("Original 10-Ks with duplicate firm-FY (fiscal-year change)", int(orig.duplicated(["ticker", "fiscal_year"]).sum())),
        ("Firm-FY 10-Ks after selection", len(sel)),
        ("Matched firm-years", len(m)),
        ("  |report_date - period_end| <= 10 days", int((gap <= 10).sum())),
        ("  |report_date - period_end| 11-45 days", int(gap.between(11, 45).sum())),
        ("  |report_date - period_end| > 45 days (flag)", int((gap > 45).sum())),
        ("Unmatched firm-years in window", len(unmatched)),
        ("  ticker without any 10-K on EDGAR", int((~unmatched["ticker"].isin(ai["ticker"])).sum())),
    ]
    return pd.DataFrame(rows, columns=["item", "value"])


def _section_coverage(clean: pd.DataFrame) -> pd.DataFrame:
    m = clean.dropna(subset=["ai_score"])
    cov = m.groupby("fiscal_year").agg(
        n=("ticker", "size"),
        business=("words_business", lambda s: (s > 0).mean()),
        risk_factors=("words_risk_factors", lambda s: (s > 0).mean()),
        mdna=("words_mdna", lambda s: (s > 0).mean()),
        all_three=("words_business", lambda s: ((s > 0) & (m.loc[s.index, "words_risk_factors"] > 0)
                                                & (m.loc[s.index, "words_mdna"] > 0)).mean()),
    )
    tot = pd.DataFrame([{
        "n": len(m),
        "business": (m["words_business"] > 0).mean(),
        "risk_factors": (m["words_risk_factors"] > 0).mean(),
        "mdna": (m["words_mdna"] > 0).mean(),
        "all_three": ((m["words_business"] > 0) & (m["words_risk_factors"] > 0) & (m["words_mdna"] > 0)).mean(),
    }], index=pd.Index(["all"], name="fiscal_year"))
    return pd.concat([cov, tot]).round(3)


def _dictionary_drift(clean: pd.DataFrame) -> pd.DataFrame:
    m = clean.dropna(subset=["ai_score"])
    out = m.groupby("fiscal_year").agg(
        n=("ticker", "size"),
        mean_score=("ai_score", "mean"),
        mean_score_fixed=("ai_score_fixed", "mean"),
        share_any=("ai_score", lambda s: (s > 0).mean()),
        share_any_fixed=("ai_score_fixed", lambda s: (s > 0).mean()),
        share_acronym=("n_acronym", lambda s: (s > 0).mean()),
    )
    out["corr_full_fixed"] = m.groupby("fiscal_year")[["ai_score", "ai_score_fixed"]].apply(
        lambda d: d["ai_score"].corr(d["ai_score_fixed"], method="spearman")
    )
    return out.round(3)


def _plot_drift(drift: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.plot(drift.index, drift["share_any"], marker="o", label="any AI term (full dictionary)")
    ax.plot(drift.index, drift["share_any_fixed"], marker="s", label="any AI term (fixed dictionary)")
    ax.plot(drift.index, drift["share_acronym"], marker="^", ls="--", label="uses acronym 'AI'")
    ax.set_xlabel("fiscal year")
    ax.set_ylabel("share of 10-Ks")
    ax.set_title("AI disclosure over time: full vs fixed dictionary")
    ax.legend()
    plots.save_fig(fig, "audit_dictionary_drift")


def run() -> None:
    clean = prep.load_clean_p2()
    tables = {
        "audit_attrition": _attrition(clean),
        "audit_10k_matching": _matching(clean),
        "audit_section_coverage": _section_coverage(clean),
        "audit_dictionary_drift": _dictionary_drift(clean),
    }
    for name, t in tables.items():
        plots.save_table(t, name, index=name in ("audit_section_coverage", "audit_dictionary_drift"))
        print(f"[AUDIT] {name}\n{t.to_string()}\n")
    _plot_drift(tables["audit_dictionary_drift"])


def main() -> None:
    run()


if __name__ == "__main__":
    main()
