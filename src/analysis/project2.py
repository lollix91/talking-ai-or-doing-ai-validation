"""Progetto 2 — Adozione IA e performance (disegno longitudinale a gruppi).

Si confrontano le aziende ad "alta adozione" IA con quelle a "bassa
adozione", nel tempo (2015-2024), su indicatori di efficienza, investimento e
valutazione.

Analisi:
  1. Confronto tra gruppi high vs low: medie + t-test di Welch (FDR).
  2. Modelli panel (regressore principale: log1p(ai_score)):
       A) entity + year FE, controllo log_revenue (specifica originaria);
       B) A con controlli completi (log_assets, leverage, revenue_growth);
       C) B con industry x year FE al posto degli year FE;
       D) B con score laggato (t-1);
       E) B con clustering two-way (impresa + anno);
       F) B con lo score a dizionario fisso (senza lessico generativo);
       G) B con score operativo (Item 1 + 7) e di rischio (Item 1A) insieme,
          con test di Wald di uguaglianza dei due coefficienti.
  3. Robustezza su B: esclusi finanziari/utility, esclusi tech/comunicazione,
     margine estensivo (ai_any), margine intensivo (solo score > 0).
  4. Grafici di trend temporale per gruppo.

Outcome: efficienza/investimento (operating_margin, roa, rd_intensity,
capex_intensity); valutazione (tobin_q, log_market_cap, pe_ratio_historical).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from linearmodels.panel import PanelOLS
from scipy import stats
from statsmodels.stats.multitest import multipletests

from src.analysis import plots, prep

OUTCOMES = ["operating_margin", "roa", "rd_intensity", "capex_intensity"]
ROBUST_OUTCOMES = ["tobin_q", "log_market_cap", "pe_ratio_historical"]
BASE_CONTROLS = ["log_revenue"]
CONTROLS = ["log_assets", "leverage", "revenue_growth"]
MAIN_REGRESSOR = "ai_score_log"
EXCLUDED_SECTORS = {
    "no_fin_util": ["Financial Services", "Financials", "Utilities", "Real Estate"],
    "no_tech": ["Technology", "Communication Services"],
}


def _add_high_low(df: pd.DataFrame) -> pd.DataFrame:
    """high = terzile alto dei positivi; low = non-adopter + terzile basso."""
    df = df.copy()
    df["adoption_hl"] = pd.NA
    df.loc[df["ai_group"] == "high", "adoption_hl"] = "high"
    df.loc[df["ai_group"].isin(["non_adopter", "low"]), "adoption_hl"] = "low"
    return df


def _fdr(pvals: pd.Series) -> pd.Series:
    """Correzione Benjamini-Hochberg su una famiglia di p-value."""
    p = pvals.astype(float)
    ok = p.notna()
    out = pd.Series(np.nan, index=p.index)
    if ok.any():
        out.loc[ok] = multipletests(p.loc[ok], method="fdr_bh")[1]
    return out.round(4)


def _group_means(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    hl = df[df["adoption_hl"].isin(["high", "low"])]
    for y in OUTCOMES + ROBUST_OUTCOMES:
        hi = hl.loc[hl["adoption_hl"] == "high", y].dropna()
        lo = hl.loc[hl["adoption_hl"] == "low", y].dropna()
        if len(hi) < 3 or len(lo) < 3:
            continue
        t, p = stats.ttest_ind(hi, lo, equal_var=False)
        rows.append({"outcome": y, "mean_high": round(hi.mean(), 4), "mean_low": round(lo.mean(), 4),
                     "diff": round(hi.mean() - lo.mean(), 4), "t_stat": round(t, 3),
                     "p_value": round(p, 4), "n_high": len(hi), "n_low": len(lo)})
    out = pd.DataFrame(rows)
    if not out.empty:
        out["p_fdr"] = _fdr(out["p_value"])
    return out


def fit_panel(
    sub: pd.DataFrame,
    outcome: str,
    regressors: list[str],
    controls: list[str],
    industry_year_fe: bool = False,
    two_way_cluster: bool = False,
) -> dict | None:
    """PanelOLS con entity FE e year FE (o industry x year FE), SE cluster."""
    controls = [c for c in controls if c != outcome]
    need = [outcome, *regressors, *controls, "ticker", "fiscal_year"]
    if industry_year_fe:
        need.append("industry")
    d = sub.dropna(subset=need).copy()
    if d["ticker"].nunique() < 10 or len(d) < 30:
        return None
    d = d.set_index(["ticker", "fiscal_year"])
    exog = d[regressors + controls].astype(float)
    try:
        if industry_year_fe:
            ind_year = pd.DataFrame(
                {"ind_year": d["industry"].astype(str) + "_" + d.index.get_level_values(1).astype(str)},
                index=d.index,
            )
            mod = PanelOLS(d[outcome], exog, entity_effects=True, other_effects=ind_year)
        else:
            mod = PanelOLS(d[outcome], exog, entity_effects=True, time_effects=True)
        res = mod.fit(cov_type="clustered", cluster_entity=True, cluster_time=two_way_cluster)
    except Exception as exc:
        print(f"[P2] PanelOLS fallito per {outcome}: {type(exc).__name__}: {exc}")
        return None
    out = {"outcome": outcome, "N": int(res.nobs), "n_firms": d.index.get_level_values(0).nunique(),
           "rsquared_within": round(res.rsquared_within, 4)}
    for i, r in enumerate(regressors):
        sfx = "" if i == 0 else f"_{i + 1}"
        out.update({f"regressor{sfx}": r, f"beta{sfx}": res.params[r], f"se{sfx}": res.std_errors[r],
                    f"p_value{sfx}": res.pvalues[r]})
    if len(regressors) == 2:
        # Wald test of equal coefficients (e.g. operational vs risk-factor score).
        w = res.wald_test(formula=f"{regressors[0]} = {regressors[1]}")
        out.update({"wald_stat_equal": w.stat, "wald_p_equal": w.pval})
    return out


def _with_lag(df: pd.DataFrame, col: str) -> pd.DataFrame:
    lag = df[["ticker", "fiscal_year", col]].dropna().copy()
    lag["fiscal_year"] += 1
    lag = lag.rename(columns={col: f"{col}_l1"}).drop_duplicates(["ticker", "fiscal_year"])
    return df.merge(lag, on=["ticker", "fiscal_year"], how="left")


def _panel_specs(df: pd.DataFrame) -> pd.DataFrame:
    df = _with_lag(df, MAIN_REGRESSOR)
    specs = [
        ("A_baseline", dict(regressors=[MAIN_REGRESSOR], controls=BASE_CONTROLS)),
        ("B_controls", dict(regressors=[MAIN_REGRESSOR], controls=CONTROLS)),
        ("C_industry_year_fe", dict(regressors=[MAIN_REGRESSOR], controls=CONTROLS, industry_year_fe=True)),
        ("D_lagged", dict(regressors=[f"{MAIN_REGRESSOR}_l1"], controls=CONTROLS)),
        ("E_twoway_cluster", dict(regressors=[MAIN_REGRESSOR], controls=CONTROLS, two_way_cluster=True)),
        ("F_fixed_dictionary", dict(regressors=["ai_score_fixed_log"], controls=CONTROLS)),
        ("G_ops_vs_risk", dict(regressors=["ai_ops_log", "ai_risk_log"], controls=CONTROLS)),
    ]
    rows = []
    for y in OUTCOMES + ROBUST_OUTCOMES:
        for name, kw_ in specs:
            r = fit_panel(df, y, **kw_)
            if r:
                rows.append({"spec": name, **r})
    out = pd.DataFrame(rows)
    out["p_fdr"] = out.groupby("spec")["p_value"].transform(_fdr)
    return out


def _robustness(df: pd.DataFrame) -> pd.DataFrame:
    """Specifica B su sottocampioni e su margine estensivo/intensivo."""
    df = df.assign(ai_any_f=df["ai_any"].astype(float))
    variants = {
        **{k: (df[~df["sector"].isin(v)], MAIN_REGRESSOR) for k, v in EXCLUDED_SECTORS.items()},
        "extensive_ai_any": (df, "ai_any_f"),
        "intensive_positive_only": (df[df["ai_score"] > 0], MAIN_REGRESSOR),
    }
    rows = []
    for name, (sub, reg) in variants.items():
        for y in OUTCOMES + ROBUST_OUTCOMES:
            r = fit_panel(sub, y, [reg], CONTROLS)
            if r:
                rows.append({"variant": name, **r})
    out = pd.DataFrame(rows)
    out["p_fdr"] = out.groupby("variant")["p_value"].transform(_fdr)
    return out


def _round(df: pd.DataFrame) -> pd.DataFrame:
    num = df.select_dtypes("number").columns
    return df.assign(**{c: df[c].round(5) for c in num})


def _trend_plots(df: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns

    hl = df[df["adoption_hl"].isin(["high", "low"])]
    for y in OUTCOMES + ["tobin_q"]:
        agg = hl.groupby(["fiscal_year", "adoption_hl"])[y].median().reset_index()
        if agg.empty:
            continue
        fig, ax = plt.subplots(figsize=(6, 3.5))
        sns.lineplot(data=agg, x="fiscal_year", y=y, hue="adoption_hl", marker="o", ax=ax,
                     palette={"high": "darkgreen", "low": "grey"})
        ax.set_title(f"P2 — {y}: high vs low AI adoption (median)")
        ax.set_xlabel("fiscal year")
        ax.legend(title="AI adoption")
        plots.save_fig(fig, f"p2_trend_{y}")


def run() -> None:
    p2 = prep.load_clean_p2()
    if "ai_group" not in p2.columns:
        print("[P2] Variabile IA assente: esegui prima nlp + prep.")
        return
    p2 = _add_high_low(p2)

    gm = _group_means(p2)
    plots.save_table(gm, "p2_group_comparison", index=False)
    print("[P2] Confronto high vs low:\n" + gm.to_string(index=False))

    fe = _round(_panel_specs(p2))
    plots.save_table(fe, "p2_panel_fe", index=False)
    print("[P2] Panel FE:\n" + fe[["outcome", "spec", "beta", "se", "p_value", "p_fdr", "N"]].to_string(index=False))

    rob = _round(_robustness(p2))
    plots.save_table(rob, "p2_robustness", index=False)
    print("[P2] Robustezza:\n" + rob[["outcome", "variant", "beta", "se", "p_value", "N"]].to_string(index=False))

    _trend_plots(p2)


def main() -> None:
    run()


if __name__ == "__main__":
    main()
