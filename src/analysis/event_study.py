"""Event-study / DiD attorno al rilascio di ChatGPT (30 novembre 2022).

Disegno:
  - Trattamento (predeterminato): media di log1p(ai_score) sui 10-K degli
    anni fiscali 2019-2021, cioe' *prima* dello shock.
  - Post-periodo: fiscal_year >= 2023 (il FY2022 e' il confine: i 10-K del
    FY2022 riflettono in gran parte il periodo pre-rilascio).
  - Specifiche (entity + year FE, controlli, SE cluster per impresa):
      a) did_continuous:  baseline_ai x Post;
      b) did_binary:      treat_high x Post (baseline sopra la mediana dei
                          baseline positivi);
      c) did_continuous_detrended: come (a) ma su outcome depurato dal trend
         lineare specifico d'impresa stimato sul pre-periodo FY2015-2021
         (risponde ai pre-trend non piatti dell'event-study);
      d) did_binary_matched: (b) su un campione abbinato 1:1 (nearest
         neighbour con reinserimento, stesso settore) su dimensione, R&D,
         Tobin's Q e trend pre-periodo dell'outcome.
      e) event-study: baseline_ai x 1[year = t], rif. 2022, grezzo e detrended.
  Nota: Callaway-Sant'Anna non si applica (trattamento non scaglionato:
  shock unico con intensita' continua); (c) e (d) sono le alternative
  standard per pre-trend in un DiD a tempo unico.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from linearmodels.panel import PanelOLS

from src.analysis import plots, prep
from src.analysis.project2 import CONTROLS, OUTCOMES, ROBUST_OUTCOMES, _fdr

BASELINE_YEARS = (2019, 2021)
POST_YEAR = 2023
REF_YEAR = 2022
TREND_YEARS = (2015, 2021)
ES_OUTCOMES = ["log_market_cap", "tobin_q", "operating_margin", "roa"]


def _build_baseline(df: pd.DataFrame) -> pd.DataFrame:
    base = (
        df.loc[df["fiscal_year"].between(*BASELINE_YEARS) & df["ai_score_log"].notna()]
        .groupby("ticker")["ai_score_log"].mean().rename("baseline_ai").reset_index()
    )
    df = df.merge(base, on="ticker", how="left")
    df["post"] = (df["fiscal_year"] >= POST_YEAR).astype(int)
    pos = df.drop_duplicates("ticker")["baseline_ai"]
    med = pos[pos > 0].median()
    df["treat_high"] = np.where(df["baseline_ai"].notna(), (df["baseline_ai"] > med).astype(float), np.nan)
    return df


def _detrend(df: pd.DataFrame, y: str) -> pd.Series:
    """Residuo rispetto al trend lineare d'impresa stimato su TREND_YEARS."""
    pre = df[df["fiscal_year"].between(*TREND_YEARS)].dropna(subset=[y])
    coef = pre.groupby("ticker").apply(
        lambda g: pd.Series(np.polyfit(g["fiscal_year"], g[y], 1) if len(g) >= 4 else [np.nan, np.nan],
                            index=["slope", "intercept"]),
        include_groups=False,
    )
    d = df[["ticker", "fiscal_year", y]].merge(coef, left_on="ticker", right_index=True, how="left")
    return (d[y] - (d["intercept"] + d["slope"] * d["fiscal_year"])).set_axis(df.index)


def _fit(d: pd.DataFrame, y: str, regs: list[str], weights: pd.Series | None = None):
    controls = [c for c in CONTROLS if c != y]
    need = [y, *regs, *controls, "ticker", "fiscal_year"]
    d = d.dropna(subset=need)
    if d["ticker"].nunique() < 10 or len(d) < 30:
        return None, d
    d = d.set_index(["ticker", "fiscal_year"])
    w = None if weights is None else d["_w"]
    res = PanelOLS(d[y], d[regs + controls].astype(float), entity_effects=True, time_effects=True,
                   weights=w).fit(cov_type="clustered", cluster_entity=True)
    return res, d


def _did_row(res, d, y, spec, reg="treat_x_post") -> dict:
    return {"outcome": y, "spec": spec, "beta": res.params[reg], "se": res.std_errors[reg],
            "p_value": res.pvalues[reg], "N": int(res.nobs), "n_firms": d.index.get_level_values(0).nunique()}


def _matched_sample(df: pd.DataFrame, y: str) -> pd.DataFrame:
    """Abbinamento 1:1 (con reinserimento) treated -> control nello stesso
    settore, su caratteristiche medie pre-periodo standardizzate."""
    pre = df[df["fiscal_year"].between(*BASELINE_YEARS)]
    feats = pre.groupby("ticker").agg(
        size=("log_assets", "mean"), rd=("rd_intensity", "mean"), q=("tobin_q", "mean"),
        lvl=(y, "mean"), sector=("sector", "first"), treat=("treat_high", "first"),
    ).dropna()
    early = df[df["fiscal_year"].between(*TREND_YEARS)].groupby("ticker")[y].mean()
    feats["trend"] = feats["lvl"] - early.reindex(feats.index)
    feats = feats.dropna()
    cols = ["size", "rd", "q", "lvl", "trend"]
    z = (feats[cols] - feats[cols].mean()) / feats[cols].std()
    weights: dict[str, float] = {}
    for sector, g in feats.groupby("sector"):
        tr, co = g[g["treat"] == 1].index, g[g["treat"] == 0].index
        if len(tr) == 0 or len(co) == 0:
            continue
        dist = ((z.loc[tr].to_numpy()[:, None, :] - z.loc[co].to_numpy()[None, :, :]) ** 2).sum(-1)
        for t, j in zip(tr, dist.argmin(1)):
            weights[t] = weights.get(t, 0) + 1
            weights[co[j]] = weights.get(co[j], 0) + 1
    m = df[df["ticker"].isin(weights)].copy()
    m["_w"] = m["ticker"].map(weights).astype(float)
    return m


def _event_study(df: pd.DataFrame, y: str, label: str) -> pd.DataFrame | None:
    d = df.copy()
    years = sorted(int(t) for t in d["fiscal_year"].dropna().unique() if t != REF_YEAR)
    cols = []
    for t in years:
        d[f"es_{t}"] = d["baseline_ai"] * (d["fiscal_year"] == t).astype(float)
        cols.append(f"es_{t}")
    res, _ = _fit(d.dropna(subset=["baseline_ai"]), y, cols)
    if res is None:
        return None
    rows = [{"outcome": y, "version": label, "fiscal_year": t, "beta": res.params[f"es_{t}"],
             "se": res.std_errors[f"es_{t}"]} for t in years]
    rows.append({"outcome": y, "version": label, "fiscal_year": REF_YEAR, "beta": 0.0, "se": 0.0})
    return pd.DataFrame(rows).sort_values("fiscal_year")


def _plot_event_study(es: pd.DataFrame, y: str) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.axhline(0, color="grey", lw=0.8)
    ax.axvline(REF_YEAR + 0.5, color="red", ls="--", lw=0.8, label="ChatGPT release")
    for (label, g), color, off in zip(es.groupby("version"), ["darkgreen", "steelblue"], [-0.1, 0.1]):
        ax.errorbar(g["fiscal_year"] + off, g["beta"], yerr=1.96 * g["se"], marker="o", capsize=3,
                    color=color, label=label)
    ax.set_title(f"Event study — {y} (baseline AI 2019-21 x year)")
    ax.set_xlabel("fiscal year")
    ax.set_ylabel("beta (ref: 2022)")
    ax.legend()
    plots.save_fig(fig, f"p2_event_study_{y}")


def run() -> None:
    p2 = prep.load_clean_p2()
    p2 = _build_baseline(p2)
    print(f"[ES] Imprese con baseline 2019-21: {p2.drop_duplicates('ticker')['baseline_ai'].notna().sum()}")

    rows = []
    for y in OUTCOMES + ROBUST_OUTCOMES:
        d = p2.copy()
        d[f"{y}_dt"] = _detrend(d, y)
        for spec, col, yy, sub in [
            ("did_continuous", "baseline_ai", y, d),
            ("did_binary", "treat_high", y, d),
            ("did_continuous_detrended", "baseline_ai", f"{y}_dt", d),
        ]:
            sub = sub.assign(treat_x_post=sub[col] * sub["post"])
            res, dd = _fit(sub, yy, ["treat_x_post"])
            if res is not None:
                rows.append(_did_row(res, dd, y, spec))
        m = _matched_sample(d, y).assign(treat_x_post=lambda x: x["treat_high"] * x["post"])
        res, dd = _fit(m, y, ["treat_x_post"], weights=m["_w"])
        if res is not None:
            rows.append(_did_row(res, dd, y, "did_binary_matched"))
    did = pd.DataFrame(rows)
    did["p_fdr"] = did.groupby("spec")["p_value"].transform(_fdr)
    num = did.select_dtypes("number").columns
    did[num] = did[num].round(5)
    plots.save_table(did, "p2_event_study_did", index=False)
    print("[ES] DiD post-ChatGPT:\n" + did.to_string(index=False))

    for y in ES_OUTCOMES:
        d = p2.copy()
        d[f"{y}_dt"] = _detrend(d, y)
        parts = [_event_study(d, y, "raw"), _event_study(d.drop(columns=[y]).rename(columns={f"{y}_dt": y}), y,
                                                         "detrended")]
        es = pd.concat([p for p in parts if p is not None])
        plots.save_table(es.round(5), f"p2_event_study_{y}", index=False)
        _plot_event_study(es, y)


def main() -> None:
    run()


if __name__ == "__main__":
    main()
