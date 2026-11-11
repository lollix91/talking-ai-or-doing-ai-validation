"""Reazione del mercato alla disclosure IA: abnormal return attorno al 10-K.

Per ogni 10-K abbinato al panel:
  - giorno 0 = primo giorno di borsa >= filing_date;
  - stima Fama-French 5 fattori sui rendimenti in eccesso, finestra
    [-250, -30] giorni di borsa (minimo 120 osservazioni);
  - CAR[-1,+1] (principale) e CAR[0,+5] = somma degli abnormal return.
Regressione cross-section: CAR ~ log1p(ai_score) + controlli + industry FE +
year FE, SE clusterizzati per impresa. Varianti: score a dizionario fisso,
score operativo vs rischio, sotto-periodi pre/post ChatGPT.

Fattori: Kenneth R. French Data Library (scaricati in data/raw/ff5_daily.csv).
Output: data/processed/car_10k.parquet, outputs/tables/p2_car_regressions.csv
"""
from __future__ import annotations

import io
import zipfile

import numpy as np
import pandas as pd
import requests
import statsmodels.formula.api as smf

import config
from src.analysis import plots, prep

FF5_URL = ("https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
           "F-F_Research_Data_5_Factors_2x3_daily_CSV.zip")
FF5_CSV = config.RAW_DIR / "ff5_daily.csv"
CAR_OUT = config.PROCESSED_DIR / "car_10k.parquet"
EST_WINDOW = (-250, -30)
MIN_EST_OBS = 120
EVENT_WINDOWS = {"car_m1_p1": (-1, 1), "car_0_p5": (0, 5)}
FACTORS = ["mkt_rf", "smb", "hml", "rmw", "cma"]
CONTROLS = ["log_market_cap", "tobin_q", "leverage", "roa", "revenue_growth"]


def load_ff5() -> pd.DataFrame:
    if not FF5_CSV.exists():
        r = requests.get(FF5_URL, timeout=60, headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
        z = zipfile.ZipFile(io.BytesIO(r.content))
        FF5_CSV.write_bytes(z.read(z.namelist()[0]))
    lines = FF5_CSV.read_text().splitlines()
    start = next(i for i, l in enumerate(lines) if l.startswith(",Mkt-RF"))
    rows = [l for l in lines[start + 1:] if l[:8].strip().isdigit()]
    ff = pd.read_csv(io.StringIO("\n".join(rows)), header=None,
                     names=["date", "mkt_rf", "smb", "hml", "rmw", "cma", "rf"])
    ff["date"] = pd.to_datetime(ff["date"].astype(str), format="%Y%m%d")
    ff[FACTORS + ["rf"]] = ff[FACTORS + ["rf"]] / 100
    return ff


def compute_car(events: pd.DataFrame) -> pd.DataFrame:
    """events: ticker, accession, filing_date."""
    px = pd.read_parquet(config.PROCESSED_DIR / "prices_eod.parquet",
                         columns=["ticker", "date", "adjusted_close"])
    px["date"] = pd.to_datetime(px["date"])
    px = px[px["ticker"].isin(events["ticker"])].sort_values(["ticker", "date"])
    px["ret"] = px.groupby("ticker")["adjusted_close"].pct_change()
    ff = load_ff5()
    px = px.merge(ff, on="date", how="inner")
    px["exret"] = px["ret"] - px["rf"]

    rows = []
    for ticker, ev in events.groupby("ticker"):
        p = px[px["ticker"] == ticker].dropna(subset=["exret"]).reset_index(drop=True)
        if p.empty:
            continue
        dates = p["date"].to_numpy()
        X_all = np.column_stack([np.ones(len(p)), p[FACTORS].to_numpy()])
        y_all = p["exret"].to_numpy()
        for _, e in ev.iterrows():
            t0 = np.searchsorted(dates, np.datetime64(pd.Timestamp(e["filing_date"])))
            lo, hi = t0 + EST_WINDOW[0], t0 + EST_WINDOW[1]
            if lo < 0 or t0 + max(w[1] for w in EVENT_WINDOWS.values()) >= len(p):
                continue
            beta, *_ = np.linalg.lstsq(X_all[lo:hi + 1], y_all[lo:hi + 1], rcond=None)
            if hi - lo + 1 < MIN_EST_OBS:
                continue
            ar = y_all - X_all @ beta
            out = {"ticker": ticker, "accession": e["accession"]}
            for name, (a, b) in EVENT_WINDOWS.items():
                out[name] = ar[t0 + a:t0 + b + 1].sum()
            rows.append(out)
    car = pd.DataFrame(rows)
    car.to_parquet(CAR_OUT, index=False)
    return car


def _ols(d: pd.DataFrame, y: str, regs: list[str]) -> dict | None:
    need = [y, *regs, *CONTROLS, "industry", "fiscal_year", "ticker"]
    d = d.dropna(subset=need)
    if len(d) < 100:
        return None
    f = f"{y} ~ {' + '.join(regs + CONTROLS)} + C(industry) + C(fiscal_year)"
    res = smf.ols(f, data=d).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(d["ticker"])[0]})
    out = {"N": int(res.nobs)}
    for i, r in enumerate(regs):
        sfx = "" if i == 0 else f"_{i + 1}"
        out.update({f"regressor{sfx}": r, f"beta{sfx}": res.params[r], f"se{sfx}": res.bse[r],
                    f"p_value{sfx}": res.pvalues[r]})
    return out


def run() -> None:
    p2 = prep.load_clean_p2()
    events = p2.dropna(subset=["accession", "filing_date"])[["ticker", "accession", "filing_date"]]
    car = compute_car(events)
    d = p2.merge(car, on=["ticker", "accession"], how="inner")
    # CAR in punti percentuali, winsorizzati 1/99 come le altre variabili continue.
    for c in EVENT_WINDOWS:
        d[c] = d[c].clip(d[c].quantile(config.WINSOR_LOWER), d[c].quantile(config.WINSOR_UPPER)) * 100
    print(f"[CAR] Eventi con CAR: {len(d)}; CAR[-1,+1] medio {d['car_m1_p1'].mean():.3f}%")

    post = d["fiscal_year"] >= 2022  # 10-K FY2022 depositati dopo il rilascio di ChatGPT
    variants = [
        ("all", d, ["ai_score_log"]),
        ("all_fixed_dictionary", d, ["ai_score_fixed_log"]),
        ("all_ops_vs_risk", d, ["ai_ops_log", "ai_risk_log"]),
        ("pre_chatgpt_FY2015_2021", d[~post], ["ai_score_log"]),
        ("post_chatgpt_FY2022_2024", d[post], ["ai_score_log"]),
    ]
    rows = []
    for y in EVENT_WINDOWS:
        for name, sub, regs in variants:
            r = _ols(sub, y, regs)
            if r:
                rows.append({"outcome": y, "sample": name, **r})
    out = pd.DataFrame(rows)
    num = out.select_dtypes("number").columns
    out[num] = out[num].round(5)
    plots.save_table(out, "p2_car_regressions", index=False)
    print(out.to_string(index=False))


def main() -> None:
    run()


if __name__ == "__main__":
    main()
