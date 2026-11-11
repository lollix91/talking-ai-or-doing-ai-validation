"""Fase B — validazione dello score IA contro le etichette manuali.

- Score v1 (dizionario originario, usato per costruire il campione di
  validazione): ricalcolato qui per conoscere la composizione per strato
  della popolazione e ripesare le metriche (il campione sovra-rappresenta
  i documenti con score positivo: 3 strati positivi su 4).
- Metriche binarie (menzione IA si'/no) score vs etichette: accuratezza,
  precision, recall, F1, kappa di Cohen; non pesate e pesate per strato
  (inverse probability weights).
- Spearman tra score continuo e sostanza ordinale (0 = nessuna menzione,
  1 = boilerplate, 2 = operativa, 3 = core).
- Accordo tra annotatori (umano vs seconda annotazione LLM, dichiarata).

Output: outputs/tables/validation_metrics.csv, validation_interannotator.csv
"""
from __future__ import annotations

import json
import re
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

import config
from src import ai_keywords as kw
from src.analysis import plots
from src.nlp import DOCS_DIR, html_to_text

P = config.PROCESSED_DIR
V1_OUT = P / "ai_scores_v1.parquet"
SAMPLE_CSV = P / "annotation_sample.csv"
HUMAN_JSON = P / "annotations.json"
LLM_JSON = P / "annotations_llm.json"
HUMAN2_JSON = P / "annotations_human2.json"
ADJ_JSON = P / "annotations_adjudicated.json"
SUBSTANCE = {"boilerplate_risk_only": 1, "operational_mention": 2, "core_to_business": 3}

# --- Score v1: conteggio per termine (sovrapposto), senza acronimi ---------
_V1_PATTERNS = [re.compile(r"\b" + re.escape(t) + r"\b") for t in kw.ALL_KEYWORDS]
_V1_AMBIG = [re.compile(r"\b" + re.escape(t) + r"\b") for t in kw.AMBIGUOUS]
_V1_CTX = re.compile(r"\b(?:" + "|".join(re.escape(w) for w in kw.CONTEXT_WORDS) + r")\b")


def _v1_score(r: dict) -> dict | None:
    fdate = pd.to_datetime(r["filing_date"]).strftime("%Y-%m-%d")
    ext = Path(str(r["primary_document"])).suffix or ".htm"
    path = DOCS_DIR / r["ticker"] / f"{fdate}_{r['accession']}{ext}"
    if not path.exists():
        return None
    text = html_to_text(path.read_text(encoding="utf-8", errors="replace")).lower()
    words = len(text.split())
    n = sum(len(p.findall(text)) for p in _V1_PATTERNS)
    for p in _V1_AMBIG:
        n += sum(1 for m in p.finditer(text)
                 if _V1_CTX.search(text[max(0, m.start() - 60):m.end() + 60]))
    return {"accession": r["accession"], "ai_score_v1": n / words * 10_000 if words else 0.0}


def build_v1_scores() -> pd.DataFrame:
    idx = pd.read_parquet(P / "edgar_10k_index.parquet")
    with ProcessPoolExecutor(max_workers=12) as ex:
        rows = [r for r in ex.map(_v1_score, idx.to_dict("records"), chunksize=8) if r]
    df = pd.DataFrame(rows)
    df.to_parquet(V1_OUT, index=False)
    return df


def _stratum_weights(sample: pd.DataFrame) -> pd.Series:
    """IPW: quota dello strato nella popolazione / quota nel campione."""
    v1 = pd.read_parquet(V1_OUT) if V1_OUT.exists() else build_v1_scores()
    n_zero = int((v1["ai_score_v1"] <= 0).sum())
    n_pos = len(v1) - n_zero
    pop = {"zero": n_zero, "pos_low": n_pos / 3, "pos_mid": n_pos / 3, "pos_high": n_pos / 3}
    pop = pd.Series(pop) / sum(pop.values())
    samp = sample["stratum"].value_counts(normalize=True)
    return sample["stratum"].map(pop / samp)


def _wkappa(a: np.ndarray, b: np.ndarray, w: np.ndarray) -> float:
    """Kappa di Cohen (binaria) con pesi per osservazione."""
    w = w / w.sum()
    po = np.sum(w * (a == b))
    pa, pb = np.sum(w * a), np.sum(w * b)
    pe = pa * pb + (1 - pa) * (1 - pb)
    return (po - pe) / (1 - pe) if pe < 1 else np.nan


def _binary_metrics(pred: np.ndarray, truth: np.ndarray, w: np.ndarray) -> dict:
    w = w / w.sum()
    tp = np.sum(w * ((pred == 1) & (truth == 1)))
    fp = np.sum(w * ((pred == 1) & (truth == 0)))
    fn = np.sum(w * ((pred == 0) & (truth == 1)))
    prec = tp / (tp + fp) if tp + fp else np.nan
    rec = tp / (tp + fn) if tp + fn else np.nan
    return {
        "accuracy": np.sum(w * (pred == truth)),
        "precision": prec,
        "recall": rec,
        "f1": 2 * prec * rec / (prec + rec) if prec and rec else np.nan,
        "kappa": _wkappa(pred, truth, w),
    }


def _load_labels(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["accession", "any", "substance"])
    ann = json.loads(path.read_text(encoding="utf-8"))
    return pd.DataFrame([
        {"accession": k,
         "any": int(v["ai_mention"] == "present"),
         "substance": SUBSTANCE.get(v.get("ai_substance"), 0) if v["ai_mention"] == "present" else 0}
        for k, v in ann.items()
    ])


def joined_sample() -> pd.DataFrame:
    sample = pd.read_csv(SAMPLE_CSV)
    sample["weight"] = _stratum_weights(sample)
    cur = pd.read_parquet(P / "ai_adoption_scores.parquet")[
        ["accession", "ai_score_per_10k", "ai_score_fixed"]
    ].rename(columns={"ai_score_per_10k": "score_v2", "ai_score_fixed": "score_v2_fixed"})
    # Lo stesso 10-K compare sotto piu' ticker per le doppie classi di azioni.
    cur = cur.drop_duplicates("accession")
    v1 = pd.read_parquet(V1_OUT).drop_duplicates("accession")
    df = sample[["ticker", "fiscal_year", "accession", "stratum", "weight"]]
    df = df.merge(v1, on="accession", how="left").merge(cur, on="accession", how="left")
    df = df.rename(columns={"ai_score_v1": "score_v1"})
    hum = _load_labels(HUMAN_JSON).add_prefix("human_").rename(columns={"human_accession": "accession"})
    llm = _load_labels(LLM_JSON).add_prefix("llm_").rename(columns={"llm_accession": "accession"})
    h2 = _load_labels(HUMAN2_JSON).add_prefix("human2_").rename(columns={"human2_accession": "accession"})
    adj = _load_labels(ADJ_JSON).add_prefix("adj_").rename(columns={"adj_accession": "accession"})
    df = (df.merge(hum, on="accession", how="left").merge(llm, on="accession", how="left")
            .merge(h2, on="accession", how="left").merge(adj, on="accession", how="left"))
    return _add_gold(df)


def _add_gold(df: pd.DataFrame) -> pd.DataFrame:
    """Gold standard sul sottocampione di 80: etichetta aggiudicata se
    presente; altrimenti l'etichetta comune dei tre annotatori quando
    concordano (binario / sostanza); altrimenti mancante. Le etichette
    originali non vengono modificate."""
    trio = ["human", "human2", "llm"]
    full = df[[f"{r}_any" for r in trio]].notna().all(axis=1)
    agree_any = full & (df[[f"{r}_any" for r in trio]].nunique(axis=1) == 1)
    agree_sub = full & (df[[f"{r}_substance" for r in trio]].nunique(axis=1) == 1)
    df["gold_any"] = np.where(df["adj_any"].notna(), df["adj_any"],
                              np.where(agree_any, df["human_any"], np.nan))
    df["gold_substance"] = np.where(df["adj_substance"].notna(), df["adj_substance"],
                                    np.where(agree_sub, df["human_substance"], np.nan))
    return df


def _metrics_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for rater in ("human", "human2", "llm", "gold"):
        d0 = df.dropna(subset=[f"{rater}_any"])
        if d0.empty:
            continue
        for score in ("score_v1", "score_v2", "score_v2_fixed"):
            d = d0.dropna(subset=[score])
            pred = (d[score] > 0).astype(int).to_numpy()
            truth = d[f"{rater}_any"].astype(int).to_numpy()
            ds = d.dropna(subset=[f"{rater}_substance"])
            rho, p = stats.spearmanr(ds[score], ds[f"{rater}_substance"]) if len(ds) > 10 else (np.nan, np.nan)
            for wname, w in (("unweighted", np.ones(len(d))), ("ipw_stratum", d["weight"].to_numpy())):
                rows.append({"labels": rater, "score": score, "weighting": wname, "n": len(d),
                             **_binary_metrics(pred, truth, w),
                             "spearman_substance": rho, "spearman_p": p})
    return pd.DataFrame(rows).round(3)


def _interannotator(df: pd.DataFrame) -> pd.DataFrame:
    """Accordo a coppie tra annotatori (umano, umano2, LLM) sui documenti
    annotati da entrambi."""
    from itertools import combinations
    from sklearn.metrics import cohen_kappa_score

    raters = [r for r in ("human", "human2", "llm") if f"{r}_any" in df and df[f"{r}_any"].notna().any()]
    rows = []
    for r1, r2 in combinations(raters, 2):
        d = df.dropna(subset=[f"{r1}_any", f"{r2}_any"])
        if len(d) < 10:
            continue
        a, b = d[f"{r1}_any"].astype(int).to_numpy(), d[f"{r2}_any"].astype(int).to_numpy()
        s1, s2 = d[f"{r1}_substance"].astype(int), d[f"{r2}_substance"].astype(int)
        rho, _ = stats.spearmanr(s1, s2)
        kq = cohen_kappa_score(s1, s2, weights="quadratic")
        for wname, w in (("unweighted", np.ones(len(d))), ("ipw_stratum", d["weight"].to_numpy())):
            rows.append({
                "pair": f"{r1} vs {r2}", "weighting": wname, "n": len(d),
                "agreement_any": np.sum(w * (a == b)) / w.sum(),
                "kappa_any": _wkappa(a, b, w),
                "agreement_substance": np.sum(w * (s1.to_numpy() == s2.to_numpy())) / w.sum(),
                "spearman_substance": rho, "kappa_substance_quadratic": kq,
            })
    return pd.DataFrame(rows).round(3)


def run() -> None:
    if not V1_OUT.exists():
        build_v1_scores()
    df = joined_sample()
    df.to_csv(P / "annotations_joined.csv", index=False)
    met = _metrics_table(df)
    plots.save_table(met, "validation_metrics", index=False)
    print(met.to_string(index=False))
    ia = _interannotator(df)
    if not ia.empty:
        plots.save_table(ia, "validation_interannotator", index=False)
        print(ia.to_string(index=False))


def main() -> None:
    run()


if __name__ == "__main__":
    main()
