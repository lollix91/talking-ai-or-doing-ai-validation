"""Layer 0 della Fase 4: pulizia e preparazione del panel di analisi.

Prende i Parquet grezzi della Fase 2 (`panel_p2`) e della Fase 3
(`ai_adoption_scores`) e produce il panel pulito, pronto per le
regressioni:

  - filtra alla finestra dello studio (config.STUDY_YEAR_MIN..STUDY_YEAR_MAX);
  - rimuove valori patologici (inf, -inf) generati da divisioni per ~0;
  - winsorizza le variabili continue ai percentili config.WINSOR_*;
  - aggiunge log_revenue (proxy di dimensione) e la variabile di adozione IA
    (continua + estensiva binaria + categoriale a 4 livelli).

Output:
  - panel_p2_clean.parquet
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import config

P2_IN = config.PROCESSED_DIR / "panel_p2.parquet"
AI_IN = config.PROCESSED_DIR / "ai_adoption_scores.parquet"
P2_OUT = config.PROCESSED_DIR / "panel_p2_clean.parquet"

# Variabili continue da winsorizzare.
P2_CONTINUOUS = [
    "operating_margin",
    "rd_intensity",
    "capex_intensity",
    "pe_ratio",
    "pe_ratio_historical",
    "market_cap",
    "market_cap_historical",
    "tobin_q",
    "roa",
    "leverage",
    "revenue_growth",
]


def _filter_window(df: pd.DataFrame) -> pd.DataFrame:
    """Tiene solo gli anni fiscali dentro la finestra dello studio."""
    return df[
        (df["fiscal_year"] >= config.STUDY_YEAR_MIN)
        & (df["fiscal_year"] <= config.STUDY_YEAR_MAX)
    ].copy()


def _clean_infinities(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Sostituisce +/-inf con NaN nelle colonne indicate."""
    for c in cols:
        if c in df.columns:
            df[c] = df[c].replace([np.inf, -np.inf], np.nan)
    return df


def _winsorize(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Winsorizza (clip) le colonne ai percentili configurati.

    Riduce l'effetto degli outlier estremi (es. rapporti con denominatore
    quasi nullo) senza eliminare osservazioni. Standard nella finanza empirica.
    """
    lo, hi = config.WINSOR_LOWER, config.WINSOR_UPPER
    for c in cols:
        if c not in df.columns:
            continue
        s = df[c]
        if s.notna().sum() == 0:
            continue
        ql, qh = s.quantile(lo), s.quantile(hi)
        df[c] = s.clip(lower=ql, upper=qh)
    return df


def select_filings(ai: pd.DataFrame) -> pd.DataFrame:
    """Un solo 10-K per impresa-anno fiscale.

    Le rettifiche 10-K/A sono quasi sempre parziali (es. solo Parte III) e
    vengono escluse; tra piu' 10-K originali per lo stesso anno (cambio di
    fine esercizio) si tiene il deposito piu' recente.
    """
    ai = ai[ai["form"] == "10-K"].dropna(subset=["fiscal_year"])
    ai = ai.sort_values("filing_date").drop_duplicates(["ticker", "fiscal_year"], keep="last")
    return ai.assign(fiscal_year=ai["fiscal_year"].astype(int))


def _filter_index_members(df: pd.DataFrame) -> pd.DataFrame:
    """Esclude i costituenti spuri (permanenza < config.MIN_INDEX_DAYS)."""
    return df[df["index_days"].fillna(0) >= config.MIN_INDEX_DAYS].copy()


def _dedupe_share_classes(df: pd.DataFrame) -> pd.DataFrame:
    """Una sola riga per impresa (CIK) e anno: le doppie classi di azioni
    (GOOG/GOOGL, FOX/FOXA, NWS/NWSA, UA/UAA) e i ticker riassegnati allo
    stesso emittente (DISCA/WBD, DTV/DTE) duplicherebbero l'impresa.
    Si tiene il ticker con piu' anni nel panel (poi ordine alfabetico)."""
    n_years = df.groupby("ticker")["fiscal_year"].transform("size")
    key = df["cik"].fillna(df["ticker"])
    df = df.assign(_n=n_years, _key=key).sort_values(["_key", "fiscal_year", "_n", "ticker"],
                                                     ascending=[True, True, False, True])
    return df.drop_duplicates(["_key", "fiscal_year"]).drop(columns=["_n", "_key"])


def _add_ai_treatment(p2: pd.DataFrame) -> pd.DataFrame:
    """Aggiunge le variabili di adozione IA al panel P2, unendo per
    ticker + anno fiscale con gli score NLP.

    Crea tre versioni della variabile di trattamento:
      - ai_score:  misura continua (score per 10.000 parole);
      - ai_any:    margine estensivo (1 se score > 0, altrimenti 0);
      - ai_group:  'non_adopter' se score == 0, altrimenti terzili dei soli
                   score positivi entro anno -> 'low'/'medium'/'high'.
    """
    if not AI_IN.exists():
        print(f"[PREP] Attenzione: {AI_IN.name} assente, salto la variabile IA.")
        return p2

    ai = select_filings(pd.read_parquet(AI_IN))
    keep = [
        c
        for c in ai.columns
        if c in ("ticker", "fiscal_year", "ai_keyword_count", "accession",
                 "filing_date", "report_date", "total_words")
        or c.startswith(("ai_score", "n_", "words_"))
    ]
    ai = ai[keep].rename(columns={"ai_score_per_10k": "ai_score"})

    p2 = p2.merge(ai, on=["ticker", "fiscal_year"], how="left")

    # Variante log per robustezza (lo score e' fortemente asimmetrico).
    p2["ai_score_log"] = np.log1p(p2["ai_score"])
    p2["ai_score_fixed_log"] = np.log1p(p2["ai_score_fixed"])
    # Disclosure operativa (Item 1 + Item 7) vs boilerplate (Item 1A): media
    # degli score di sezione disponibili, in log1p come lo score principale.
    p2["ai_ops_log"] = np.log1p(p2[["ai_score_business", "ai_score_mdna"]].mean(axis=1))
    p2["ai_risk_log"] = np.log1p(p2["ai_score_risk_factors"])

    # Margine estensivo: menziona l'IA almeno una volta.
    p2["ai_any"] = np.where(p2["ai_score"].notna(), (p2["ai_score"] > 0).astype("Int64"), pd.NA)

    # Categoriale: non-adopter + terzili dei positivi entro anno.
    # Soglie globali (su tutti i positivi) usate come fallback deterministico
    # per anni con troppo pochi positivi per stimare terzili propri.
    p2["ai_group"] = pd.NA
    has_score = p2["ai_score"].notna()
    p2.loc[has_score & (p2["ai_score"] <= 0), "ai_group"] = "non_adopter"

    positives = p2[has_score & (p2["ai_score"] > 0)]
    labels = ["low", "medium", "high"]
    g_t33, g_t67 = positives["ai_score"].quantile([1 / 3, 2 / 3])

    def _global_class(v: float) -> str:
        return "high" if v >= g_t67 else ("low" if v <= g_t33 else "medium")

    for year, grp in positives.groupby("fiscal_year"):
        if len(grp) < 3:
            p2.loc[grp.index, "ai_group"] = grp["ai_score"].map(_global_class)
            continue
        try:
            cats = pd.qcut(grp["ai_score"], q=3, labels=labels, duplicates="drop")
        except ValueError:
            cats = grp["ai_score"].map(_global_class)
        p2.loc[grp.index, "ai_group"] = cats.astype(object)

    return p2


def build_clean_p2() -> pd.DataFrame:
    p2 = pd.read_parquet(P2_IN)
    p2 = _filter_index_members(p2)
    p2 = _filter_window(p2)
    p2 = _dedupe_share_classes(p2)
    p2 = _clean_infinities(p2, P2_CONTINUOUS)
    p2 = _winsorize(p2, P2_CONTINUOUS)
    p2["log_revenue"] = np.log(p2["revenue"].where(p2["revenue"] > 0))
    p2 = _add_ai_treatment(p2)
    p2.to_parquet(P2_OUT, index=False)
    n_group = p2["ai_group"].notna().sum() if "ai_group" in p2.columns else 0
    print(
        f"[PREP] P2 pulito: {p2.shape} "
        f"({p2['ticker'].nunique()} ticker, {p2['fiscal_year'].min()}-"
        f"{p2['fiscal_year'].max()}); righe con gruppo IA: {n_group} -> {P2_OUT.name}"
    )
    if "ai_group" in p2.columns and n_group:
        print(f"[PREP] Distribuzione ai_group: {p2['ai_group'].value_counts().to_dict()}")
    return p2


def load_clean_p2() -> pd.DataFrame:
    return pd.read_parquet(P2_OUT)


def main() -> None:
    build_clean_p2()


if __name__ == "__main__":
    main()
