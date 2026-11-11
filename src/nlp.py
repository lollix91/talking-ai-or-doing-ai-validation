"""NLP: misura l'adozione di IA nei 10-K e classifica le aziende.

Pipeline:
  1. Parsing dei documenti HTML 10-K (estrarre testo pulito).
  2. Conteggio delle keyword IA (dizionario da letteratura, vedi ai_keywords.py)
     con word boundaries, gestione degli acronimi e validazione del contesto
     per i termini ambigui.
  3. Normalizzazione per 10.000 parole (approccio AIDI).
  4. Classificazione in alta/media/bassa adozione (terzili, approccio Q1).

Output:
  - ai_adoption_scores.parquet  (ticker x anno -> score + classificazione)
"""
from __future__ import annotations

import os
import re
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

import config
from src import ai_keywords as kw

OUT = config.PROCESSED_DIR / "ai_adoption_scores.parquet"
DOCS_DIR = config.RAW_DIR / "edgar" / "10k"

# Pattern HTML per estrarre testo
_TAG_RE = re.compile(r"<[^>]+>")
_ENTITY_RE = re.compile(r"&[a-zA-Z]+;|&#\d+;")
_WS_RE = re.compile(r"\s+")


def html_to_text(html: str) -> str:
    """Estrae testo leggibile da un documento HTML/SEC."""
    # Rimuove blocchi non testuali (script, style).
    html = re.sub(r"(?is)<(script|style).*?</\1>", " ", html)
    # Rimuove i tag.
    text = _TAG_RE.sub(" ", html)
    # Decodifica entita' comuni.
    text = text.replace("&nbsp;", " ").replace("&amp;", "&")
    text = text.replace("&lt;", "<").replace("&gt;", ">")
    text = _ENTITY_RE.sub(" ", text)
    text = _WS_RE.sub(" ", text).strip()
    return text


def count_words(text: str) -> int:
    return len(text.split())


def _build_alternation(terms: list[str]) -> re.Pattern:
    """Un'unica alternation con confini di parola, termini piu' lunghi per
    primi: i match sono non sovrapposti, quindi "convolutional neural
    network" conta una volta sola (non anche come "neural network") e
    "gpt-4" non conta anche come "gpt"."""
    alts = sorted({t.lower() for t in terms}, key=len, reverse=True)
    return re.compile(r"\b(" + "|".join(re.escape(t) for t in alts) + r")\b", re.IGNORECASE)


# Precompila i pattern una sola volta.
_KW_RE = _build_alternation(kw.ALL_KEYWORDS)
_AMBIG_RE = _build_alternation(kw.AMBIGUOUS)
_ACRONYM_RE = re.compile(r"\b(" + "|".join(kw.ACRONYMS) + r")\b")
# Word boundaries anche sulle parole di contesto: senza, "ai" matcherebbe
# dentro "said", "email", "again" ecc. rendendo il filtro inutile.
_CTX_RE = re.compile(
    r"\b(?:" + "|".join(re.escape(w) for w in kw.CONTEXT_WORDS) + r")\b",
    re.IGNORECASE,
)

# --- Parsing per sezione del 10-K -------------------------------------------
# Le sezioni hanno segnale diverso: menzioni IA in Item 1 (Business) e
# Item 7 (MD&A) descrivono attivita' operative; in Item 1A (Risk Factors)
# sono spesso boilerplate legale. Gli heading "Item N. <titolo>" compaiono
# piu' volte (indice, corpo, riferimenti incrociati): per ogni sezione si
# sceglie l'occorrenza che apre il segmento piu' lungo prima del successivo
# heading atteso (es. Item 1 -> Item 1A/1B/2). Le occorrenze senza un
# heading successivo dopo di se' (tipicamente riferimenti incrociati nel
# corpo o nelle note) sono scartate.
_ITEM_TITLES = {
    "1": "business",
    "1a": "risk factors",
    "1b": "unresolved",
    "1c": "cybersecurity",
    "2": "properties",
    "3": "legal",
    "7": "management",
    "7a": "quantitative",
    "8": "financial",
}
def _spaced(word: str) -> str:
    """Regex tollerante agli spazi spuri tra lettere (maiuscoletto HTML:
    "B USINESS", "R ISK F ACTORS")."""
    return r"\s?".join(re.escape(c) for c in word.replace(" ", ""))


_ITEM_PATTERNS = {
    label: re.compile(
        r"\bitem\s+" + label + r"\.?[^a-zA-Z]{0,40}" + _spaced(title),
        re.IGNORECASE,
    )
    for label, title in _ITEM_TITLES.items()
}
# Heading che chiudono ciascuna sezione di interesse.
_SECTION_END = {
    "1": ("1a", "1b", "1c", "2"),
    "1a": ("1b", "1c", "2"),
    "7": ("7a", "8"),
}
# Lunghezza minima (caratteri) per considerare valida una sezione trovata.
_MIN_SECTION_CHARS = 500


def find_item_positions(text: str) -> dict[str, list[int]]:
    """Tutte le posizioni di ciascun heading "Item N. <titolo>"."""
    return {
        label: [m.start() for m in pat.finditer(text)]
        for label, pat in _ITEM_PATTERNS.items()
    }


def split_sections(text: str) -> dict[str, str | None]:
    """Suddivide il testo del 10-K nelle sezioni principali.

    Ritorna {'business': ..., 'risk_factors': ..., 'mdna': ...} con None
    quando la sezione non e' identificabile in modo affidabile.
    """
    pos = find_item_positions(text)

    def _seg(label: str) -> str | None:
        ends = sorted(p for e in _SECTION_END[label] for p in pos.get(e, []))
        best = None
        for start in pos.get(label, []):
            end = next((e for e in ends if e > start), None)
            if end is not None and (best is None or end - start > best[1] - best[0]):
                best = (start, end)
        if best is None or best[1] - best[0] < _MIN_SECTION_CHARS:
            return None
        return text[best[0]:best[1]]

    return {
        "business": _seg("1"),
        "risk_factors": _seg("1a"),
        "mdna": _seg("7"),
    }


def count_keywords(text: str) -> dict[str, int]:
    """Conta le occorrenze delle keyword IA nel testo.

    I termini ambigui vengono contati solo se compaiono entro una finestra
    di contesto (±60 caratteri) vicino a una parola di contesto AI.
    """
    text_lower = text.lower()
    counts: dict[str, int] = {}

    # Tier 1-3: conteggio diretto (match non sovrapposti).
    for m in _KW_RE.finditer(text_lower):
        counts[m.group(1)] = counts.get(m.group(1), 0) + 1

    # Termini ambigui: validazione del contesto.
    for m in _AMBIG_RE.finditer(text_lower):
        start = max(0, m.start() - 60)
        end = min(len(text_lower), m.end() + 60)
        if _CTX_RE.search(text_lower[start:end]):
            counts[m.group(1)] = counts.get(m.group(1), 0) + 1

    # Acronimi case-sensitive, esclusa la definizione "artificial intelligence (AI)"
    # e l'omonimo "avian influenza (AI)".
    for m in _ACRONYM_RE.finditer(text):
        before = text_lower[max(0, m.start() - 16):m.start()].rstrip()
        if not before.endswith(("intelligence (", "influenza (")):
            counts[m.group(1)] = counts.get(m.group(1), 0) + 1

    return counts


_FIXED = set(kw.FIXED_KEYWORDS) | set(kw.AMBIGUOUS)


def ai_score(text: str) -> tuple[float, int, int]:
    """Ritorna (score_per_10k_words, total_ai_count, total_words)."""
    total_words = count_words(text)
    if total_words == 0:
        return 0.0, 0, 0
    counts = count_keywords(text)
    total_ai = sum(counts.values())
    score = (total_ai / total_words) * 10_000
    return score, total_ai, total_words


def score_document(text: str) -> dict:
    """Score totale, score a dizionario fisso e conteggi per tier."""
    total_words = count_words(text)
    counts = count_keywords(text)
    total_ai = sum(counts.values())
    fixed = sum(n for t, n in counts.items() if t in _FIXED)
    per10k = (lambda n: n / total_words * 10_000 if total_words else 0.0)
    out = {
        "ai_keyword_count": total_ai,
        "total_words": total_words,
        "ai_score_per_10k": per10k(total_ai),
        "ai_score_fixed": per10k(fixed),
    }
    for tier in ("tier1", "acronym", "tier2", "tier3", "ambiguous"):
        out[f"n_{tier}"] = sum(n for t, n in counts.items() if kw.TIER_OF.get(t) == tier)
    return out


def classify_adoption(scores: pd.Series, method: str = "tercile") -> pd.Series:
    """Classifica le aziende in alta/media/bassa adozione IA.

    method:
      - 'tercile': terzili (33/33/33) -- raccomandato per Q1
      - 'median': split binario alla mediana
      - 'quartile': quartili (top/bottom 25%)
    """
    s = scores.copy()
    if method == "median":
        med = s.median()
        return s.apply(lambda x: "high" if x >= med else "low")
    if method == "quartile":
        q25, q75 = s.quantile(0.25), s.quantile(0.75)
        return s.apply(
            lambda x: "high" if x >= q75 else ("low" if x <= q25 else "medium")
        )
    # default: tercile
    t33, t67 = s.quantile(1 / 3), s.quantile(2 / 3)
    return s.apply(
        lambda x: "high" if x >= t67 else ("low" if x <= t33 else "medium")
    )


def _fiscal_year_of_filing(row: pd.Series) -> int | None:
    """Anno fiscale = anno del periodo coperto (report_date), NON della data
    di deposito: un 10-K con FYE dicembre depositato a febbraio 2024 copre
    il FY2023. Fallback sull'anno di filing se report_date manca.
    Anni fiscali a 52/53 settimane che chiudono nei primi giorni di gennaio
    (es. 2016-01-02) appartengono all'anno precedente: si arretra di 7 giorni,
    coerentemente con la data di bilancio normalizzata di EODHD (2015-12-31)."""
    rd = pd.to_datetime(row.get("report_date"), errors="coerce")
    if pd.notna(rd):
        return int((rd - pd.Timedelta(days=7)).year)
    fd = pd.to_datetime(row.get("filing_date"), errors="coerce")
    return int(fd.year) if pd.notna(fd) else None


def _process_filing(r: dict) -> dict | None:
    """Score di un singolo 10-K (eseguito nei worker del process pool)."""
    ticker = r["ticker"]
    fdate = pd.to_datetime(r["filing_date"]).strftime("%Y-%m-%d")
    ext = Path(str(r["primary_document"])).suffix or ".htm"
    doc_path = DOCS_DIR / ticker / f"{fdate}_{r['accession']}{ext}"
    if not doc_path.exists():
        return None
    try:
        html = doc_path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return None

    text = html_to_text(html)
    doc = score_document(text)

    # Score per sezione (Item 1 / 1A / 7): lo stesso conteggio applicato
    # a sottoinsiemi del documento per separare segnale operativo da
    # boilerplate legale.
    sec = {}
    for sec_name, sec_text in split_sections(text).items():
        sec[f"ai_score_{sec_name}"] = np.nan if sec_text is None else ai_score(sec_text)[0]
        sec[f"words_{sec_name}"] = 0 if sec_text is None else count_words(sec_text)

    return {
        "ticker": ticker,
        "form": r.get("form"),
        "filing_date": pd.to_datetime(r["filing_date"]),
        "report_date": pd.to_datetime(r.get("report_date"), errors="coerce"),
        "fiscal_year": _fiscal_year_of_filing(r),
        "accession": r["accession"],
        **doc,
        **sec,
    }


def run(index: pd.DataFrame | None = None) -> pd.DataFrame:
    """Calcola gli score di adozione IA per tutti i 10-K scaricati.

    Args:
        index: DataFrame dell'indice 10-K (edgar_10k_index.parquet).
               Se None, lo legge da disco.
    """
    if index is None:
        idx_path = config.PROCESSED_DIR / "edgar_10k_index.parquet"
        if not idx_path.exists():
            raise RuntimeError(
                f"Indice 10-K non trovato ({idx_path}). "
                "Esegui prima: python run.py --edgar --no-docs"
            )
        index = pd.read_parquet(idx_path)

    records = index.to_dict("records")
    workers = max(1, (os.cpu_count() or 2) - 2)
    with ProcessPoolExecutor(max_workers=workers) as ex:
        results = list(tqdm(ex.map(_process_filing, records, chunksize=8),
                            total=len(records), desc="NLP AI adoption"))
    rows = [r for r in results if r is not None]

    df = pd.DataFrame(rows)
    if df.empty:
        print("[NLP] Nessun documento processato.")
        return df

    # Classificazione per anno (terzili entro ogni anno fiscale).
    df["adoption_class"] = df.groupby("fiscal_year")["ai_score_per_10k"].transform(
        lambda s: classify_adoption(s, method="tercile")
    )
    # Se in un anno c'e' un solo ticker, la classificazione e' degenere:
    # fallback su classificazione globale.
    single_year = df.groupby("fiscal_year")["ticker"].transform("count") == 1
    if single_year.any():
        global_cls = classify_adoption(df["ai_score_per_10k"], method="tercile")
        df.loc[single_year, "adoption_class"] = global_cls[single_year]

    df = df.sort_values(["ticker", "filing_date"]).reset_index(drop=True)
    df.to_parquet(OUT, index=False)
    print(
        f"[NLP] {len(df)} documenti analizzati, "
        f"{df['ticker'].nunique()} aziende -> {OUT}"
    )
    print(f"[NLP] Distribuzione classi: {df['adoption_class'].value_counts().to_dict()}")
    return df


def main() -> None:
    run()


if __name__ == "__main__":
    main()
