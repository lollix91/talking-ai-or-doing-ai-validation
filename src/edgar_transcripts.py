"""Scarica le trascrizioni delle earnings call da SEC EDGAR via 8-K exhibit.

Molte aziende S&P 500 depositano il transcript della earnings call come
allegato (EX-99.1 / EX-99.2) di un filing 8-K (item 2.02 "Results of
Operations and Financial Condition"). Questo e':
  - gratuito (API pubblica EDGAR)
  - legale (documenti SEC pubblici)
  - riproducibile

Limite dichiarato: copertura parziale (non tutte le aziende depositano i
transcript come exhibit). Da dichiarare esplicitamente nel paper.

Flusso:
  1. risolve il CIK (riusa la logica di edgar_10k);
  2. interroga submissions e filtra i form 8-K con item 2.02;
  3. per ogni 8-K, scarica l'index file del filing per trovare gli exhibit
     EX-99.* che contengono il transcript;
  4. salva un indice Parquet + i documenti su disco.
"""
from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from tqdm import tqdm

import config
from src.edgar_10k import resolve_ciks
from src.edgar_client import SECSession

INDEX_OUT = config.PROCESSED_DIR / "edgar_transcripts_index.parquet"
DOCS_DIR = config.RAW_DIR / "edgar" / "transcripts"

# Item 2.02 = "Results of Operations and Financial Condition" (earnings).
EARNINGS_ITEM = "2.02"
# Pattern per identificare exhibit che sono transcript (non solo press release).
TRANSCRIPT_KEYWORDS = [
    "transcript",
    "earnings call",
    "conference call",
    "prepared remarks",
    "q&a",
    "q and a",
    "operator:",
    "question and answer",
]


def _filing_index_url(cik10: str, accession: str) -> str:
    acc_nodash = accession.replace("-", "")
    return (
        f"https://www.sec.gov/Archives/edgar/data/{int(cik10)}/{acc_nodash}/"
        f"{accession}-index.html"
    )


def _filing_dir_url(cik10: str, accession: str) -> str:
    acc_nodash = accession.replace("-", "")
    return f"https://www.sec.gov/Archives/edgar/data/{int(cik10)}/{acc_nodash}/"


def _iter_filing_rows(recent: dict):
    n = len(recent.get("accessionNumber", []))
    items_list = recent.get("items", [])
    primary_list = recent.get("primaryDocument", [])
    for i in range(n):
        items = items_list[i] if i < len(items_list) else ""
        primary = primary_list[i] if i < len(primary_list) else ""
        yield {
            "accession": recent["accessionNumber"][i],
            "form": recent["form"][i],
            "filing_date": recent["filingDate"][i],
            "items": items,
            "primary_document": primary,
        }


def fetch_8k_filings(sec: SECSession, cik10: str) -> list[dict]:
    """Recupera tutti i filing 8-K con item 2.02 per un CIK."""
    url = config.EDGAR_SUBMISSIONS_URL.format(cik10=cik10)
    resp = sec.get(url)
    if resp.status_code == 404:
        return []
    data = resp.json()
    filings = data.get("filings", {})
    rows = list(_iter_filing_rows(filings.get("recent", {})))

    # File paginati (filing piu' vecchi).
    for extra in filings.get("files", []) or []:
        if extra.get("filingTo", "") < config.START_DATE:
            continue
        eurl = config.EDGAR_SUBMISSIONS_FILE_URL.format(name=extra.get("name"))
        er = sec.get(eurl)
        if er.status_code == 200:
            rows += list(_iter_filing_rows(er.json()))

    # Filtra: form 8-K, item 2.02 presente, nella finestra.
    result = []
    for r in rows:
        if r["form"] != "8-K":
            continue
        if EARNINGS_ITEM not in (r.get("items") or ""):
            continue
        if not (config.START_DATE <= r["filing_date"] <= config.END_DATE):
            continue
        result.append(r)
    return result


def find_transcript_exhibit(
    sec: SECSession, cik10: str, accession: str
) -> dict | None:
    """Scarica l'index.html strutturato del filing e cerca l'exhibit EX-99.*
    che sembra un transcript (in base al tipo documento e descrizione)."""
    index_url = _filing_index_url(cik10, accession)
    resp = sec.get(index_url)
    if resp.status_code != 200:
        return None

    html = resp.text
    # L'index.html ha una tabella con colonne: Seq, Document Type, Filename,
    # Description, Size. Estraiamo le righe con i link ai documenti.
    table = re.search(r"<table.*?</table>", html, re.DOTALL)
    if not table:
        return None

    candidates = []
    dir_url = _filing_dir_url(cik10, accession)
    for row in re.finditer(r"<tr[^>]*>(.*?)</tr>", table.group(), re.DOTALL):
        cells = re.findall(r"<td[^>]*>(.*?)</td>", row.group(), re.DOTALL)
        cells_clean = [re.sub(r"<[^>]+>", "", c).strip() for c in cells]
        if len(cells_clean) < 4:
            continue
        # Struttura: [seq, doc_type, filename, description, size]
        seq, doc_type, filename, description = cells_clean[0], cells_clean[1], cells_clean[2], cells_clean[3]
        # Estrai il nome file dal link (rimuove "iXBRL" etc.)
        fname = re.sub(r"&nbsp;.*", "", filename).strip()
        if not fname or fname.endswith((".jpg", ".png", ".gif", ".xml", ".xsd", ".zip")):
            continue
        doc_type_lower = doc_type.lower()
        fname_lower = fname.lower()
        # Deve essere un exhibit EX-99.
        if "ex-99" not in doc_type_lower and "ex99" not in fname_lower:
            continue
        # Heuristic: e' un transcript se la descrizione menziona keyword tipiche.
        desc_lower = description.lower()
        is_transcript = any(k in desc_lower for k in TRANSCRIPT_KEYWORDS)
        if not is_transcript and "transcript" in fname_lower:
            is_transcript = True
        candidates.append(
            {
                "filename": fname,
                "description": description,
                "doc_type": doc_type,
                "url": dir_url + fname,
                "is_transcript": is_transcript,
            }
        )

    # Preferisci candidati marcati come transcript; altrimenti primo EX-99.1.
    transcripts = [c for c in candidates if c["is_transcript"]]
    if transcripts:
        return transcripts[0]
    for c in candidates:
        if "99.1" in c["doc_type"] or "ex991" in c["filename"].lower():
            return c
    return candidates[0] if candidates else None


def _process_ticker_8k(sec: SECSession, ticker: str, cik: str) -> list[dict]:
    """Recupera tutti i filing 8-K con item 2.02 e i relativi exhibit per un
    singolo ticker. Funzione worker per la parallelizzazione."""
    rows = []
    filings = fetch_8k_filings(sec, cik)
    for f in filings:
        exhibit = find_transcript_exhibit(sec, cik, f["accession"])
        if not exhibit:
            continue
        rows.append(
            {
                "ticker": ticker,
                "cik": cik,
                "filing_date": f["filing_date"],
                "accession": f["accession"],
                "exhibit_filename": exhibit["filename"],
                "exhibit_description": exhibit["description"],
                "exhibit_url": exhibit["url"],
                "is_transcript": exhibit["is_transcript"],
            }
        )
    return rows


def build_index(
    tickers: list[str], sec: SECSession | None = None
) -> pd.DataFrame:
    sec = sec or SECSession()
    ciks = resolve_ciks(tickers)

    to_process = [
        (r["ticker"], r["cik"])
        for _, r in ciks.iterrows()
        if r["cik"]
    ]

    rows: list[dict] = []
    failed: list[tuple[str, str]] = []  # (ticker, cik) per retry
    workers = config.SEC_PARALLEL_WORKERS

    # Mappa ticker -> cik per il retry.
    cik_map = {t: c for t, c in to_process}
    pending = list(to_process)

    for round_n in (1, 2):
        if not pending:
            break
        label = "EDGAR 8-K transcripts" if round_n == 1 else "EDGAR 8-K retry"
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(_process_ticker_8k, sec, ticker, cik): ticker
                for ticker, cik in pending
            }
            pending = []  # raccolgo i falliti qui per il giro 2
            with tqdm(total=len(futures), desc=label) as pbar:
                for fut in as_completed(futures):
                    ticker = futures[fut]
                    try:
                        rows.extend(fut.result())
                    except Exception as exc:
                        # Un ticker fallito non abortisce l'indice: se e' il
                        # primo giro, lo mette da parte per il retry; al
                        # secondo giro lo logga e lo salta definitivamente.
                        if round_n == 1:
                            pending.append((ticker, cik_map[ticker]))
                        else:
                            failed.append(f"{ticker} ({type(exc).__name__})")
                    pbar.update(1)

    if failed:
        print(f"[EDGAR-T] {len(failed)} ticker falliti dopo retry (es. {failed[:5]})")

    df = pd.DataFrame(rows)
    if not df.empty:
        df["filing_date"] = pd.to_datetime(df["filing_date"])
        df = df.sort_values(["ticker", "filing_date"]).reset_index(drop=True)
    df.to_parquet(INDEX_OUT, index=False)
    n_transcripts = df["is_transcript"].sum() if not df.empty else 0
    print(
        f"[EDGAR-T] {len(df)} exhibit 8-K trovati "
        f"({n_transcripts} identificati come transcript, "
        f"{df['ticker'].nunique() if not df.empty else 0} aziende) -> {INDEX_OUT}"
    )
    return df


def _download_one_transcript(sec: SECSession, r: pd.Series) -> bool:
    """Scarica un singolo exhibit transcript. Ritorna True se scaricato."""
    fdate = pd.to_datetime(r["filing_date"]).strftime("%Y-%m-%d")
    ext = Path(str(r["exhibit_filename"])).suffix or ".htm"
    out_dir = DOCS_DIR / str(r["ticker"])
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{fdate}_{r['accession']}{ext}"
    if out.exists():
        return False
    resp = sec.get(r["exhibit_url"])
    if resp.status_code == 200:
        out.write_bytes(resp.content)
        return True
    return False


def download_documents(
    index: pd.DataFrame | None = None, sec: SECSession | None = None
) -> int:
    sec = sec or SECSession()
    if index is None:
        index = pd.read_parquet(INDEX_OUT)
    DOCS_DIR.mkdir(parents=True, exist_ok=True)

    records = [r for _, r in index.iterrows()]
    workers = config.SEC_PARALLEL_WORKERS
    downloaded = 0

    for round_n in (1, 2):
        if not records:
            break
        label = "EDGAR transcript docs" if round_n == 1 else "EDGAR transcript docs retry"
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(_download_one_transcript, sec, r) for r in records]
            with tqdm(total=len(futures), desc=label) as pbar:
                for fut in as_completed(futures):
                    try:
                        if fut.result():
                            downloaded += 1
                    except Exception:
                        pass
                    pbar.update(1)
        if round_n == 1:
            still_missing = []
            for r in records:
                fdate = pd.to_datetime(r["filing_date"]).strftime("%Y-%m-%d")
                ext = Path(str(r["exhibit_filename"])).suffix or ".htm"
                out = DOCS_DIR / str(r["ticker"]) / f"{fdate}_{r['accession']}{ext}"
                if not out.exists():
                    still_missing.append(r)
            records = still_missing

    failed = len(records)
    if failed:
        print(f"[EDGAR-T] {failed} download falliti dopo retry (saltati)")
    print(f"[EDGAR-T] Documenti scaricati: {downloaded} -> {DOCS_DIR}")
    return downloaded


def run(tickers: list[str], download: bool = True) -> pd.DataFrame:
    sec = SECSession()
    index = build_index(tickers, sec=sec)
    if download and not index.empty:
        download_documents(index, sec=sec)
    return index


def main() -> None:
    from src.sample import load_sample_tickers

    run(load_sample_tickers(), download=True)


if __name__ == "__main__":
    main()
