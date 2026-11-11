"""Scarica i report annuali 10-K (e 10-K/A) da SEC EDGAR per il campione.

Flusso:
  1. risolve il CIK di ogni ticker (dai Fundamentals EODHD, fallback sulla
     mappatura ufficiale SEC CIK<->Ticker);
  2. interroga l'API submissions di EDGAR (gestendo la paginazione dei filing
     piu' vecchi) e filtra i form 10-K / 10-K/A nella finestra dello studio;
  3. salva un indice Parquet di tutti i filing trovati;
  4. (opzionale) scarica i documenti primari su disco per l'analisi NLP.

Rispetta il limite SEC (<=10 req/s) e usa lo User-Agent richiesto.
"""
from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

import config
from src.edgar_client import SECSession

INDEX_OUT = config.PROCESSED_DIR / "edgar_10k_index.parquet"
DOCS_DIR = config.RAW_DIR / "edgar" / "10k"


# --- risoluzione CIK --------------------------------------------------------
def _base_code(ticker: str) -> str:
    return ticker.split(".")[0].upper()


def resolve_ciks(tickers: list[str]) -> pd.DataFrame:
    """Ritorna DataFrame ticker -> cik (10 cifre). Preferisce il CIK EODHD
    (copre anche i delisted), fallback sulla mappatura SEC per ticker."""
    mapping: dict[str, str] = {}

    gen_path = config.PROCESSED_DIR / "fundamentals_general.parquet"
    if gen_path.exists():
        gen = pd.read_parquet(gen_path)
        for _, row in gen.iterrows():
            cik = row.get("cik")
            if pd.notna(cik) and str(cik).strip("0"):
                mapping[row["ticker"]] = str(cik).zfill(10)

    sec_path = config.PROCESSED_DIR / "sec_cik_ticker.parquet"
    sec = pd.read_parquet(sec_path) if sec_path.exists() else None

    rows = []
    for t in tickers:
        cik = mapping.get(t)
        if not cik and sec is not None:
            hit = sec.loc[sec["ticker"] == _base_code(t), "cik_padded"]
            if not hit.empty:
                cik = hit.iloc[0]
        rows.append({"ticker": t, "cik": cik})
    return pd.DataFrame(rows)


# --- parsing submissions ----------------------------------------------------
def _iter_filing_rows(recent: dict):
    n = len(recent.get("accessionNumber", []))
    for i in range(n):
        yield {
            "accession": recent["accessionNumber"][i],
            "form": recent["form"][i],
            "filing_date": recent["filingDate"][i],
            "report_date": recent["reportDate"][i],
            "primary_document": recent["primaryDocument"][i],
            "primary_doc_description": recent["primaryDocDescription"][i],
        }


def fetch_filings_for_cik(sec: SECSession, cik10: str) -> list[dict]:
    """Tutti i filing (recent + file paginati) per un CIK."""
    url = config.EDGAR_SUBMISSIONS_URL.format(cik10=cik10)
    resp = sec.get(url)
    if resp.status_code == 404:
        return []
    data = resp.json()
    filings = data.get("filings", {})
    rows = list(_iter_filing_rows(filings.get("recent", {})))
    # File aggiuntivi con i filing piu' vecchi.
    for extra in filings.get("files", []) or []:
        name = extra.get("name")
        # Scarica solo se la finestra dello studio puo' toccarli.
        if extra.get("filingTo", "") < config.START_DATE:
            continue
        eurl = config.EDGAR_SUBMISSIONS_FILE_URL.format(name=name)
        er = sec.get(eurl)
        if er.status_code == 200:
            rows += list(_iter_filing_rows(er.json()))
    return rows


def _doc_url(cik10: str, accession: str, document: str) -> str:
    return config.EDGAR_ARCHIVE_DOC_URL.format(
        cik=int(cik10),
        accession_nodash=accession.replace("-", ""),
        document=document,
    )


def _process_cik_10k(sec: SECSession, ticker: str, cik: str) -> list[dict]:
    """Recupera e filtra i filing 10-K per un singolo CIK. Funzione worker
    per la parallelizzazione."""
    rows = []
    for f in fetch_filings_for_cik(sec, cik):
        if f["form"] not in config.EDGAR_FORM_TYPES:
            continue
        if not (config.START_DATE <= f["filing_date"] <= config.END_DATE):
            continue
        rows.append(
            {
                "ticker": ticker,
                "cik": cik,
                **f,
                "doc_url": _doc_url(cik, f["accession"], f["primary_document"]),
            }
        )
    return rows


def build_index(
    tickers: list[str],
    sec: SECSession | None = None,
) -> pd.DataFrame:
    sec = sec or SECSession()
    ciks = resolve_ciks(tickers)

    # Separa ticker con/senza CIK.
    to_process = []
    missing: list[str] = []
    for _, r in ciks.iterrows():
        if r["cik"]:
            to_process.append((r["ticker"], r["cik"]))
        else:
            missing.append(r["ticker"])

    rows: list[dict] = []
    failed: list[str] = []
    workers = config.SEC_PARALLEL_WORKERS

    cik_map = {t: c for t, c in to_process}
    pending = list(to_process)

    for round_n in (1, 2):
        if not pending:
            break
        label = "EDGAR submissions" if round_n == 1 else "EDGAR submissions retry"
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(_process_cik_10k, sec, ticker, cik): ticker
                for ticker, cik in pending
            }
            pending = []
            with tqdm(total=len(futures), desc=label) as pbar:
                for fut in as_completed(futures):
                    ticker = futures[fut]
                    try:
                        rows.extend(fut.result())
                    except Exception as exc:
                        if round_n == 1:
                            pending.append((ticker, cik_map[ticker]))
                        else:
                            failed.append(f"{ticker} ({type(exc).__name__})")
                    pbar.update(1)

    if failed:
        print(f"[EDGAR] {len(failed)} ticker falliti dopo retry (es. {failed[:5]})")

    df = pd.DataFrame(rows)
    if not df.empty:
        df["filing_date"] = pd.to_datetime(df["filing_date"])
        df["report_date"] = pd.to_datetime(df["report_date"], errors="coerce")
        df = df.sort_values(["ticker", "filing_date"]).reset_index(drop=True)
    df.to_parquet(INDEX_OUT, index=False)

    if missing:
        print(f"[EDGAR] CIK non risolto per {len(missing)} ticker (es. {missing[:5]})")
    print(
        f"[EDGAR] {len(df)} filing 10-K indicizzati "
        f"({df['ticker'].nunique() if not df.empty else 0} aziende) -> {INDEX_OUT}"
    )
    return df


def _download_one_10k(sec: SECSession, r: pd.Series) -> bool:
    """Scarica un singolo documento 10-K. Ritorna True se scaricato."""
    ext = Path(str(r["primary_document"])).suffix or ".htm"
    fdate = pd.to_datetime(r["filing_date"]).strftime("%Y-%m-%d")
    out_dir = DOCS_DIR / str(r["ticker"])
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{fdate}_{r['accession']}{ext}"
    if out.exists():
        return False
    resp = sec.get(r["doc_url"])
    if resp.status_code == 200:
        out.write_bytes(resp.content)
        return True
    return False


def download_documents(
    index: pd.DataFrame | None = None,
    sec: SECSession | None = None,
) -> int:
    """Scarica i documenti primari elencati nell'indice. Salta quelli gia'
    presenti su disco. Ritorna il numero di file scaricati."""
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
        label = "EDGAR 10-K docs" if round_n == 1 else "EDGAR 10-K docs retry"
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(_download_one_10k, sec, r) for r in records]
            with tqdm(total=len(futures), desc=label) as pbar:
                for fut in as_completed(futures):
                    try:
                        if fut.result():
                            downloaded += 1
                    except Exception:
                        pass  # gestito sotto: ricostruisco dai file mancanti
                    pbar.update(1)
        # Ricostruisce la lista dei falliti dai file ancora mancanti su disco.
        if round_n == 1:
            still_missing = []
            for r in records:
                ext = Path(str(r["primary_document"])).suffix or ".htm"
                fdate = pd.to_datetime(r["filing_date"]).strftime("%Y-%m-%d")
                out = DOCS_DIR / str(r["ticker"]) / f"{fdate}_{r['accession']}{ext}"
                if not out.exists():
                    still_missing.append(r)
            records = still_missing

    failed = len(records)  # cio' che resta dopo il secondo giro
    if failed:
        print(f"[EDGAR] {failed} download falliti dopo retry (saltati)")
    print(f"[EDGAR] Documenti scaricati: {downloaded} -> {DOCS_DIR}")
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
