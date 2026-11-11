"""Configurazione centrale del progetto di download dati EODHD/SEC.

Definisce percorsi, finestra temporale, indice di riferimento e parametri
delle API. Pensato per uno studio replicabile (target journal Q1).
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# Carica variabili da .env se presente (l'API key NON va hardcodata nel codice).
load_dotenv()

# --- Percorsi ---------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"          # risposte grezze (JSON) per riproducibilita'
PROCESSED_DIR = DATA_DIR / "processed"  # output puliti in Parquet

OUTPUTS_DIR = ROOT_DIR / "outputs"    # tabelle (CSV) e figure (PNG) della Fase 4

for _d in (RAW_DIR, PROCESSED_DIR, RAW_DIR / "fundamentals", RAW_DIR / "eod"):
    _d.mkdir(parents=True, exist_ok=True)

# --- Finestra temporale dello studio ---------------------------------------
# 10 anni per il progetto longitudinale (P2); l'ultimo anno serve alla
# volatilita' di P1.
START_DATE = "2015-01-01"
END_DATE = "2025-12-31"
# Anno di riferimento per la volatilita' (P1).
VOLATILITY_YEAR = 2024
# Finestra dello studio come interi (per filtrare i panel della Fase 4).
STUDY_YEAR_MIN = 2015
# I 10-K sono scaricati per filing_date <= END_DATE: i 10-K del FY2025 sono
# depositati nel 2026 e mancano quasi tutti (match ~22%), quindi l'ultimo
# anno fiscale analizzabile in modo coerente e' il 2024.
STUDY_YEAR_MAX = 2024
# Percentili per la winsorizzazione delle variabili continue (Fase 4).
WINSOR_LOWER = 0.01
WINSOR_UPPER = 0.99
# Permanenza minima nell'S&P 500 (giorni, dentro la finestra di studio):
# esclude le spin-off inserite nell'indice per 1-3 giorni e i costituenti
# spuri (es. ADR estere senza 10-K).
MIN_INDEX_DAYS = 30
# Correzioni note allo storico costituenti EODHD: Wabtec (aggiunta 27/02/2019)
# e Ingersoll Rand Inc. (subentrata a Ingersoll-Rand plc il 03/03/2020) sono
# tuttora nell'indice ma il feed le riporta con permanenza di un giorno.
INDEX_MEMBERSHIP_CORRECTIONS = {
    "WAB.US": ("2019-02-26", None),
    "IR.US": ("2020-03-02", None),
}

# --- Indice di riferimento (campione) ---------------------------------------
# S&P 500 ricostruito point-in-time per evitare survivorship bias.
INDEX_CODE = "GSPC.INDX"
# Suffisso di exchange EODHD per i titoli USA.
US_EXCHANGE_SUFFIX = "US"

# --- API EODHD --------------------------------------------------------------
EODHD_API_KEY = os.getenv("EODHD_API_KEY", "")
EODHD_BASE = "https://eodhd.com/api"
EODHD_FUNDAMENTALS_URL = EODHD_BASE + "/fundamentals/{ticker}"
EODHD_EOD_URL = EODHD_BASE + "/eod/{ticker}"
EODHD_INDEX_URL = EODHD_BASE + "/fundamentals/{index}"

# Rispetto del limite per-minuto (1000 req/min sui piani a pagamento):
# manteniamo un margine di sicurezza.
REQUESTS_PER_MINUTE = 500
MAX_RETRIES = 4
RETRY_BACKOFF_SEC = 3.0
REQUEST_TIMEOUT_SEC = 60

# --- SEC / EDGAR ------------------------------------------------------------
SEC_CIK_TICKER_URL = "https://www.sec.gov/files/company_tickers.json"
# La SEC richiede uno User-Agent identificativo (email di contatto).
SEC_USER_AGENT = os.getenv(
    "SEC_USER_AGENT", "Dati EODHD Research research@example.com"
)
# EDGAR submissions API (storico filing per CIK).
EDGAR_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik10}.json"
EDGAR_SUBMISSIONS_FILE_URL = "https://data.sec.gov/submissions/{name}"
# URL del documento primario di un filing.
EDGAR_ARCHIVE_DOC_URL = (
    "https://www.sec.gov/Archives/edgar/data/{cik}/{accession_nodash}/{document}"
)
# Form da scaricare (report annuali + eventuali rettifiche).
EDGAR_FORM_TYPES = ("10-K", "10-K/A")
# Limite ufficiale SEC: max 10 richieste/secondo. Teniamo un margine.
SEC_REQUESTS_PER_SECOND = 7
# Numero di worker paralleli per i download EDGAR. Il rate limiter
# (SEC_REQUESTS_PER_SECOND) e' condiviso tra tutti i worker via lock,
# quindi questo controlla solo quante richieste sono in-flight
# contemporaneamente, non il throughput globale.
SEC_PARALLEL_WORKERS = 8


def require_api_key() -> str:
    """Ritorna l'API key EODHD o solleva un errore esplicito se mancante."""
    if not EODHD_API_KEY:
        raise RuntimeError(
            "EODHD_API_KEY non impostata. Crea un file .env (vedi .env.example) "
            "oppure esporta la variabile d'ambiente EODHD_API_KEY."
        )
    return EODHD_API_KEY
