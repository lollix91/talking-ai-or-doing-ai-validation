"""Orchestratore della pipeline dati del Progetto 2 (EODHD + SEC).

Progetto 2 -- Adozione dell'IA e performance aziendale: studio
longitudinale (2015-2025) che confronta aziende ad alta vs bassa
adozione IA su efficienza, investimento e valutazione. L'adozione IA e'
misurata via NLP sui report annuali 10-K da SEC EDGAR.

Esempi:
    python run.py --all
    python run.py --sec --sample
    python run.py --fundamentals --prices
    python run.py --all --limit 5      # test rapido sui primi 5 ticker
    python run.py --features           # feature engineering (Fase 2)
    python run.py --nlp                # scoring adozione IA sui 10-K
    python run.py --transcripts        # earnings call transcript da 8-K
    python run.py --analysis           # analisi statistica/econometrica (Fase 4)
"""
from __future__ import annotations

import argparse

import config
from src import (
    edgar_10k,
    edgar_transcripts,
    features,
    fundamentals,
    nlp,
    prices,
    sample,
    sec_mapping,
)
from src.eodhd_client import EODHDClient


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Pipeline dati Progetto 2 (adozione IA)"
    )
    p.add_argument("--all", action="store_true", help="Esegue tutti gli step (Fasi 1-3)")
    p.add_argument("--sec", action="store_true", help="Mappatura SEC CIK<->Ticker")
    p.add_argument("--sample", action="store_true", help="Campione S&P500 point-in-time")
    p.add_argument("--fundamentals", action="store_true", help="Dati fondamentali")
    p.add_argument("--prices", action="store_true", help="Prezzi EOD giornalieri")
    p.add_argument("--edgar", action="store_true", help="Report 10-K da SEC EDGAR")
    p.add_argument("--transcripts", action="store_true", help="Earnings call transcript (8-K exhibit)")
    p.add_argument("--features", action="store_true", help="Feature engineering -> panel P2")
    p.add_argument("--nlp", action="store_true", help="Scoring adozione IA sui 10-K")
    p.add_argument("--analysis", action="store_true", help="Analisi statistica/econometrica (Fase 4)")
    p.add_argument(
        "--no-docs",
        action="store_true",
        help="EDGAR: costruisci solo l'indice, non scaricare i documenti",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limita il numero di ticker (utile per test)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    any_step = any(
        [
            args.sec, args.sample, args.fundamentals, args.prices,
            args.edgar, args.transcripts, args.features, args.nlp,
            args.analysis,
        ]
    )
    # --all copre la pipeline di acquisizione (Fasi 1-3). L'analisi (Fase 4)
    # e' uno step separato ed esplicito: non viene inclusa in --all per non
    # accoppiare download pesanti e post-processing.
    run_all = args.all or not any_step

    if args.sec or run_all:
        sec_mapping.main()

    client = None
    tickers = None
    if args.sample or run_all:
        client = EODHDClient()
        sample.build_sample(client)

    needs_tickers = any(
        [args.fundamentals, args.prices, args.edgar, args.transcripts, run_all]
    )
    if needs_tickers:
        tickers = sample.load_sample_tickers()
        if args.limit:
            tickers = tickers[: args.limit]
            print(f"[RUN] Limite attivo: primi {len(tickers)} ticker")

    if args.fundamentals or run_all:
        client = client or EODHDClient()
        fundamentals.download_fundamentals(tickers=tickers, client=client)

    if args.prices or run_all:
        client = client or EODHDClient()
        prices.download_prices(tickers=tickers, client=client)

    if args.edgar or run_all:
        edgar_10k.run(tickers=tickers, download=not args.no_docs)

    if args.transcripts or run_all:
        edgar_transcripts.run(tickers=tickers, download=not args.no_docs)

    if args.features or run_all:
        features.main()

    if args.nlp or run_all:
        nlp.main()

    if args.analysis:
        # Import lazy: le dipendenze di analisi (statsmodels, linearmodels,
        # matplotlib) sono pesanti e servono solo per la Fase 4.
        from src.analysis import audit, car, descriptive, event_study, prep, project2, validation

        prep.main()
        audit.main()
        validation.main()
        descriptive.main()
        project2.main()
        event_study.main()
        car.main()
        print(f"[RUN] Analisi completata. Output in: {config.OUTPUTS_DIR}")

    print(f"[RUN] Completato. Output in: {config.PROCESSED_DIR}")


if __name__ == "__main__":
    main()
