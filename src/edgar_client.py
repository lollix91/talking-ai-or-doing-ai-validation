"""Sessione HTTP condivisa per le API SEC/EDGAR.

Rispetta il limite ufficiale SEC (<=10 req/s) e usa lo User-Agent richiesto.
Thread-safe: il rate limiter e' protetto da un lock cosicche' piu' thread
possono condividere la stessa istanza senza superare il limite.
"""
from __future__ import annotations

import threading
import time

import requests

import config


class SECSession:
    """Sessione HTTP verso SEC con throttling thread-safe per rispettare i
    rate limit. requests.Session usa un connection pool urllib3 che e'
    thread-safe per richieste concorrenti."""

    def __init__(self) -> None:
        self.session = requests.Session()
        self.session.headers.update(
            {"User-Agent": config.SEC_USER_AGENT, "Accept-Encoding": "gzip, deflate"}
        )
        self._min_interval = 1.0 / max(config.SEC_REQUESTS_PER_SECOND, 1)
        self._last = 0.0
        self._lock = threading.Lock()

    def _throttle(self) -> None:
        """Prenota uno slot nel rate limiter (thread-safe). Registra subito il
        timestamp cosi' il thread successivo aspetta il suo turno."""
        with self._lock:
            wait = self._min_interval - (time.monotonic() - self._last)
            if wait > 0:
                time.sleep(wait)
            self._last = time.monotonic()

    def get(self, url: str) -> requests.Response:
        # La richiesta HTTP avviene fuori dal lock: piu' thread possono
        # avere richieste in-flight contemporaneamente, ma il rate limiter
        # garantisce che non ne partano piu' di SEC_REQUESTS_PER_SECOND al
        # secondo. Ogni tentativo riprenota il proprio slot.
        last_exc: Exception | None = None
        for attempt in range(1, config.MAX_RETRIES + 1):
            self._throttle()
            try:
                resp = self.session.get(url, timeout=config.REQUEST_TIMEOUT_SEC)
            except requests.exceptions.RequestException as exc:
                # Errori transitori di rete (ReadTimeout, ConnectionError,
                # ecc.): backoff e retry. Con molti worker paralleli su SEC
                # un timeout occasionale e' normale e non deve abortire.
                last_exc = exc
                time.sleep(config.RETRY_BACKOFF_SEC * attempt)
                continue
            if resp.status_code == 404:
                return resp
            if resp.status_code in (429, 500, 502, 503, 504):
                time.sleep(config.RETRY_BACKOFF_SEC * attempt)
                continue
            resp.raise_for_status()
            return resp
        # Esauriti i retry: se l'ultima causa era un'eccezione di rete, rilanciala.
        if last_exc is not None:
            raise last_exc
        resp.raise_for_status()
        return resp
