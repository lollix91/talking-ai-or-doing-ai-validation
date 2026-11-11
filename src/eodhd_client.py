"""Client HTTP minimale per le API EODHD.

Gestisce autenticazione, rate limiting soft, retry con backoff e caching
opzionale delle risposte grezze su disco (utile per riproducibilita').
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import requests

import config


class EODHDClient:
    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or config.require_api_key()
        self.session = requests.Session()
        self._min_interval = 60.0 / max(config.REQUESTS_PER_MINUTE, 1)
        self._last_request_ts = 0.0

    # -- rate limiting ------------------------------------------------------
    def _throttle(self) -> None:
        elapsed = time.monotonic() - self._last_request_ts
        wait = self._min_interval - elapsed
        if wait > 0:
            time.sleep(wait)
        self._last_request_ts = time.monotonic()

    # -- richiesta base -----------------------------------------------------
    def _get(self, url: str, params: dict[str, Any]) -> requests.Response:
        params = {**params, "api_token": self.api_key}
        last_exc: Exception | None = None
        for attempt in range(1, config.MAX_RETRIES + 1):
            self._throttle()
            try:
                resp = self.session.get(
                    url, params=params, timeout=config.REQUEST_TIMEOUT_SEC
                )
                # 404 = simbolo non disponibile: non ha senso ritentare.
                if resp.status_code == 404:
                    return resp
                # 429/5xx: ritenta con backoff.
                if resp.status_code in (429, 500, 502, 503, 504):
                    raise requests.HTTPError(f"HTTP {resp.status_code}")
                resp.raise_for_status()
                return resp
            except (requests.RequestException, requests.HTTPError) as exc:
                last_exc = exc
                if attempt < config.MAX_RETRIES:
                    time.sleep(config.RETRY_BACKOFF_SEC * attempt)
        raise RuntimeError(f"Richiesta fallita dopo i retry: {url} ({last_exc})")

    # -- endpoint di alto livello ------------------------------------------
    def get_json(
        self,
        url: str,
        params: dict[str, Any] | None = None,
        cache_path: Path | None = None,
    ) -> Any | None:
        """GET che ritorna JSON. Ritorna None se 404. Salva la risposta grezza
        in cache_path se fornito."""
        params = params or {}
        params.setdefault("fmt", "json")
        resp = self._get(url, params)
        if resp.status_code == 404:
            return None
        data = resp.json()
        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(json.dumps(data), encoding="utf-8")
        return data

    def fundamentals(self, ticker: str, cache_path: Path | None = None) -> Any | None:
        url = config.EODHD_FUNDAMENTALS_URL.format(ticker=ticker)
        return self.get_json(url, cache_path=cache_path)

    def index_filter(self, index_code: str, filter_expr: str) -> Any | None:
        url = config.EODHD_INDEX_URL.format(index=index_code)
        return self.get_json(url, params={"filter": filter_expr})

    def eod(
        self,
        ticker: str,
        start: str,
        end: str,
        cache_path: Path | None = None,
    ) -> Any | None:
        url = config.EODHD_EOD_URL.format(ticker=ticker)
        return self.get_json(
            url,
            params={"from": start, "to": end, "period": "d", "order": "a"},
            cache_path=cache_path,
        )
