"""Web-app locale per la validazione manuale dello score di adozione IA.

Uso:
    python annotation_app.py --sample      # campiona 150 10-K stratificati
    python annotation_app.py --serve       # server locale http://127.0.0.1:8787
    python annotation_app.py --agreement   # metriche di validazione -> CSV
    python annotation_app.py --serve --blind --subsample --annotator human2
                                           # seconda annotazione umana cieca (80 doc)

Le annotazioni sono salvate in data/processed/annotations.json (incrementale:
puoi chiudere e riprendere in qualunque momento).

Schema di annotazione per ciascun 10-K:
  - ai_mention:   'none' | 'present'   -> il documento menziona l'IA?
  - ai_substance: 'boilerplate_risk_only' | 'operational_mention'
                  | 'core_to_business'  -> se presente, quanto e' sostanziale
  - notes:        testo libero opzionale
"""
from __future__ import annotations

import argparse
import html
import json
import re
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np
import pandas as pd

import config
from src import ai_keywords as kw
from src.nlp import DOCS_DIR, html_to_text

SAMPLE_CSV = config.PROCESSED_DIR / "annotation_sample.csv"
ANNOT_JSON = config.PROCESSED_DIR / "annotations.json"
N_SAMPLE = 150
SEED = 42
PORT = 8787
# Modalita' cieca: nasconde score automatico e strato all'annotatore.
BLIND = False
ANNOTATOR = "human"

# ---------------------------------------------------------------------------
# Campionamento stratificato
# ---------------------------------------------------------------------------

def build_sample(n: int = N_SAMPLE, seed: int = SEED) -> pd.DataFrame:
    """150 10-K stratificati sullo score: non-zero-inflated + terzili dei
    positivi, cosi' il campione copre tutto lo spettro."""
    scores = pd.read_parquet(config.PROCESSED_DIR / "ai_adoption_scores.parquet")
    scores = scores.dropna(subset=["fiscal_year", "ai_score_per_10k"])

    # primary_document serve per ritrovare il file su disco.
    idx = pd.read_parquet(config.PROCESSED_DIR / "edgar_10k_index.parquet")[
        ["ticker", "accession", "primary_document"]
    ]
    scores = scores.merge(idx, on=["ticker", "accession"], how="left")
    scores = scores.dropna(subset=["primary_document"])

    def _stratum(s: float) -> str:
        return "zero" if s <= 0 else "positive"

    scores["stratum"] = scores["ai_score_per_10k"].map(_stratum)
    pos = scores["ai_score_per_10k"] > 0
    q = scores.loc[pos, "ai_score_per_10k"].quantile([1 / 3, 2 / 3])
    scores.loc[pos, "stratum"] = pd.cut(
        scores.loc[pos, "ai_score_per_10k"],
        bins=[-np.inf, q.iloc[0], q.iloc[1], np.inf],
        labels=["pos_low", "pos_mid", "pos_high"],
    ).astype(str)

    rng = np.random.default_rng(seed)
    per_stratum = n // scores["stratum"].nunique()
    parts = []
    for stratum, grp in scores.groupby("stratum"):
        k = min(per_stratum, len(grp))
        parts.append(grp.sample(n=k, random_state=int(rng.integers(1e9))))
    sample = pd.concat(parts).sort_values(["ticker", "filing_date"])
    sample = sample.reset_index(drop=True)
    sample.to_csv(SAMPLE_CSV, index=False)
    print(
        f"[ANN] Campione {len(sample)} documenti -> {SAMPLE_CSV}\n"
        f"[ANN] Strati: {sample['stratum'].value_counts().to_dict()}"
    )
    return sample


# ---------------------------------------------------------------------------
# Estrazione snippet (contesti delle keyword)
# ---------------------------------------------------------------------------

_ALL_PATTERNS = {
    **{t: re.compile(r"\b" + re.escape(t.lower()) + r"\b", re.IGNORECASE)
       for t in kw.ALL_KEYWORDS + kw.AMBIGUOUS},
    **{t: re.compile(r"\b" + re.escape(t) + r"\b") for t in kw.ACRONYMS},
}


def _doc_path(row: pd.Series) -> Path:
    fdate = pd.to_datetime(row["filing_date"]).strftime("%Y-%m-%d")
    ext = Path(str(row["primary_document"])).suffix or ".htm"
    return DOCS_DIR / str(row["ticker"]) / f"{fdate}_{row['accession']}{ext}"


def extract_snippets(row: pd.Series, window: int = 260, max_snip: int = 30):
    """Testo grezzo + finestre di contesto attorno alle keyword IA."""
    path = _doc_path(row)
    if not path.exists():
        return None, []
    text = html_to_text(path.read_text(encoding="utf-8", errors="replace"))
    # posizioni di tutte le keyword
    spans = []
    for term, pat in _ALL_PATTERNS.items():
        for m in pat.finditer(text):
            spans.append((max(0, m.start() - window), min(len(text), m.end() + window)))
    spans.sort()
    # unisce finestre sovrapposte
    merged: list[list[int]] = []
    for s, e in spans:
        if merged and s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    snippets = [text[s:e] for s, e in merged[:max_snip]]
    return text, snippets


def _highlight(text: str) -> str:
    esc = html.escape(text)
    for term in sorted(kw.ALL_KEYWORDS + kw.AMBIGUOUS, key=len, reverse=True):
        esc = re.sub(
            r"(?i)\b(" + re.escape(html.escape(term)) + r")\b",
            r"<mark>\1</mark>",
            esc,
        )
    for term in kw.ACRONYMS:
        esc = re.sub(r"\b(" + re.escape(term) + r")\b", r"<mark>\1</mark>", esc)
    return esc


# ---------------------------------------------------------------------------
# Persistenza annotazioni
# ---------------------------------------------------------------------------

def load_annotations() -> dict:
    if ANNOT_JSON.exists():
        return json.loads(ANNOT_JSON.read_text(encoding="utf-8"))
    return {}


def save_annotation(accession: str, data: dict) -> None:
    ann = load_annotations()
    ann[accession] = {**data, "annotator": ANNOTATOR}
    ANNOT_JSON.write_text(json.dumps(ann, indent=1, ensure_ascii=False),
                          encoding="utf-8")


# ---------------------------------------------------------------------------
# Server HTTP (stdlib, nessuna dipendenza)
# ---------------------------------------------------------------------------

CSS = """
body{font-family:system-ui,Arial,sans-serif;max-width:960px;margin:24px auto;padding:0 16px;color:#222}
table{border-collapse:collapse;width:100%}td,th{border:1px solid #ddd;padding:4px 8px;font-size:13px}
mark{background:#ffe58a;padding:0 1px}
.snip{border-left:3px solid #4a90d9;background:#f7f9fc;padding:8px 10px;margin:10px 0;font-size:14px}
form{border:1px solid #ccc;padding:16px;margin-top:20px;background:#fff}
label{display:block;margin:8px 0}
textarea{width:100%;height:60px}
button{padding:8px 18px;font-size:15px;cursor:pointer}
.done{color:green}.todo{color:#b00}
nav{margin:12px 0}
"""


def _page(body: str) -> bytes:
    return (
        f"<!doctype html><html><head><meta charset='utf-8'>"
        f"<title>AI 10-K annotation</title><style>{CSS}</style></head>"
        f"<body>{body}</body></html>"
    ).encode("utf-8")


def render_index(sample: pd.DataFrame, ann: dict) -> str:
    done = sample["accession"].isin(ann).sum()
    rows = []
    for i, r in sample.iterrows():
        a = ann.get(r["accession"])
        status = (
            f"<span class='done'>✓ {a['ai_mention']}/{a.get('ai_substance','')}</span>"
            if a else "<span class='todo'>da fare</span>"
        )
        score_cells = "" if BLIND else (
            f"<td>{r['ai_score_per_10k']:.2f}</td><td>{r.get('stratum','')}</td>"
        )
        rows.append(
            f"<tr><td>{i}</td><td>{r['ticker']}</td><td>{int(r['fiscal_year'])}</td>"
            f"{score_cells}<td>{status}</td><td><a href='/doc/{i}'>annota</a></td></tr>"
        )
    score_head = "" if BLIND else "<th>score</th><th>strato</th>"
    body = (
        f"<h1>Validazione manuale score IA{' (cieca)' if BLIND else ''}</h1>"
        f"<p>Annotati: <b>{done}/{len(sample)}</b> — salvataggio in "
        f"<code>{ANNOT_JSON}</code></p>"
        f"<table><tr><th>#</th><th>ticker</th><th>FY</th>{score_head}"
        f"<th>stato</th><th></th></tr>{''.join(rows)}</table>"
    )
    return _page(body).decode()


def render_doc(sample: pd.DataFrame, ann: dict, idx: int, saved: bool = False) -> str:
    r = sample.iloc[idx]
    text, snippets = extract_snippets(r)
    prev_a = ann.get(r["accession"], {})
    snip_html = ""
    if snippets:
        for s in snippets:
            snip_html += f"<div class='snip'>…{_highlight(s)}…</div>"
    elif text is not None:
        snip_html = (
            "<p><i>Nessuna keyword IA trovata — estratto iniziale:</i></p>"
            f"<div class='snip'>{html.escape(text[:2000])}</div>"
        )
    else:
        snip_html = "<p><b>File non trovato su disco.</b></p>"

    sec_scores = " | ".join(
        f"{k.replace('ai_score_','')}: {r[k]:.2f}"
        for k in ["ai_score_business", "ai_score_risk_factors", "ai_score_mdna"]
        if k in r and pd.notna(r[k])
    )
    nav = (
        f"<nav><a href='/'>← elenco</a> | "
        f"<a href='/doc/{idx-1}'>← prec</a> | <a href='/doc/{idx+1}'>succ →</a>"
        f" | <a href='/next'>prossimo da fare</a></nav>"
        if idx > 0 else
        f"<nav><a href='/'>← elenco</a> | <a href='/doc/{idx+1}'>succ →</a>"
        f" | <a href='/next'>prossimo da fare</a></nav>"
    )
    note = "<p style='color:green'><b>Annotazione salvata.</b></p>" if saved else ""
    score_html = "" if BLIND else f"""
<p>Score automatico: <b>{r['ai_score_per_10k']:.2f}</b> per 10k parole
({int(r['ai_keyword_count'])} keyword) — strato: {r.get('stratum','')}<br>
Per sezione: {sec_scores or 'n.d.'}</p>"""
    body = f"""
{nav}
<h2>{r['ticker']} — 10-K FY{int(r['fiscal_year'])}</h2>
{score_html}
{note}
{snip_html}
<form method='post' action='/annotate/{idx}'>
  <b>Il documento menziona l'IA?</b>
  <label><input type='radio' name='ai_mention' value='none' {'checked' if prev_a.get('ai_mention')=='none' else ''} required> Nessuna menzione / solo termini fuorvianti</label>
  <label><input type='radio' name='ai_mention' value='present' {'checked' if prev_a.get('ai_mention')=='present' else ''}> Menziona l'IA</label>
  <b>Se presente, quanto è sostanziale?</b>
  <label><input type='radio' name='ai_substance' value='boilerplate_risk_only' {'checked' if prev_a.get('ai_substance')=='boilerplate_risk_only' else ''}> Solo boilerplate / fattori di rischio</label>
  <label><input type='radio' name='ai_substance' value='operational_mention' {'checked' if prev_a.get('ai_substance')=='operational_mention' else ''}> Menzione operativa (progetti, prodotti, investimenti)</label>
  <label><input type='radio' name='ai_substance' value='core_to_business' {'checked' if prev_a.get('ai_substance')=='core_to_business' else ''}> Centrale nel business</label>
  <label>Note (opzionale):<br><textarea name='notes'>{html.escape(prev_a.get('notes',''))}</textarea></label>
  <button type='submit'>Salva e continua</button>
</form>
{nav}
"""
    return _page(body).decode()


def make_handler(sample: pd.DataFrame):
    class Handler(BaseHTTPRequestHandler):
        def _send(self, body: str | bytes, code: int = 200):
            data = body.encode() if isinstance(body, str) else body
            self.send_response(code)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _redirect(self, loc: str):
            self.send_response(303)
            self.send_header("Location", loc)
            self.end_headers()

        def do_GET(self):
            path = urlparse(self.path).path
            ann = load_annotations()
            if path == "/":
                return self._send(render_index(sample, ann))
            if path == "/next":
                pending = sample[~sample["accession"].isin(ann)]
                if pending.empty:
                    return self._redirect("/")
                return self._redirect(f"/doc/{pending.index[0]}")
            m = re.match(r"^/doc/(\d+)$", path)
            if m:
                i = int(m.group(1))
                saved = "saved=1" in (urlparse(self.path).query or "")
                if 0 <= i < len(sample):
                    return self._send(render_doc(sample, ann, i, saved=saved))
            self._send("Not found", 404)

        def do_POST(self):
            m = re.match(r"^/annotate/(\d+)$", urlparse(self.path).path)
            if not m:
                return self._send("Not found", 404)
            idx = int(m.group(1))
            length = int(self.headers.get("Content-Length", 0))
            form = parse_qs(self.rfile.read(length).decode("utf-8"))
            accession = sample.iloc[idx]["accession"]
            save_annotation(
                accession,
                {
                    "ticker": sample.iloc[idx]["ticker"],
                    "fiscal_year": int(sample.iloc[idx]["fiscal_year"]),
                    "ai_mention": form.get("ai_mention", [""])[0],
                    "ai_substance": form.get("ai_substance", [""])[0],
                    "notes": form.get("notes", [""])[0],
                },
            )
            self._redirect(f"/doc/{idx}?saved=1")

        def log_message(self, *args):
            pass

    return Handler


# ---------------------------------------------------------------------------
# Accordo umano-macchina
# ---------------------------------------------------------------------------

def compute_agreement() -> None:
    """Metriche di validazione (pesate per strato) e accordo tra annotatori:
    delegate a src/analysis/validation.py."""
    from src.analysis import validation
    validation.run()


# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Validazione manuale score IA")
    ap.add_argument("--sample", action="store_true")
    ap.add_argument("--serve", action="store_true")
    ap.add_argument("--agreement", action="store_true")
    ap.add_argument("--blind", action="store_true",
                    help="nasconde score e strato (seconda annotazione indipendente)")
    ap.add_argument("--subsample", action="store_true",
                    help="serve gli 80 documenti della seconda annotazione "
                         "(annotation_subsample_llm.csv, ordine casuale)")
    ap.add_argument("--annotator", default="human",
                    help="nome annotatore; != 'human' salva in annotations_<nome>.json")
    args = ap.parse_args()

    global ANNOT_JSON, BLIND, ANNOTATOR
    BLIND, ANNOTATOR = args.blind, args.annotator
    if args.annotator != "human":
        ANNOT_JSON = config.PROCESSED_DIR / f"annotations_{args.annotator}.json"

    if args.sample:
        build_sample()
    if args.agreement:
        compute_agreement()
    if args.serve:
        if args.subsample:
            sample = pd.read_csv(config.PROCESSED_DIR / "annotation_subsample_llm.csv")
        else:
            if not SAMPLE_CSV.exists():
                build_sample()
            sample = pd.read_csv(SAMPLE_CSV)
        url = f"http://127.0.0.1:{PORT}"
        print(f"[ANN] Server su {url} — Ctrl+C per uscire")
        webbrowser.open(url)
        ThreadingHTTPServer(("127.0.0.1", PORT), make_handler(sample)).serve_forever()


if __name__ == "__main__":
    main()
