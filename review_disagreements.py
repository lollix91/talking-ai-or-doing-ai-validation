"""Rassegna delle divergenze tra annotatori sul sottocampione di 80 10-K.

Genera outputs/disagreements.html (e data/processed/disagreements.csv) con i
soli documenti in cui gli annotatori (human, human2, llm) non concordano
sull'etichetta binaria (menziona l'IA) o sulla sostanza, con le tre etichette
affiancate, le note e gli estratti di testo evidenziati.

Le etichette originali NON vengono modificate: le decisioni finali
(aggiudicazione) si scelgono nella pagina interattiva e vengono salvate a
parte in data/processed/annotations_adjudicated.json.

Uso:
    python review_disagreements.py --serve            # pagina interattiva
    python review_disagreements.py --serve --binary   # solo divergenze sul binario
    python review_disagreements.py                    # solo rassegna HTML/CSV
"""
from __future__ import annotations

import argparse
import html
import json
import re
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs

import pandas as pd

import annotation_app as app
import config

P = config.PROCESSED_DIR
FILES = {"human": P / "annotations.json", "human2": P / "annotations_human2.json",
         "llm": P / "annotations_llm.json"}
OUT_HTML = config.OUTPUTS_DIR / "disagreements.html"
OUT_CSV = P / "disagreements.csv"
ADJ_JSON = P / "annotations_adjudicated.json"
PORT = 8788
SUBSTANCE = {"boilerplate_risk_only": 1, "operational_mention": 2, "core_to_business": 3}


def _label(a: dict | None) -> tuple[int | None, str]:
    if a is None:
        return None, ""
    if a["ai_mention"] != "present":
        return 0, "none"
    return SUBSTANCE.get(a.get("ai_substance"), 0), a.get("ai_substance", "")


def build(binary_only: bool) -> pd.DataFrame:
    sub = pd.read_csv(P / "annotation_subsample_llm.csv")
    ann = {r: json.loads(f.read_text(encoding="utf-8")) for r, f in FILES.items() if f.exists()}
    rows = []
    for i, r in sub.iterrows():
        labs = {k: _label(v.get(r["accession"])) for k, v in ann.items()}
        if any(l[0] is None for l in labs.values()):
            continue
        scores = [l[0] for l in labs.values()]
        binary = {int(s > 0) for s in scores}
        if len(binary) > 1 or (not binary_only and len(set(scores)) > 1):
            rows.append({
                "idx": i, "accession": r["accession"], "ticker": r["ticker"],
                "fiscal_year": int(r["fiscal_year"]), "stratum": r["stratum"],
                "binary_disagreement": int(len(binary) > 1),
                **{f"{k}": l[1] for k, l in labs.items()},
                **{f"{k}_note": ann[k][r["accession"]].get("notes", "") for k in ann},
                "adjudicated": "",
            })
    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)
    return df


def render(df: pd.DataFrame) -> None:
    sub = pd.read_csv(P / "annotation_subsample_llm.csv")
    parts = [f"<h1>Divergenze tra annotatori ({len(df)} documenti)</h1>"]
    for _, d in df.iterrows():
        r = sub.iloc[int(d["idx"])]
        _, snips = app.extract_snippets(r, window=260, max_snip=6)
        labs = "".join(
            f"<tr><td><b>{k}</b></td><td>{html.escape(d[k])}</td>"
            f"<td>{html.escape(str(d[f'{k}_note']))}</td></tr>"
            for k in ("human", "human2", "llm") if k in d
        )
        sn = "".join(f"<div class='snip'>…{app._highlight(s)}…</div>" for s in snips) or "<i>nessun estratto</i>"
        parts.append(
            f"<h2>#{d['idx']} — {d['ticker']} FY{d['fiscal_year']} "
            f"{'(divergenza binaria)' if d['binary_disagreement'] else '(solo sostanza)'}</h2>"
            f"<table><tr><th>annotatore</th><th>etichetta</th><th>nota</th></tr>{labs}</table>{sn}"
        )
    OUT_HTML.write_bytes(app._page("".join(parts)))


def _load_adj() -> dict:
    return json.loads(ADJ_JSON.read_text(encoding="utf-8")) if ADJ_JSON.exists() else {}


def render_interactive(df: pd.DataFrame) -> bytes:
    sub = pd.read_csv(P / "annotation_subsample_llm.csv")
    adj = _load_adj()
    done = int(df["accession"].isin(adj).sum())
    parts = [f"<h1>Aggiudicazione divergenze: {done}/{len(df)} decise</h1>"
             "<p>Per ogni documento scegli l'etichetta finale e premi Salva. "
             "Le etichette originali non vengono modificate.</p>"]
    for _, d in df.iterrows():
        r = sub.iloc[int(d["idx"])]
        prev = adj.get(d["accession"], {})
        _, snips = app.extract_snippets(r, window=260, max_snip=6)
        labs = "".join(
            f"<tr><td><b>{k}</b></td><td>{html.escape(d[k])}</td>"
            f"<td>{html.escape(str(d[f'{k}_note']))}</td></tr>"
            for k in ("human", "human2", "llm") if k in d
        )
        sn = "".join(f"<div class='snip'>…{app._highlight(s)}…</div>" for s in snips) or "<i>nessun estratto</i>"
        opts = [("none", "Nessuna menzione dell'IA"), ("boilerplate_risk_only", "Solo boilerplate / rischio"),
                ("operational_mention", "Menzione operativa"), ("core_to_business", "Centrale nel business")]
        cur = prev.get("ai_substance") if prev.get("ai_mention") == "present" else ("none" if prev else None)
        radios = "".join(
            f"<label><input type='radio' name='choice' value='{v}' {'checked' if cur == v else ''} required> {t}</label>"
            for v, t in opts
        )
        mark = "<span class='done'>decisa</span>" if d["accession"] in adj else "<span class='todo'>da decidere</span>"
        parts.append(
            f"<h2 id='d{d['idx']}'>#{d['idx']} — {d['ticker']} FY{d['fiscal_year']} {mark} "
            f"{'(divergenza binaria)' if d['binary_disagreement'] else '(solo sostanza)'}</h2>"
            f"<table><tr><th>annotatore</th><th>etichetta</th><th>nota</th></tr>{labs}</table>{sn}"
            f"<form method='post' action='/save/{d['idx']}'>{radios}"
            f"<label>Nota (opzionale):<br><textarea name='notes'>{html.escape(prev.get('notes', ''))}</textarea></label>"
            f"<button type='submit'>Salva</button></form>"
        )
    return app._page("".join(parts))


def serve(df: pd.DataFrame) -> None:
    sub = pd.read_csv(P / "annotation_subsample_llm.csv")
    by_idx = {int(r["idx"]): r for _, r in df.iterrows()}

    class H(BaseHTTPRequestHandler):
        def do_GET(self):
            body = render_interactive(df)
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            m = re.match(r"^/save/(\d+)$", self.path)
            n = int(self.headers.get("Content-Length", 0))
            form = parse_qs(self.rfile.read(n).decode("utf-8"))
            if m and int(m.group(1)) in by_idx:
                i = int(m.group(1))
                choice = form["choice"][0]
                adj = _load_adj()
                adj[by_idx[i]["accession"]] = {
                    "ticker": by_idx[i]["ticker"], "fiscal_year": int(by_idx[i]["fiscal_year"]),
                    "ai_mention": "none" if choice == "none" else "present",
                    "ai_substance": "" if choice == "none" else choice,
                    "notes": form.get("notes", [""])[0], "annotator": "adjudicated",
                }
                ADJ_JSON.write_text(json.dumps(adj, indent=1, ensure_ascii=False), encoding="utf-8")
                self.send_response(303)
                self.send_header("Location", f"/#d{i}")
                self.end_headers()
                return
            self.send_response(404)
            self.end_headers()

        def log_message(self, *a):
            pass

    url = f"http://127.0.0.1:{PORT}"
    print(f"[DIV] Server su {url} — Ctrl+C per uscire. Decisioni in {ADJ_JSON}")
    webbrowser.open(url)
    ThreadingHTTPServer(("127.0.0.1", PORT), H).serve_forever()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--binary", action="store_true", help="solo divergenze sul binario")
    ap.add_argument("--no-open", action="store_true")
    ap.add_argument("--serve", action="store_true", help="pagina interattiva per decidere le etichette finali")
    args = ap.parse_args()
    df = build(args.binary)
    print(f"[DIV] {len(df)} documenti con divergenze "
          f"({int(df['binary_disagreement'].sum()) if len(df) else 0} sul binario) -> {OUT_CSV}")
    if args.serve and len(df):
        serve(df)
        return
    if len(df):
        render(df)
        print(f"[DIV] Rassegna HTML -> {OUT_HTML}")
        if not args.no_open:
            webbrowser.open(OUT_HTML.as_uri())


if __name__ == "__main__":
    main()
