"""Seconda annotazione (LLM, dichiarata) su un sottocampione del set di validazione.

Protocollo:
  - sottocampione di 80 10-K (20 per strato, seed fisso) dei 148 annotati
    manualmente, per misurare l'accordo tra annotatori;
  - l'annotatore vede SOLO gli estratti di testo (niente ticker, anno,
    score o strato): annotazione cieca;
  - oltre ai contesti delle keyword del dizionario, vengono mostrati
    contesti di termini "di richiamo" non presenti nel dizionario (analytics,
    algorithm, automation, intelligent, data science...) per non vincolare
    la recall dell'annotatore al dizionario stesso;
  - stesso schema di etichette dell'annotazione umana (annotation_app.py);
  - le etichette sono salvate in data/processed/annotations_llm.json con
    annotator = LLM_ANNOTATOR e vanno dichiarate come tali nel paper.

Uso:
    python second_annotation.py --dump 0 20     # stampa gli estratti
    python second_annotation.py --save labels.json
"""
from __future__ import annotations

import argparse
import json
import re

import pandas as pd

import annotation_app as app
import config

SUB_CSV = config.PROCESSED_DIR / "annotation_subsample_llm.csv"
LLM_JSON = config.PROCESSED_DIR / "annotations_llm.json"
LLM_ANNOTATOR = "llm:claude-opus-5.5 (Anthropic)"
N_PER_STRATUM = 20
SEED = 7
MAX_KW_SNIPPETS = 6
MAX_RECALL_SNIPPETS = 2
WINDOW = 220
RECALL_TERMS = re.compile(
    r"\b(analytics|algorithm\w*|automat\w+|intelligent|data science|machine|"
    r"cognitive|robot\w*|digital transformation|smart)\b",
    re.IGNORECASE,
)


def build_subsample() -> pd.DataFrame:
    sample = pd.read_csv(app.SAMPLE_CSV)
    sub = (sample.groupby("stratum", group_keys=False)
           .apply(lambda g: g.sample(n=min(N_PER_STRATUM, len(g)), random_state=SEED))
           .sample(frac=1, random_state=SEED)  # ordine casuale: niente raggruppamento per strato
           .reset_index(drop=True))
    sub.to_csv(SUB_CSV, index=False)
    return sub


def _windows(text: str, pattern: re.Pattern, k: int) -> list[str]:
    spans = [(max(0, m.start() - WINDOW), min(len(text), m.end() + WINDOW))
             for m in pattern.finditer(text)]
    if not spans:
        return []
    step = max(1, len(spans) // k)
    return [text[s:e] for s, e in spans[::step][:k]]


def dump(start: int, end: int) -> None:
    sub = pd.read_csv(SUB_CSV) if SUB_CSV.exists() else build_subsample()
    for i in range(start, min(end, len(sub))):
        r = sub.iloc[i]
        text, kw_snips = app.extract_snippets(r, window=WINDOW, max_snip=200)
        if text is None:
            print(f"\n===== DOC {i} =====\n[FILE MANCANTE]")
            continue
        step = max(1, len(kw_snips) // MAX_KW_SNIPPETS)
        chosen = kw_snips[::step][:MAX_KW_SNIPPETS]
        recall = _windows(text, RECALL_TERMS, MAX_RECALL_SNIPPETS)
        body = "\n---\n".join(chosen) if chosen else "[nessuna keyword del dizionario]"
        body += "\n--- (termini di richiamo) ---\n" + ("\n---\n".join(recall) or "[nessuno]")
        print(f"\n===== DOC {i} =====\n{body}")


def save(path: str) -> None:
    """labels: {"<indice>": {"ai_mention": ..., "ai_substance": ..., "notes": ...}}"""
    sub = pd.read_csv(SUB_CSV)
    labels = json.loads(open(path, encoding="utf-8").read())
    ann = json.loads(LLM_JSON.read_text(encoding="utf-8")) if LLM_JSON.exists() else {}
    for k, v in labels.items():
        r = sub.iloc[int(k)]
        ann[r["accession"]] = {
            "ticker": r["ticker"], "fiscal_year": int(r["fiscal_year"]),
            "ai_mention": v["ai_mention"], "ai_substance": v.get("ai_substance", ""),
            "notes": v.get("notes", ""), "annotator": LLM_ANNOTATOR,
        }
    LLM_JSON.write_text(json.dumps(ann, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"[LLM] {len(labels)} etichette salvate ({len(ann)} totali) -> {LLM_JSON}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", nargs=2, type=int)
    ap.add_argument("--save")
    args = ap.parse_args()
    if args.dump:
        dump(*args.dump)
    if args.save:
        save(args.save)


if __name__ == "__main__":
    main()
