#!/usr/bin/env python3
"""
Baseline: a naive generic-regex PII detector (email / Thai phone / bare 13-digit
ID / IPv4). No person names, no addresses, no gazetteer — the "what a
practitioner writes in an afternoon" lower bound.

Emits scalar-offset spans (Python str indices == code points) to
<exports>/regex_{B,C,D}.jsonl, ready for score.py.

Usage: python3 run_regex.py --exports exports
"""
import argparse, json, os, re

PATTERNS = [
    ("EMAIL",       re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")),
    ("NATIONAL_ID", re.compile(r"\b\d{13}\b")),
    ("PHONE",       re.compile(r"\b0\d[-\s]?\d{3}[-\s]?\d{3,4}\b")),
    ("IP",          re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")),
]

def read_docs(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]

def detect(text):
    out = []
    for typ, rx in PATTERNS:
        for m in rx.finditer(text):
            out.append((typ, m.start(), m.end()))
    return out

def run(docs_path, out_path):
    if not os.path.exists(docs_path):
        print(f"skip {docs_path} (missing)"); return
    rows = []
    for d in read_docs(docs_path):
        did = d.get("doc_id") or d.get("id")     # GEN uses "id"; C/D use "doc_id"
        for typ, a, b in detect(d["text"]):
            rows.append(json.dumps({"doc_id": did, "type": typ,
                                    "start": a, "end": b}, ensure_ascii=False))
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(rows) + ("\n" if rows else ""))
    print(f"{out_path}: {len(rows)} spans")

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--exports", default="exports")
    a = ap.parse_args()
    run(os.path.join(a.exports, "docs_B.jsonl"), os.path.join(a.exports, "regex_B.jsonl"))
    run(os.path.join(a.exports, "docs_C.jsonl"), os.path.join(a.exports, "regex_C.jsonl"))
    run(os.path.join(a.exports, "docs_D.jsonl"), os.path.join(a.exports, "regex_D.jsonl"))
    run(os.path.join(a.exports, "docs_AMB.jsonl"), os.path.join(a.exports, "regex_AMB.jsonl"))

if __name__ == "__main__":
    main()
