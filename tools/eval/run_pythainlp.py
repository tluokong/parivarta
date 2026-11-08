#!/usr/bin/env python3
"""
Baseline: PyThaiNLP neural NER (thainer). Emits scalar-offset spans to
<exports>/pythainlp_{B,C,D}.jsonl for score.py.

Run with the project venv (pythainlp installed):
    tools/venv/bin/python tools/eval/run_pythainlp.py --exports exports [--sets C D]

thainer entity labels are folded to the comparison vocabulary by score.py's
canonical map (PERSON/LOCATION/ORGANIZATION/PHONE/EMAIL/…).
"""
import argparse, json, os

def load_ner():
    from pythainlp.tag import NER
    try:
        return NER("thainer")
    except Exception:
        return NER(engine="thainer")

def bio_spans(text, tagged):
    """tagged: list of (word, netag) → merged char spans (start,end,type)."""
    spans, cur = [], 0
    cs = ce = ctype = None
    def flush():
        nonlocal cs, ce, ctype
        if cs is not None:
            spans.append((ctype, cs, ce))
        cs = ce = ctype = None
    for item in tagged:
        word = item[0]
        netag = item[-1] if len(item) > 1 else "O"
        idx = text.find(word, cur)
        if idx < 0:
            idx = cur
        start, end = idx, idx + len(word)
        cur = end
        if netag and netag != "O":
            prefix, _, typ = netag.partition("-")
            typ = typ or prefix
            if prefix == "B" or ctype != typ:
                flush(); cs, ce, ctype = start, end, typ
            else:
                ce = end
        else:
            flush()
    flush()
    return spans

def run(ner, docs_path, out_path):
    if not os.path.exists(docs_path):
        print(f"skip {docs_path} (missing)"); return
    with open(docs_path, encoding="utf-8") as f:
        docs = [json.loads(l) for l in f if l.strip()]
    rows = []
    for i, d in enumerate(docs):
        did = d.get("doc_id") or d.get("id")
        try:
            tagged = ner.tag(d["text"])
        except Exception as e:
            print(f"  ! {did}: {e}"); continue
        for typ, a, b in bio_spans(d["text"], tagged):
            rows.append(json.dumps({"doc_id": did, "type": typ,
                                    "start": a, "end": b}, ensure_ascii=False))
        if (i + 1) % 250 == 0:
            print(f"  {os.path.basename(docs_path)}: {i+1}/{len(docs)}")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(rows) + ("\n" if rows else ""))
    print(f"{out_path}: {len(rows)} spans")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exports", default="exports")
    ap.add_argument("--sets", nargs="+", default=["C", "D", "B"])
    a = ap.parse_args()
    ner = load_ner()
    m = {"B": "docs_B.jsonl", "C": "docs_C.jsonl", "D": "docs_D.jsonl", "AMB": "docs_AMB.jsonl"}
    for s in a.sets:
        run(ner, os.path.join(a.exports, m[s]),
            os.path.join(a.exports, f"pythainlp_{s}.jsonl"))

if __name__ == "__main__":
    main()
