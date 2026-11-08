#!/usr/bin/env python3
"""
Baseline: spaCy NER. Default multilingual model 'xx_ent_wiki_sm'
(python -m spacy download xx_ent_wiki_sm); pass --model en_core_web_trf for the
English transformer. Emits scalar-offset spans to <exports>/spacy_{B,C,D}.jsonl.
ent.start_char/end_char are code-point offsets (== scalar).

    python3 run_spacy.py --exports exports --model xx_ent_wiki_sm --sets C D B
"""
import argparse, json, os

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exports", default="exports")
    ap.add_argument("--model", default="xx_ent_wiki_sm")
    ap.add_argument("--sets", nargs="+", default=["C", "D", "B"])
    a = ap.parse_args()
    import spacy
    nlp = spacy.load(a.model)
    m = {"B": "docs_B.jsonl", "C": "docs_C.jsonl", "D": "docs_D.jsonl", "AMB": "docs_AMB.jsonl"}
    for s in a.sets:
        path = os.path.join(a.exports, m[s])
        if not os.path.exists(path):
            print(f"skip {path}"); continue
        with open(path, encoding="utf-8") as f:
            docs = [json.loads(l) for l in f if l.strip()]
        rows = []
        for d in docs:
            did = d.get("doc_id") or d.get("id")
            for ent in nlp(d["text"]).ents:                 # PERSON/GPE/LOC/ORG...
                rows.append(json.dumps({"doc_id": did, "type": ent.label_,
                                        "start": ent.start_char, "end": ent.end_char},
                                       ensure_ascii=False))
        out = os.path.join(a.exports, f"spacy_{s}.jsonl")
        open(out, "w", encoding="utf-8").write("\n".join(rows) + ("\n" if rows else ""))
        print(f"{out}: {len(rows)} spans")

if __name__ == "__main__":
    main()
