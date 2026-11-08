#!/usr/bin/env python3
"""
Baseline: Microsoft Presidio (presidio-analyzer). Out of the box Presidio is
English-centric; we run its default analyzer and additionally register a few
Thai-format recognizers (Thai national ID, Thai phone) so the comparison is
fair on formatted PII. Emits scalar-offset spans to
<exports>/presidio_{B,C,D}.jsonl (result.start/end are code-point offsets).

    python3 run_presidio.py --exports exports --sets C D B
"""
import argparse, json, os

def build_analyzer():
    from presidio_analyzer import AnalyzerEngine, PatternRecognizer, Pattern
    engine = AnalyzerEngine()
    thai_id = PatternRecognizer(
        supported_entity="TH_NATIONAL_ID",
        patterns=[Pattern("th_id", r"\b\d[-\s]?\d{4}[-\s]?\d{5}[-\s]?\d{2}[-\s]?\d\b", 0.6)])
    thai_phone = PatternRecognizer(
        supported_entity="TH_PHONE",
        patterns=[Pattern("th_phone", r"(?:\+?66|0)[\s-]?\d{1,2}[\s-]?\d{3,4}[\s-]?\d{3,4}", 0.4)])
    engine.registry.add_recognizer(thai_id)
    engine.registry.add_recognizer(thai_phone)
    return engine

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exports", default="exports")
    ap.add_argument("--sets", nargs="+", default=["C", "D", "B"])
    ap.add_argument("--language", default="en")
    a = ap.parse_args()
    engine = build_analyzer()
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
            for r in engine.analyze(text=d["text"], language=a.language):
                rows.append(json.dumps({"doc_id": did, "type": r.entity_type,
                                        "start": r.start, "end": r.end},
                                       ensure_ascii=False))
        out = os.path.join(a.exports, f"presidio_{s}.jsonl")
        open(out, "w", encoding="utf-8").write("\n".join(rows) + ("\n" if rows else ""))
        print(f"{out}: {len(rows)} spans")

if __name__ == "__main__":
    main()
