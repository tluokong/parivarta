#!/usr/bin/env python3
"""
Unified scorer for the M1 baseline comparison (Parivarta, Computers & Security).

Scores any system's predicted spans against the gold, uniformly, on:
  - Dataset B (2500 docs, 1 target/doc): coverage / typed / exact-span
  - Dataset C (35 hard-negatives):       specificity (1 - FP-doc rate)
  - Dataset D (hard positives):          coverage (any-type containment)

All offsets are Unicode scalar (== Python str code-point) indices, so gold and
predictions are directly comparable (see EXPERIMENTS.md).

Predicted spans for system S are read from  <exports>/S_{B,C,D}.jsonl
Gold/docs are read from                     <exports>/{gold_B,gold_D,docs_C}.jsonl

Stdlib only. Usage:
    python3 score.py --exports exports --system ours [--latex]
    python3 score.py --exports exports --system presidio
"""
import argparse, json, math, os, random
from collections import defaultdict

# Canonical type map: fold each system's / gold's labels into one vocabulary.
CANON = {
    "PERSON": "PERSON", "PER": "PERSON",
    "PHONE": "PHONE", "PHONE_NUMBER": "PHONE", "PHONE_NUMBER_TH": "PHONE", "TH_PHONE": "PHONE",
    "TH_NATIONAL_ID": "NATIONAL_ID",
    "EMAIL": "EMAIL",
    "NATIONAL_ID": "NATIONAL_ID", "THAI_ID": "NATIONAL_ID", "ID": "NATIONAL_ID",
    "ADDRESS": "ADDRESS",
    "IP_ADDRESS": "IP", "IP": "IP",
    "LOCATION": "LOCATION", "LOC": "LOCATION", "GPE": "LOCATION",
    "ORGANIZATION": "ORG", "ORG": "ORG",
    "CUSTOM_KEYWORD": "KEYWORD",
}
def canon(t): return CANON.get(t, t)

def read_jsonl(path):
    if not os.path.exists(path): return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]

def spans_by_doc(rows):
    d = defaultdict(list)
    for r in rows:
        d[r["doc_id"]].append((canon(r["type"]), int(r["start"]), int(r["end"])))
    return d

def masked_set(spans):
    m = set()
    for _, a, b in spans: m.update(range(a, b))
    return m

# ---- metric helpers (per-document booleans, so we can bootstrap over docs) ----

def pct(x): return f"{100*x:5.1f}%"

def bootstrap_ci(items, stat, n=10000, seed=0):
    """items: list of per-doc records; stat: list->float in [0,1]. Returns (lo,hi)."""
    if not items: return (0.0, 0.0)
    rnd = random.Random(seed)
    N = len(items)
    vals = []
    for _ in range(n):
        sample = [items[rnd.randrange(N)] for _ in range(N)]
        vals.append(stat(sample))
    vals.sort()
    return (vals[int(0.025*n)], vals[int(0.975*n)])

# ------------------------------ Dataset B -------------------------------------

def score_B(exports, system, only=None):
    gold = read_jsonl(os.path.join(exports, "gold_B.jsonl"))
    if only:
        ids = {(r.get("doc_id") or r.get("id")) for r in read_jsonl(only)}
        gold = [g for g in gold if g["id"] in ids]
    pred = spans_by_doc(read_jsonl(os.path.join(exports, f"{system}_B.jsonl")))
    per = []   # per gold mention (== per doc, 1 target/doc)
    for g in gold:
        did = g["id"]; gt = canon(g["target_entity_type"])
        gs, ge = int(g["annotation_start"]), int(g["annotation_end"])
        sp = pred.get(did, [])
        m = masked_set(sp)
        covered = all(i in m for i in range(gs, ge))
        typed = any(t == gt and a <= gs and b >= ge for (t, a, b) in sp)
        exact = any(t == gt and (a, b) == (gs, ge) for (t, a, b) in sp)
        per.append((gt, covered, typed, exact))
    n = len(per)
    cov = sum(p[1] for p in per) / n
    typ = sum(p[2] for p in per) / n
    exa = sum(p[3] for p in per) / n
    ci = bootstrap_ci(per, lambda s: sum(p[1] for p in s)/len(s))
    # per-type coverage
    bytype = defaultdict(lambda: [0, 0])
    for gt, c, _, _ in per:
        bytype[gt][0] += c; bytype[gt][1] += 1
    return dict(n=n, coverage=cov, typed=typ, exact=exa, cov_ci=ci,
                bytype={k: v[0]/v[1] for k, v in bytype.items()})

# --------------------- paired significance (McNemar, Dataset B) ---------------

def doc_coverage_B(exports, system):
    """Per-document coverage decision (doc_id -> bool) on Dataset B."""
    gold = read_jsonl(os.path.join(exports, "gold_B.jsonl"))
    pred = spans_by_doc(read_jsonl(os.path.join(exports, f"{system}_B.jsonl")))
    out = {}
    for g in gold:
        did = g["id"]; gs, ge = int(g["annotation_start"]), int(g["annotation_end"])
        m = masked_set(pred.get(did, []))
        out[did] = all(i in m for i in range(gs, ge))
    return out

def mcnemar(a, b):
    """a,b: dict doc_id->bool. Returns (b01, b10, p) where b01 = a right & b wrong."""
    ids = set(a) & set(b)
    b01 = sum(1 for i in ids if a[i] and not b[i])
    b10 = sum(1 for i in ids if b[i] and not a[i])
    n = b01 + b10
    if n == 0:
        return b01, b10, 1.0
    if n < 25:  # exact two-sided binomial (p=0.5)
        k = min(b01, b10)
        p = sum(math.comb(n, i) for i in range(k + 1)) * (0.5 ** n) * 2
        return b01, b10, min(1.0, p)
    chi2 = (abs(b01 - b10) - 1) ** 2 / n   # continuity-corrected
    p = math.erfc(math.sqrt(chi2 / 2))      # df=1
    return b01, b10, p

# ------------------------------ Dataset C -------------------------------------

def score_C(exports, system):
    docs = read_jsonl(os.path.join(exports, "docs_C.jsonl"))
    ids = [d["doc_id"] for d in docs]
    pred = spans_by_doc(read_jsonl(os.path.join(exports, f"{system}_C.jsonl")))
    per = [(1 if pred.get(i) else 0) for i in ids]      # 1 = has a false positive
    n = len(per); fp = sum(per)
    spec = 1 - fp / n
    ci = bootstrap_ci(per, lambda s: 1 - sum(s)/len(s))
    return dict(n=n, fp_docs=fp, specificity=spec, spec_ci=ci)

# ------------------------------ Dataset D -------------------------------------

def score_D(exports, system):
    gold = read_jsonl(os.path.join(exports, "gold_D.jsonl"))
    pred = spans_by_doc(read_jsonl(os.path.join(exports, f"{system}_D.jsonl")))
    # group gold mentions by doc; coverage is any-type containment
    per = []
    by_doc = defaultdict(list)
    for g in gold: by_doc[g["doc_id"]].append((int(g["start"]), int(g["end"])))
    for did, mentions in by_doc.items():
        m = masked_set(pred.get(did, []))
        for gs, ge in mentions:
            per.append(all(i in m for i in range(gs, ge)))
    n = len(per); cov = sum(per) / n if n else 0
    ci = bootstrap_ci(per, lambda s: sum(s)/len(s))
    return dict(n=n, coverage=cov, cov_ci=ci)

# ------------------------------ Dataset AMB -----------------------------------

def score_AMB(exports, system):
    """Minimal-pair ambiguity suite. Each judgment is on the shared candidate span:
       POS correct  = candidate span fully masked (detection sensitivity, TPR);
       NEG correct  = no prediction overlaps the candidate span (1 - FPR);
       pair correct = POS and NEG (same pair_id) both correct."""
    gold = read_jsonl(os.path.join(exports, "gold_AMB.jsonl"))
    pred = spans_by_doc(read_jsonl(os.path.join(exports, f"{system}_AMB.jsonl")))
    pos, neg = [], []                         # per-member correctness (label-agnostic)
    pairs = defaultdict(dict)                 # pair_id -> {"POS": bool, "NEG": bool}
    by_type = defaultdict(list)               # type -> [correct bool] over POS+NEG members
    for g in gold:
        did = g["id"]; typ = canon(g["target_entity_type"])
        cs, ce = int(g["candidate_start"]), int(g["candidate_end"])
        sp = pred.get(did, [])
        if g["subset"] == "ambiguous_positive":
            ok = all(i in masked_set(sp) for i in range(cs, ce))      # candidate covered
            pos.append(ok); pairs[g["pair_id"]]["POS"] = ok
        else:                                                          # hard_negative
            ok = not any(a < ce and b > cs for (_, a, b) in sp)        # no overlap
            neg.append(ok); pairs[g["pair_id"]]["NEG"] = ok
        by_type[typ].append(ok)
    pair_items = [p["POS"] and p["NEG"] for p in pairs.values() if "POS" in p and "NEG" in p]
    mean = lambda s: (sum(s) / len(s)) if s else 0.0
    return dict(n_pairs=len(pair_items), n_pos=len(pos), n_neg=len(neg),
                pos_recall=mean(pos), pos_ci=bootstrap_ci(pos, mean),
                neg_spec=mean(neg), neg_ci=bootstrap_ci(neg, mean),
                pair_acc=mean(pair_items), pair_ci=bootstrap_ci(pair_items, mean),
                indep=mean(pos) * mean(neg),
                bytype={k: sum(v) / len(v) for k, v in by_type.items()})

# ---------------------------------- main --------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exports", default="exports")
    ap.add_argument("--system", required=True)
    ap.add_argument("--latex", action="store_true")
    ap.add_argument("--vs", nargs="+", default=[],
                    help="baselines to McNemar-test against --system on Dataset B coverage")
    ap.add_argument("--only", default="",
                    help="restrict Dataset B to doc_ids in this docs/ids file (e.g. docs_Bsample.jsonl)")
    a = ap.parse_args()

    if a.vs:
        ours = doc_coverage_B(a.exports, a.system)
        print(f"\n===== McNemar (Dataset B coverage): {a.system} vs. baseline =====")
        print("baseline      ours-only  base-only   p-value")
        for base in a.vs:
            other = doc_coverage_B(a.exports, base)
            b01, b10, p = mcnemar(ours, other)
            star = "***" if p < 1e-3 else "**" if p < 1e-2 else "*" if p < 0.05 else ""
            print(f"{base:<12}  {b01:>7}   {b10:>8}   {p:>9.2e} {star}")
        return

    B = score_B(a.exports, a.system, only=(a.only or None))
    C = score_C(a.exports, a.system)
    D = score_D(a.exports, a.system)

    print(f"\n===== System: {a.system} =====")
    print(f"B (n={B['n']}): coverage {pct(B['coverage'])} "
          f"[{pct(B['cov_ci'][0])},{pct(B['cov_ci'][1])}]  "
          f"typed {pct(B['typed'])}  exact {pct(B['exact'])}")
    print("   per-type coverage: " +
          ", ".join(f"{k} {pct(v)}" for k, v in sorted(B["bytype"].items())))
    print(f"C (n={C['n']}): specificity {pct(C['specificity'])} "
          f"[{pct(C['spec_ci'][0])},{pct(C['spec_ci'][1])}]  (FP docs={C['fp_docs']})")
    print(f"D (n={D['n']}): coverage {pct(D['coverage'])} "
          f"[{pct(D['cov_ci'][0])},{pct(D['cov_ci'][1])}]")

    if os.path.exists(os.path.join(a.exports, "gold_AMB.jsonl")) and \
       os.path.exists(os.path.join(a.exports, f"{a.system}_AMB.jsonl")):
        A = score_AMB(a.exports, a.system)
        print(f"AMB (n={A['n_pairs']} pairs): "
              f"POS-recall {pct(A['pos_recall'])} [{pct(A['pos_ci'][0])},{pct(A['pos_ci'][1])}]  "
              f"NEG-specificity {pct(A['neg_spec'])} [{pct(A['neg_ci'][0])},{pct(A['neg_ci'][1])}]")
        print(f"   pair-accuracy {pct(A['pair_acc'])} [{pct(A['pair_ci'][0])},{pct(A['pair_ci'][1])}]  "
              f"(independence baseline {pct(A['indep'])})")
        print("   per-type pair-relevant accuracy: " +
              ", ".join(f"{k} {pct(v)}" for k, v in sorted(A["bytype"].items())))

    if a.latex:
        print("\n% LaTeX row (System & B-cov & B-exact & C-spec & D-cov)")
        print(f"{a.system} & {pct(B['coverage'])} & {pct(B['exact'])} & "
              f"{pct(C['specificity'])} & {pct(D['coverage'])} \\\\")

if __name__ == "__main__":
    main()
