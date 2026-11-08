# Comparative evaluation harness (M1 / B1)

Uniformly compares **Parivarta** against baseline PII detectors on the same
datasets and metrics. Because Parivarta is a Swift/Core ML app and the
baselines are Python, comparison happens at the **detector-output** level: each
system emits one JSON-Lines file of spans; a single Python scorer computes all
metrics.

## Common output format

One record per detected span:

```json
{"doc_id": "GEN-PERSON-000001", "type": "PERSON", "start": 39, "end": 56}
```

`start`/`end` are **Unicode scalar** offsets into the document text (matches the
Python code-point offsets used by the gold data — see `EXPERIMENTS.md`).

## Systems to compare

| System | How to run | Notes |
|--------|------------|-------|
| Parivarta (ours) | export CLI from the app / a test target that dumps detections | needs a small `--export` harness (M0 follow-up) |
| Microsoft Presidio | `presidio-analyzer` | English recognizers; add custom Thai regex recognizers for parity |
| spaCy | `en_core_web_trf` | English/code-mixed NER |
| PyThaiNLP NER | `pythainlp.tag.named_entity` | Thai NER (thainer) |
| LLM zero-shot | Claude / GPT-4o-mini via API | fixed PII-extraction prompt; report cost/latency |
| regex-only | Parivarta L1 only | lower bound |

## Datasets (inputs)

- B: `Parivarta/ParivartaTests/Fixtures/{GEN_Entity,Goal_Entity}.jsonl`
- C: negatives (export from `NegativeEvaluationTests`)
- D: robustness probes (export from `RobustnessEvaluationTests`)
- E: independent real-carrier set (M2 — to be created here)

## Scorer

`score.py` (to be written): loads gold + each system's spans, computes
coverage/leak, typed coverage, exact-span, specificity, over-mask; bootstrap
95% CI; McNemar paired test vs. ours per document; emits the LaTeX tables for
`paper/sections/results.tex`.

## Layout (planned)

```
tools/eval/
  README.md            # this file
  requirements.txt     # presidio-analyzer, spacy, pythainlp, anthropic, numpy, scipy
  export_gold.py       # normalize datasets to a common gold format
  run_presidio.py  run_spacy.py  run_pythainlp.py  run_llm.py  run_regex.py
  score.py             # unified metrics + CIs + significance + LaTeX emit
```

## Reproduce

```bash
# 1. Export Parivarta detections (Swift) — writes to the app container, then:
#    RunSomeTests ExportDetections/exportForBaselineComparison, then
#    cp "<container>/…/Parivarta/exports/"*.jsonl tools/eval/exports/
#    (+ copy GEN_Entity→docs_B.jsonl, Goal_Entity→gold_B.jsonl)

# 2. Baselines (each writes <system>_{B,C,D}.jsonl into exports/)
python3            tools/eval/run_regex.py     --exports tools/eval/exports
tools/venv/bin/python tools/eval/run_pythainlp.py --exports tools/eval/exports
python3            tools/eval/run_spacy.py     --exports tools/eval/exports    # needs spaCy model
python3            tools/eval/run_presidio.py  --exports tools/eval/exports    # needs presidio
# provider-agnostic LLM baseline: {ollama|openai|openrouter|gemini|anthropic};
# ollama = local & free (no key). Record provider+model+date.
tools/venv/bin/python tools/eval/run_llm.py --exports tools/eval/exports \
    --provider ollama --model qwen2.5:7b --sets C D B

# 3. Score any system (bootstrap 95% CIs; --latex for a paper row)
python3 tools/eval/score.py --exports tools/eval/exports --system ours --latex
```

## Results

Final results for all systems (including the zero-shot LLM, `qwen2.5:7b` via local Ollama) are
reported in `EXPERIMENTS.md` §5.10–§5.11 and reproduced by `score.py` from the exports in this folder.
