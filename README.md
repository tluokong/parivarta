# Parivarta

**An on-device zero-trust pseudonymization proxy for privacy-preserving use of cloud LLMs with Thai text.**

Parivarta is a native macOS app that sits between the user and a cloud LLM. Before a prompt leaves the
device, a hybrid detection ensemble finds personally identifiable information (PII) in Thai and English
text, and each value is replaced by a reversible, referentially consistent surrogate (e.g. `PERSON_001`).
The original values are kept only in an encrypted local vault, and the model's reply is restored on the
device. The provider never sees the original PII or the mapping.

This repository contains the source code, the evaluation datasets, every system's prediction exports, and
the scorer used in the paper, so that all tables can be regenerated from a clean checkout.

> Paper: *Parivarta: An On-Device Zero-Trust Pseudonymization Proxy for Privacy-Preserving Use of Cloud
> LLMs with Thai Text*, KICSS 2026. See [`CITATION.cff`](CITATION.cff).

## Repository layout

| Path | Contents |
|------|----------|
| `Parivarta/` | macOS app (Swift / SwiftUI / Core ML): pre-filter, detection ensemble (L1–L6), span fusion, pseudonymization proxy, encrypted mapping vault, LLM client |
| `ParivartaTests/` | Unit tests and the evaluation harnesses (`BenchmarkEvaluationTests`, `NegativeEvaluationTests`, `RobustnessEvaluationTests`, `ThaiNEREvaluationTests`, `ExportDetectionsTests`); `Fixtures/` holds Dataset B and Dataset AMB |
| `tools/convert_thai_ner.py` | Builds the Core ML Thai NER model (member L5) — required before building the app |
| `tools/eval/` | Cross-system comparison: baseline runners, unified scorer `score.py`, datasets and prediction exports |
| `EXPERIMENTS.md` | Full experimental log (datasets, metrics, ablation E0–E8, baselines, Dataset AMB) |
| `METHODOLOGY.md` | System design and threat model |

## Reproducing the paper's results (no build required)

The scorer is standard-library Python (3.8+) and needs no network access.

```bash
cd tools/eval
python3 score.py --exports exports --system ours          # Parivarta
python3 score.py --exports exports --system presidio      # likewise: regex, spacy, pythainlp
python3 score.py --exports exports --system ours --vs presidio regex spacy pythainlp   # paired McNemar on Dataset B
```

Expected output for `ours`: Dataset B coverage / typed / exact 100.0%, Dataset C specificity 94.3%,
Dataset D coverage 95.5%, Dataset AMB POS-recall 74.1%, NEG-specificity 21.8%, pair-accuracy 8.6%.
Each line carries a bootstrap 95% confidence interval (10,000 resamples).

The zero-shot LLM baseline (`qwen2.5:7b` via a local Ollama server) was run on a separate machine;
only its Dataset C and D exports are included here. To regenerate it:

```bash
ollama pull qwen2.5:7b
python3 run_llm.py --exports exports --provider ollama --model qwen2.5:7b --tag llm --sets B C D --resume
python3 score.py --exports exports --system llm
```

## Re-running the baselines

```bash
cd tools/eval
pip install -r requirements.txt
python -m spacy download xx_ent_wiki_sm
python3 run_regex.py    --exports exports
python3 run_spacy.py    --exports exports --sets B C D AMB
python3 run_pythainlp.py --exports exports --sets B C D AMB
python3 run_presidio.py --exports exports --sets B C D AMB
```

Versions used in the paper: presidio-analyzer 2.2.360, spaCy 3.7.5 with `xx_ent_wiki_sm` 3.7.0, and
PyThaiNLP 5.3.4 (CRF-based `thainer` engine).

## Building the app

Requirements: macOS and Xcode (the project targets macOS 26.5). Swift package dependencies, including
[swift-transformers](https://github.com/huggingface/swift-transformers), are resolved by Xcode.

1. **Generate the Thai NER model.** The Core ML weights (~210 MB) exceed GitHub's file-size limit and are
   not committed. The script downloads `pythainlp/thainer-corpus-v2-base-model` from the Hugging Face Hub,
   converts it, and writes `ThaiNER.mlpackage` plus the tokenizer and label files into `Parivarta/`:

   ```bash
   python3 -m venv tools/venv
   tools/venv/bin/pip install -r tools/requirements.txt
   tools/venv/bin/python tools/convert_thai_ner.py
   ```

2. **Open `Parivarta.xcodeproj`**, select your own signing team under *Signing & Capabilities*
   (the team ID has been removed from the project), then build and run.

3. **Run the evaluation harnesses** from Xcode's Test navigator. `ExportDetectionsTests` writes the
   Parivarta prediction exports (`ours_*.jsonl`) that `tools/eval/score.py` consumes.

## Datasets

All PII in every dataset is synthetic; no real personal data is included.

| Dataset | Files | Description |
|---------|-------|-------------|
| B | `tools/eval/exports/docs_B.jsonl`, `gold_B.jsonl` (also `ParivartaTests/Fixtures/GEN_Entity.jsonl`, `Goal_Entity.jsonl`) | 2,500 Thai documents, one target entity each, 500 per type: PERSON, PHONE, EMAIL, ADDRESS, NATIONAL_ID |
| C | `tools/eval/exports/docs_C.jsonl` | 35 hard-negative sentences that contain no PII |
| D | `tools/eval/exports/docs_D.jsonl`, `gold_D.jsonl` | Hard positives (non-template addresses, multiple and nested entities, romanized names) |
| AMB | `tools/eval/exports/docs_AMB.jsonl`, `gold_AMB.jsonl` | 1,250 minimal pairs: the same surface form as PII and as a non-PII look-alike |

Offsets are Unicode-scalar (Python code-point) indices. Every system's predictions are stored as
span-JSONL, `{"doc_id", "type", "start", "end"}`, in `tools/eval/exports/<system>_<set>.jsonl`.

## Security notes

Never commit API keys. `run_llm.py` reads provider keys from environment variables or from a
`--key-file` that is excluded by `.gitignore`. The LLM baseline sends text to the chosen provider and is
intended only for the synthetic evaluation data.

## License

Code and data are released under the [Apache License 2.0](LICENSE). See [`NOTICE`](NOTICE) for
third-party material (PyThaiNLP word lists and the separately downloaded Thai NER model).

## Acknowledgment

This work was supported by the National Cyber Security Agency (NCSA), Thailand, with research facilities
and support from Rajamangala University of Technology Lanna (RMUTL).
