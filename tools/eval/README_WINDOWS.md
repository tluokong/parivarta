# Running the full Dataset B LLM baseline on Windows (qwen2.5:3b via Ollama)

The macOS run of the full 2,500-doc LLM baseline is slow (~3 s/doc on an M2 Air,
~2 h total). Run it on a faster Windows machine instead. Everything here is
**cross-platform stdlib Python** (no venv needed) and **resume-safe**.

## What to copy to the Windows machine

Copy the whole `tools/eval/` folder. The only inputs it needs are already inside
`tools/eval/exports/`:

- `docs_B.jsonl`  — the 2,500 Dataset-B documents (input)
- `gold_B.jsonl`  — the gold spans (for scoring)

(Everything else — `run_llm.py`, `score.py`, `run_B.ps1` — is code.)

## Prerequisites

1. **Python 3.8+** — https://www.python.org/downloads/ (tick "Add python.exe to PATH").
   Verify: `python --version`
2. **Ollama for Windows** — https://ollama.com/download/windows.
   After install it runs a local server on `http://localhost:11434`.
   Verify: `ollama --version`

## Run (PowerShell)

```powershell
cd path\to\tools\eval
powershell -ExecutionPolicy Bypass -File .\run_B.ps1
```

`run_B.ps1` pulls `qwen2.5:3b`, runs all 2,500 docs, and **auto-resumes** if it
is interrupted (progress is tracked in `exports\qwen_B.done`; already-processed
docs are skipped). It prints progress every 50 docs and scores at the end.

### Manual equivalent (cmd or PowerShell)

```powershell
ollama pull qwen2.5:3b
python run_llm.py --exports exports --provider ollama --model qwen2.5:3b --tag qwen --sets B --resume
python score.py   --exports exports --system qwen --latex
```

If it stops for any reason, **just run the same `run_llm.py` line again** — it
continues where it left off.

## Send back

- `exports\qwen_B.jsonl`  (the predictions)
- the `score.py` output (B coverage / exact-span, with 95% CIs, and the LaTeX row)

I'll merge it into the paper's baseline table as the LLM (local, qwen2.5:3b)
row on the full 2,500-doc Dataset B.

## Notes

- Output/system name is `qwen` (files `qwen_B.jsonl`, `qwen_B.done`).
- Record the **model + date** for the paper (e.g. `qwen2.5:3b`, run on Windows,
  YYYY-MM-DD).
- A bigger/faster model is fine if the machine has the RAM:
  `.\run_B.ps1 -Model qwen2.5:7b` (or `-Model llama3.1:8b`).
- This LLM baseline sends text to a local model only (no cloud); it is a
  comparison point, not part of the privacy-preserving system.
