#!/usr/bin/env python3
"""
Baseline: LLM zero-shot PII extraction — provider-agnostic (stdlib only, raw HTTP).

The model returns PII as exact substrings; we locate each in the source to recover
scalar offsets (the model never counts characters). Emits <exports>/llm_{B,C,D}.jsonl.

Providers (choose with --provider):
  ollama      local, free, no key    (OLLAMA_BASE_URL, default http://localhost:11434)
  openai      OPENAI_API_KEY
  openrouter  OPENROUTER_API_KEY
  gemini      GEMINI_API_KEY
  anthropic   ANTHROPIC_API_KEY

Examples:
  # local & free (install Ollama, then: ollama pull llama3.1)
  python3 run_llm.py --provider ollama --model llama3.1 --sets C D
  python3 run_llm.py --provider openai --model gpt-4o-mini --sets C D
  python3 run_llm.py --provider gemini --model gemini-2.5-flash --sets C D

Record provider+model+date in the paper. This baseline sends text to a provider,
so it is appropriate only for the synthetic evaluation data, and is reported as a
comparison point — not as part of the privacy-preserving system.
"""
import argparse, json, os, time, urllib.request, urllib.error

PROMPT = (
    "You extract personally identifiable information (PII) from Thai/English text. "
    "Return ONLY a JSON array of objects {\"type\": <PERSON|PHONE|EMAIL|ADDRESS|"
    "NATIONAL_ID|IP|LOCATION|ORG>, \"text\": <exact substring>}. "
    "Copy each value verbatim from the input. No commentary.\n\nText:\n"
)

def _post(url, payload, headers, retries=6):
    data = json.dumps(payload).encode("utf-8")
    for attempt in range(retries):
        req = urllib.request.Request(url, data=data,
                                     headers={"Content-Type": "application/json", **headers})
        try:
            with urllib.request.urlopen(req, timeout=180) as r:
                return json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            # rate-limit / transient server errors → exponential backoff
            if e.code in (429, 500, 502, 503, 504) and attempt < retries - 1:
                wait = min(60, 2 ** attempt)
                print(f"    [retry {attempt+1}] HTTP {e.code}, sleeping {wait}s", flush=True)
                time.sleep(wait); continue
            raise

def call(provider, model, prompt):
    """Return the model's raw text reply for a single prompt."""
    if provider in ("ollama", "openai", "openrouter"):
        if provider == "ollama":
            base = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
            url, key = f"{base}/v1/chat/completions", None
        elif provider == "openai":
            url, key = "https://api.openai.com/v1/chat/completions", os.environ["OPENAI_API_KEY"]
        else:
            url, key = "https://openrouter.ai/api/v1/chat/completions", os.environ["OPENROUTER_API_KEY"]
        headers = {"Authorization": f"Bearer {key}"} if key else {}
        data = _post(url, {"model": model, "messages": [{"role": "user", "content": prompt}]}, headers)
        return data["choices"][0]["message"]["content"]
    if provider == "gemini":
        key = os.environ["GEMINI_API_KEY"]
        url = (f"https://generativelanguage.googleapis.com/v1beta/models/{model}"
               f":generateContent?key={key}")
        data = _post(url, {"contents": [{"parts": [{"text": prompt}]}]}, {})
        parts = data["candidates"][0]["content"].get("parts", [{}])
        return "".join(p.get("text", "") for p in parts)
    if provider == "anthropic":
        data = _post("https://api.anthropic.com/v1/messages",
                     {"model": model, "max_tokens": 1024,
                      "messages": [{"role": "user", "content": prompt}]},
                     {"x-api-key": os.environ["ANTHROPIC_API_KEY"], "anthropic-version": "2023-06-01"})
        return "".join(b.get("text", "") for b in data.get("content", []) if b.get("type") == "text")
    raise SystemExit(f"unknown provider: {provider}")

def locate_all(text, sub):
    out, i = [], (text.find(sub) if sub else -1)
    while i >= 0:
        out.append((i, i + len(sub))); i = text.find(sub, i + 1)
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exports", default="exports")
    ap.add_argument("--provider", default="ollama",
                    choices=["ollama", "openai", "openrouter", "gemini", "anthropic"])
    ap.add_argument("--model", default="llama3.1")
    ap.add_argument("--sets", nargs="+", default=["C", "D"])
    ap.add_argument("--limit", type=int, default=0, help="docs per set after offset (0 = all)")
    ap.add_argument("--offset", type=int, default=0, help="start index within the set")
    ap.add_argument("--append", action="store_true", help="append to <tag>_<set>.jsonl (for chunked runs)")
    ap.add_argument("--tag", default="llm", help="output/system name → <tag>_<set>.jsonl (e.g. gemini)")
    ap.add_argument("--sleep", type=float, default=0.0, help="seconds to wait between requests (respect RPM)")
    ap.add_argument("--key-file", default="", help="read the provider API key from this file (gitignored)")
    ap.add_argument("--docs", default="", help="explicit docs file (overrides the set's default; use one --set)")
    ap.add_argument("--resume", action="store_true",
                    help="incremental + interrupt-safe: skip docs already in <tag>_<set>.done, append the rest")
    a = ap.parse_args()

    if a.key_file and os.path.exists(a.key_file):
        env = {"gemini": "GEMINI_API_KEY", "openai": "OPENAI_API_KEY",
               "openrouter": "OPENROUTER_API_KEY", "anthropic": "ANTHROPIC_API_KEY"}.get(a.provider)
        if env:
            os.environ[env] = open(a.key_file).read().strip()

    m = {"B": "docs_B.jsonl", "C": "docs_C.jsonl", "D": "docs_D.jsonl"}
    for s in a.sets:
        path = a.docs or os.path.join(a.exports, m[s])
        if not os.path.exists(path):
            print(f"skip {path}"); continue
        with open(path, encoding="utf-8") as f:
            docs = [json.loads(l) for l in f if l.strip()]
        docs = docs[a.offset:]
        if a.limit:
            docs = docs[:a.limit]

        out = os.path.join(a.exports, f"{a.tag}_{s}.jsonl")
        done_path = os.path.join(a.exports, f"{a.tag}_{s}.done")
        done = set()
        if a.resume and os.path.exists(done_path):
            done = set(open(done_path, encoding="utf-8").read().split())
        elif not a.append:
            open(out, "w").close(); open(done_path, "w").close()   # fresh start

        # incremental + interrupt-safe: write each doc's spans and its id immediately
        of = open(out, "a", encoding="utf-8")
        df = open(done_path, "a", encoding="utf-8")
        written = 0
        for i, d in enumerate(docs):
            did = d.get("doc_id") or d.get("id")
            if did in done:
                continue
            try:
                raw = call(a.provider, a.model, PROMPT + d["text"]).strip()
                items = json.loads(raw[raw.find("["): raw.rfind("]") + 1])
            except (urllib.error.URLError, KeyError, ValueError, json.JSONDecodeError) as e:
                print(f"  ! {did}: {e}"); items = []
            for it in items if isinstance(items, list) else []:
                for a0, b0 in locate_all(d["text"], it.get("text", "")):
                    of.write(json.dumps({"doc_id": did, "type": it.get("type", "PII"),
                                         "start": a0, "end": b0}, ensure_ascii=False) + "\n")
                    written += 1
            of.flush()
            df.write(did + "\n"); df.flush()      # mark done (even if 0 spans)
            done.add(did)
            if a.sleep:
                time.sleep(a.sleep)
            if (i + 1) % 50 == 0:
                print(f"  {s}: {a.offset + i + 1}/{a.offset + len(docs)}  done={len(done)}", flush=True)
        of.close(); df.close()
        print(f"{out}: {written} new spans, {len(done)} docs done "
              f"(provider={a.provider} model={a.model})")

if __name__ == "__main__":
    main()
