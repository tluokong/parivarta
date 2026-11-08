# run_B.ps1 — full Dataset B (2,500 docs) LLM baseline with qwen2.5:3b via local Ollama.
# Resume-safe: re-run any time; already-done docs are skipped (tracked in exports\<Tag>_B.done).
# Usage (PowerShell, from the tools\eval folder):
#     .\run_B.ps1                    # qwen2.5:3b, tag "qwen"
#     .\run_B.ps1 -Model qwen2.5:7b  # a bigger model if RAM allows
param(
  [string]$Model = "qwen2.5:3b",
  [string]$Tag   = "qwen",
  [string]$Py    = "python"          # or full path to python.exe
)
$ErrorActionPreference = "Continue"
Set-Location $PSScriptRoot

Write-Host "Pulling model $Model ..." -ForegroundColor Cyan
ollama pull $Model | Out-Host

$total = (Get-Content .\exports\docs_B.jsonl | Measure-Object -Line).Lines
Write-Host "Dataset B has $total documents." -ForegroundColor Cyan

for ($attempt = 1; $attempt -le 200; $attempt++) {
    & $Py .\run_llm.py --exports exports --provider ollama --model $Model --tag $Tag --sets B --resume
    $done = 0
    if (Test-Path ".\exports\${Tag}_B.done") {
        $done = (Get-Content ".\exports\${Tag}_B.done" | Measure-Object -Line).Lines
    }
    Write-Host "progress: $done / $total" -ForegroundColor Yellow
    if ($done -ge $total) { break }
    Write-Host "resuming (attempt $attempt) ..." -ForegroundColor DarkYellow
    Start-Sleep -Seconds 3
}

Write-Host "`n=== Scoring (${Tag}) ===" -ForegroundColor Green
& $Py .\score.py --exports exports --system $Tag --latex
Write-Host "`nDone. Send back: exports\${Tag}_B.jsonl (and the score output above)." -ForegroundColor Green
