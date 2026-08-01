<#
.SYNOPSIS
    Launch the AML agent demo with the LLM planner enabled.

.DESCRIPTION
    Sets the environment a live demo needs and starts one component. Exists
    because two settings are easy to get wrong by hand and both fail quietly:

      AML_LLM_PLANNER  defaults to 0, so the planner stays deterministic and
                       the plan trace shows none of the `planner:` audit lines
                       the demo is meant to show. It defaults off on purpose --
                       there is no tests/conftest.py, so a default of on would
                       let the test suite make real API calls.

      OLLAMA_MODEL     defaults to a 7B that may not be pulled. A model that
                       is not on disk makes every LLM call return None, and
                       the system falls back so gracefully that the run looks
                       successful -- just deterministic.

    Environment variables take precedence over .env in pydantic-settings, so
    this overrides for the current process only. Nothing on disk is modified.

.PARAMETER Component
    backend  - uvicorn on :8000
    frontend - streamlit on :8501
    check    - verify Ollama is up and the configured model is pulled, then exit

.PARAMETER Model
    Ollama model to use. Defaults to qwen2.5:3b-instruct, which runs on a 4GB
    laptop GPU. Pass a larger one if you have it pulled.

.PARAMETER Deterministic
    Leave the LLM planner off, so the deterministic intent->tool table runs.
    Useful for showing the two modes side by side.

.EXAMPLE
    .\scripts\run_demo.ps1 check
    .\scripts\run_demo.ps1 backend      # terminal 1
    .\scripts\run_demo.ps1 frontend     # terminal 2
#>
[CmdletBinding()]
param(
    [Parameter(Position = 0)]
    [ValidateSet('backend', 'frontend', 'check')]
    [string]$Component = 'check',

    [string]$Model = 'qwen2.5:3b-instruct',

    [switch]$Deterministic
)

$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo

$python = Join-Path $repo '.venv\Scripts\python.exe'
if (-not (Test-Path $python)) {
    Write-Error "no virtualenv at $python - create one and pip install -r requirements.txt"
}

function Test-Ollama {
    param([string]$WantModel)

    try {
        $tags = Invoke-RestMethod -Uri 'http://localhost:11434/api/tags' -TimeoutSec 5
    }
    catch {
        Write-Host "Ollama is NOT reachable at localhost:11434" -ForegroundColor Red
        Write-Host "  start it, or run with -Deterministic to skip the LLM entirely."
        return $false
    }

    $have = @($tags.models | ForEach-Object { $_.name })
    Write-Host "Ollama is up. Models on disk:" -ForegroundColor Green
    $have | ForEach-Object { Write-Host "  $_" }

    # Ollama reports "name:tag"; a bare name matches its :latest tag.
    $match = $have | Where-Object { $_ -eq $WantModel -or $_ -eq "${WantModel}:latest" }
    if ($match) {
        Write-Host "requested model '$WantModel' is pulled." -ForegroundColor Green
        return $true
    }

    Write-Host "requested model '$WantModel' is NOT pulled." -ForegroundColor Red
    Write-Host "  fix:  ollama pull $WantModel"
    Write-Host "  or:   .\scripts\run_demo.ps1 $Component -Model <one listed above>"
    return $false
}

if ($Component -eq 'check') {
    $ok = Test-Ollama -WantModel $Model
    if ($ok) { Write-Host "`nready:  .\scripts\run_demo.ps1 backend" -ForegroundColor Cyan }
    exit ($(if ($ok) { 0 } else { 1 }))
}

if ($Component -eq 'frontend') {
    # The frontend speaks HTTP only and reads AML_API_URL; none of the LLM
    # settings apply to it.
    if (-not $env:AML_API_URL) { $env:AML_API_URL = 'http://localhost:8000' }
    Write-Host "starting streamlit against $env:AML_API_URL" -ForegroundColor Cyan
    & $python -m streamlit run frontend/app.py
    exit $LASTEXITCODE
}

# --- backend -------------------------------------------------------------
$env:AML_USE_MOCKS = '0'
$env:PYTHONPATH = $repo

if ($Deterministic) {
    $env:AML_LLM_PLANNER = '0'
    Write-Host "LLM planner OFF - deterministic intent->tool table" -ForegroundColor Yellow
}
else {
    $env:AML_LLM_PLANNER = '1'
    $env:LLM_PROVIDER = 'ollama'
    $env:OLLAMA_MODEL = $Model
    if (-not (Test-Ollama -WantModel $Model)) {
        Write-Host "`nrefusing to start with a model that is not pulled - the run would" -ForegroundColor Yellow
        Write-Host "look fine and silently use the deterministic planner for everything."
        exit 1
    }
    Write-Host "LLM planner ON - model=$Model" -ForegroundColor Green
    Write-Host "watch the plan trace for 'planner: source=llm' / 'planner: rejected'."
}

Write-Host "`nbackend on http://localhost:8000  (health: /health)" -ForegroundColor Cyan
& $python -m uvicorn backend.main:app --port 8000
