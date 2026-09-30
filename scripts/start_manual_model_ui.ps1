<# Start the separate research-only UI on the fixed loopback address. #>
[CmdletBinding()]
param([switch]$NoBrowser)

$ErrorActionPreference = 'Stop'
$repositoryRoot = (Resolve-Path -LiteralPath (Split-Path -Parent $PSScriptRoot)).Path
$pythonExecutable = Join-Path $repositoryRoot '.venv\Scripts\python.exe'
$bundlePath = Join-Path $repositoryRoot 'artifacts\models\eris_xgboost_v1.joblib'
$url = 'http://127.0.0.1:8501/'

if (-not (Test-Path -LiteralPath (Join-Path $repositoryRoot 'pyproject.toml') -PathType Leaf) -or
    -not (Test-Path -LiteralPath (Join-Path $repositoryRoot 'configs\features\feature_set_v1_full.yaml') -PathType Leaf)) {
    throw 'The script is not inside the expected ERIS repository.'
}
if (-not (Test-Path -LiteralPath $pythonExecutable -PathType Leaf)) {
    throw 'Python virtual environment is missing: .venv\Scripts\python.exe'
}
if (-not (Test-Path -LiteralPath $bundlePath -PathType Leaf)) {
    throw 'Frozen bundle is missing: artifacts\models\eris_xgboost_v1.joblib'
}
if (Get-NetTCPConnection -LocalPort 8501 -State Listen -ErrorAction SilentlyContinue) {
    throw 'Port 8501 is already in use. Stop the process listening there, then rerun this script.'
}

Push-Location -LiteralPath $repositoryRoot
try {
    # The existing CLI checks the pinned hash, sidecars, version and feature contract
    # with concise errors and no traceback or production service token.
    & $pythonExecutable (Join-Path $PSScriptRoot 'manual_model_test.py') --example low --no-explanation --json-output | Out-Null
    if ($LASTEXITCODE -ne 0) { throw 'Frozen bundle integrity check failed; local UI was not started.' }

    $browserJob = $null
    if (-not $NoBrowser) {
        $browserJob = Start-Job -ArgumentList $url -ScriptBlock {
            param($address)
            for ($attempt = 0; $attempt -lt 60; $attempt++) {
                try {
                    $response = Invoke-WebRequest -Uri ($address + 'health') -UseBasicParsing -TimeoutSec 1
                    if ($response.StatusCode -eq 200) {
                        Start-Process $address
                        return
                    }
                } catch { }
                Start-Sleep -Milliseconds 500
            }
        }
    }
    Write-Host "ERIS manual research UI: $url"
    Write-Host 'Press Ctrl+C to stop. No service token is needed.'
    try {
        & $pythonExecutable -m uvicorn eris_ml.manual_ui.app:app --host 127.0.0.1 --port 8501 --no-access-log
    } finally {
        if ($null -ne $browserJob) {
            Stop-Job -Job $browserJob -ErrorAction SilentlyContinue
            Remove-Job -Job $browserJob -Force -ErrorAction SilentlyContinue
        }
    }
} finally {
    Pop-Location
}
