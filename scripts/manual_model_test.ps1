<#
Local wrapper for the frozen ERIS model. No server or service token is required.
#>
[CmdletBinding()]
param(
    [ValidateSet('low', 'high')]
    [string]$Example,
    [string]$InputFile,
    [switch]$Interactive,
    [switch]$NoExplanation,
    [ValidateRange(1, 10)]
    [int]$TopK = 5,
    [switch]$JsonOutput,
    [switch]$DebugMode
)

$ErrorActionPreference = 'Stop'
$repositoryRoot = (Resolve-Path -LiteralPath (Split-Path -Parent $PSScriptRoot)).Path
$pythonExecutable = Join-Path $repositoryRoot '.venv\Scripts\python.exe'
$bundlePath = Join-Path $repositoryRoot 'artifacts\models\eris_xgboost_v1.joblib'
$cliPath = Join-Path $PSScriptRoot 'manual_model_test.py'

if (-not (Test-Path -LiteralPath (Join-Path $repositoryRoot 'pyproject.toml') -PathType Leaf) -or
    -not (Test-Path -LiteralPath (Join-Path $repositoryRoot 'configs\features\feature_set_v1_full.yaml') -PathType Leaf)) {
    Write-Error 'The script is not inside the expected ERIS repository.'
    exit 1
}

if (-not (Test-Path -LiteralPath $pythonExecutable -PathType Leaf)) {
    Write-Error 'Python virtual environment is missing: .venv\Scripts\python.exe'
    exit 1
}
if (-not (Test-Path -LiteralPath $bundlePath -PathType Leaf)) {
    Write-Error 'Frozen model bundle is missing: artifacts\models\eris_xgboost_v1.joblib'
    exit 1
}

$modeCount = [int][bool]$Example + [int][bool]$InputFile + [int]$Interactive.IsPresent
if ($modeCount -gt 1) {
    Write-Error 'Choose only one of -Example, -InputFile, or -Interactive.'
    exit 2
}

$pythonArguments = @($cliPath)
if ($Example) { $pythonArguments += @('--example', $Example) }
if ($InputFile) { $pythonArguments += @('--input-file', $InputFile) }
if ($Interactive) { $pythonArguments += '--interactive' }
if ($NoExplanation) { $pythonArguments += '--no-explanation' }
$pythonArguments += @('--top-k', [string]$TopK)
if ($JsonOutput) { $pythonArguments += '--json-output' }
if ($DebugMode) { $pythonArguments += '--debug' }

$resultCode = 1
Push-Location -LiteralPath $repositoryRoot
try {
    & $pythonExecutable @pythonArguments
    $resultCode = $LASTEXITCODE
} finally {
    Pop-Location
}
exit $resultCode
