param(
    [ValidateSet("bot", "terminal", "api")]
    [string]$Mode = "bot"
)

$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $PSScriptRoot

$python = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
    $pythonCommand = Get-Command python -ErrorAction SilentlyContinue
    if ($null -eq $pythonCommand) {
        throw "Python was not found. Install Python 3.13 and follow the setup steps in README.md."
    }
    $python = $pythonCommand.Source
}

if (-not (Test-Path -LiteralPath (Join-Path $PSScriptRoot ".env") -PathType Leaf)) {
    throw "Missing .env. Run 'Copy-Item .env.example .env' and configure it as described in README.md."
}

if ($Mode -eq "bot") {
    $settings = @{}
    foreach ($line in Get-Content -LiteralPath (Join-Path $PSScriptRoot ".env")) {
        if ($line -match '^\s*(TELEGRAM_TOKEN|TELEGRAM_USER_ID)\s*=\s*(.*?)\s*$') {
            $settings[$Matches[1]] = $Matches[2].Trim().Trim('"').Trim("'")
        }
    }
    $hasToken = $settings["TELEGRAM_TOKEN"] -and $settings["TELEGRAM_TOKEN"] -ne "your-bot-token"
    $hasUserId = $settings["TELEGRAM_USER_ID"] -and $settings["TELEGRAM_USER_ID"] -ne "your-numeric-id"
    if (-not $hasToken -or -not $hasUserId) {
        throw "Set TELEGRAM_TOKEN and TELEGRAM_USER_ID in .env before starting the bot."
    }
    $module = "app.telegram_bot"
}
elseif ($Mode -eq "terminal") {
    $module = "app.agent"
}
else {
    $module = "uvicorn"
}

if ($Mode -eq "api") {
    & $python -m $module app.api:app --reload
}
else {
    & $python -m $module
}
exit $LASTEXITCODE
