# Run from a downloaded/cloned MOLT repository. No administrator rights required.
[CmdletBinding()]
param([switch]$EagerOnly, [switch]$Plan)

$ErrorActionPreference = 'Stop'
$syncArgs = @('sync', '--locked', '--no-dev', '--extra', 'qlora', '--extra', 'data')
if (-not $EagerOnly) { $syncArgs += @('--extra', 'windows-fusion') }
$checkArgs = @('run', '--no-sync', 'python', '-m', 'molt_stream.setup_check')
if (-not $EagerOnly) { $checkArgs += '--compile' }
if ($Plan) {
    @{
        project = $PSScriptRoot
        sync = $syncArgs
        check = $checkArgs
        legacy_distribution = 'molt-ai-infrastructure'
    } | ConvertTo-Json
    return
}
if ($env:OS -ne 'Windows_NT') { throw 'This installer targets Windows. See docs/INSTALL.md.' }
if (-not (Test-Path -LiteralPath (Join-Path $PSScriptRoot 'uv.lock'))) {
    throw 'Download the complete MOLT repository, not only install.ps1.'
}
Write-Host 'MOLT setup: stop training first. Downloads may require several GB.' -ForegroundColor Green
Write-Host 'Installs into .venv. No driver, antivirus, power, or app-suspension changes.'
$uvCommand = Get-Command uv -ErrorAction SilentlyContinue
$uv = if ($uvCommand) { $uvCommand.Source } else { Join-Path $PSScriptRoot '.tools\uv\uv.exe' }
if (-not $uvCommand -and -not (Test-Path -LiteralPath $uv)) {
    Write-Host 'Installing uv from the official Astral installer into .tools/uv ...'
    $toolsDir = Join-Path $PSScriptRoot '.tools'
    New-Item -ItemType Directory -Force -Path $toolsDir | Out-Null
    $bootstrap = Join-Path $toolsDir 'install-uv.ps1'
    Invoke-WebRequest -UseBasicParsing -Uri 'https://astral.sh/uv/0.12.9/install.ps1' -OutFile $bootstrap
    $previousDir = $env:UV_INSTALL_DIR
    $previousPathSetting = $env:UV_NO_MODIFY_PATH
    try {
        $env:UV_INSTALL_DIR = Join-Path $toolsDir 'uv'
        $env:UV_NO_MODIFY_PATH = '1'
        & powershell.exe -NoProfile -ExecutionPolicy Bypass -File $bootstrap
        if ($LASTEXITCODE -ne 0) { throw 'uv bootstrap failed.' }
    } finally {
        $env:UV_INSTALL_DIR = $previousDir
        $env:UV_NO_MODIFY_PATH = $previousPathSetting
    }
    if (-not (Test-Path -LiteralPath $uv)) { throw 'uv was not installed at the expected path.' }
}
Push-Location $PSScriptRoot
try {
    $legacyRemoved = $false
    $venvPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
    if (Test-Path -LiteralPath $venvPython) {
        $legacyMetadata = & $uv pip show --python $venvPython molt-ai-infrastructure 2>$null
        if ($LASTEXITCODE -eq 0 -and $legacyMetadata) {
            Write-Host 'Removing legacy molt-ai-infrastructure package metadata...'
            & $uv pip uninstall --python $venvPython molt-ai-infrastructure
            if ($LASTEXITCODE -ne 0) { throw 'Legacy package-name migration failed.' }
            $legacyRemoved = $true
        }
    }
    if ($legacyRemoved) {
        # Both distributions expose the same console-script name. Removing the
        # legacy wheel can therefore remove molt.exe even when moltengine is
        # already installed; force recreation of the current launcher.
        $syncArgs += @('--reinstall-package', 'moltengine')
    }
    & $uv @syncArgs
    if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed; setup is incomplete.' }
    & $uv @checkArgs
    if ($LASTEXITCODE -ne 0) {
        throw 'Runtime check failed. Read the error above; setup is not verified. See docs/INSTALL.md.'
    }
    Write-Host 'Setup checks passed. This is not a model-fit or sustained thermal guarantee.' -ForegroundColor Green
    Write-Host 'Launch from this folder: .\.venv\Scripts\molt.exe --ui inspect'
} finally {
    Pop-Location
}
