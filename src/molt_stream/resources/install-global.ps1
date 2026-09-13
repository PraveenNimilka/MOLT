# Install or manage one per-user MOLT runtime. Administrator rights are not required.
[CmdletBinding()]
param(
    [ValidateSet('Install', 'Update', 'Repair', 'Uninstall')]
    [string]$Action = 'Install',
    [string]$Version = '',
    [string]$PackageSource = '',
    [switch]$SkipCompile,
    [switch]$Plan
)

$ErrorActionPreference = 'Stop'
$releaseVersion = '0.12.0'
$runtimeHome = if ($env:MOLT_HOME) {
    [IO.Path]::GetFullPath($env:MOLT_HOME)
} else {
    Join-Path $env:LOCALAPPDATA 'MOLT'
}
$runtime = Join-Path $runtimeHome 'runtime'
$runtimePython = Join-Path $runtime 'Scripts\python.exe'
$bin = Join-Path $runtimeHome 'bin'
$launcher = Join-Path $bin 'molt.cmd'
$cache = Join-Path $runtimeHome 'cache\uv'
$tools = Join-Path $runtimeHome 'tools'
$uv = Join-Path $tools 'uv\uv.exe'
$savedInstaller = Join-Path $runtimeHome 'installer.ps1'

function Get-TargetVersion {
    if ($Version) { return $Version }
    if ($Action -eq 'Update') {
        $metadata = Invoke-RestMethod -Uri 'https://pypi.org/pypi/moltengine/json'
        return [string]$metadata.info.version
    }
    if ($Action -eq 'Repair' -and (Test-Path -LiteralPath $runtimePython)) {
        $installed = & $runtimePython -c "import importlib.metadata as m; print(m.version('moltengine'))" 2>$null
        if ($LASTEXITCODE -eq 0 -and $installed) { return [string]$installed.Trim() }
    }
    return $releaseVersion
}

function Set-MoltUserPath([bool]$Present) {
    $current = [Environment]::GetEnvironmentVariable('Path', 'User')
    $normalizedBin = $bin.Trim().TrimEnd('\')
    $parts = @($current -split ';' | Where-Object {
        $_ -and ($_.Trim().TrimEnd('\') -ine $normalizedBin)
    })
    if ($Present) { $parts = @($bin) + $parts }
    [Environment]::SetEnvironmentVariable('Path', ($parts -join ';'), 'User')
    $processParts = @($env:Path -split ';' | Where-Object {
        $_ -and ($_.Trim().TrimEnd('\') -ine $normalizedBin)
    })
    if ($Present) {
        $env:Path = (@($bin) + $processParts) -join ';'
    } else {
        $env:Path = $processParts -join ';'
    }
}

function Write-Launcher {
    New-Item -ItemType Directory -Force -Path $bin | Out-Null
    $content = "@echo off`r`n`"$runtime\Scripts\molt.exe`" %*`r`n"
    [IO.File]::WriteAllText($launcher, $content, [Text.Encoding]::ASCII)
    Set-MoltUserPath $true
}

function Get-LauncherConflicts {
    $expected = [IO.Path]::GetFullPath($launcher)
    return @(Get-Command molt -All -ErrorAction SilentlyContinue | Where-Object {
        [IO.Path]::GetFullPath($_.Source) -ne $expected
    } | Select-Object -ExpandProperty Source -Unique)
}

function Install-Uv {
    if (Test-Path -LiteralPath $uv) { return }
    New-Item -ItemType Directory -Force -Path $tools | Out-Null
    $bootstrap = Join-Path $tools 'install-uv.ps1'
    Write-Host 'Downloading the pinned uv package manager...'
    Invoke-WebRequest -UseBasicParsing -Uri 'https://astral.sh/uv/0.12.9/install.ps1' -OutFile $bootstrap
    $expected = '69DE475BF929F1AC248EFB5A85189177A45517E2346CD68762BDE453FEC10A6B'
    if ((Get-FileHash -Algorithm SHA256 -LiteralPath $bootstrap).Hash -ne $expected) {
        throw 'uv bootstrap integrity verification failed.'
    }
    $oldInstallDir = $env:UV_INSTALL_DIR
    $oldNoPath = $env:UV_NO_MODIFY_PATH
    try {
        $env:UV_INSTALL_DIR = Join-Path $tools 'uv'
        $env:UV_NO_MODIFY_PATH = '1'
        & powershell.exe -NoProfile -ExecutionPolicy Bypass -File $bootstrap
        if ($LASTEXITCODE -ne 0) { throw 'uv bootstrap failed.' }
    } finally {
        $env:UV_INSTALL_DIR = $oldInstallDir
        $env:UV_NO_MODIFY_PATH = $oldNoPath
    }
    if (-not (Test-Path -LiteralPath $uv)) { throw 'uv was not installed at the expected path.' }
}

$targetVersion = if ($Action -eq 'Uninstall') { $null } else { Get-TargetVersion }
if ($targetVersion -and $targetVersion -notmatch '^\d+\.\d+\.\d+(?:a\d+|b\d+|rc\d+)?$') {
    throw "Invalid package version returned: $targetVersion"
}

if ($Plan) {
    [ordered]@{
        action = $Action
        version = $targetVersion
        home = $runtimeHome
        runtime = $runtime
        launcher = $launcher
        cache = $cache
        cuda_index = 'https://download.pytorch.org/whl/cu128'
        package_source = $PackageSource
    } | ConvertTo-Json
    return
}

if ($env:OS -ne 'Windows_NT') { throw 'The global installer currently supports Windows only.' }

if ($Action -eq 'Uninstall') {
    Set-MoltUserPath $false
    $cleanup = Join-Path $env:TEMP ("molt-uninstall-{0}.ps1" -f [guid]::NewGuid())
    $quotedHome = $runtimeHome.Replace("'", "''")
    $quotedCleanup = $cleanup.Replace("'", "''")
    [IO.File]::WriteAllText($cleanup, @"
for (`$attempt = 0; `$attempt -lt 60; `$attempt++) {
    try {
        if (Test-Path -LiteralPath '$quotedHome') {
            Remove-Item -LiteralPath '$quotedHome' -Recurse -Force -ErrorAction Stop
        }
        break
    } catch {
        Start-Sleep -Seconds 1
    }
}
if (-not (Test-Path -LiteralPath '$quotedHome')) {
    Remove-Item -LiteralPath '$quotedCleanup' -Force
}
"@, [Text.UTF8Encoding]::new($false))
    Start-Process powershell.exe -WindowStyle Hidden -ArgumentList @(
        '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', $cleanup
    ) | Out-Null
    Write-Host 'MOLT uninstall scheduled. The runtime, launcher, and cached wheels will be removed.' -ForegroundColor Green
    return
}

Write-Host "MOLT $Action`: installing $targetVersion into $runtime" -ForegroundColor Green
New-Item -ItemType Directory -Force -Path $runtimeHome, $cache | Out-Null
Install-Uv
if (-not (Test-Path -LiteralPath $runtimePython)) {
    & $uv venv --python 3.12 $runtime
    if ($LASTEXITCODE -ne 0) { throw 'Python 3.12 runtime creation failed.' }
}

# Pin the CUDA build explicitly. uv's persistent cache prevents repeat downloads.
& $uv pip install --python $runtimePython --cache-dir $cache `
    --extra-index-url 'https://download.pytorch.org/whl/cu128' `
    --index-strategy unsafe-best-match 'torch==2.8.0+cu128'
if ($LASTEXITCODE -ne 0) { throw 'CUDA PyTorch installation failed.' }

$extras = '[qlora,data,windows-fusion,windows-monitoring]'
$package = if ($PackageSource) {
    ([IO.Path]::GetFullPath($PackageSource) + $extras)
} else {
    "moltengine$extras==$targetVersion"
}
$installArgs = @('pip', 'install', '--python', $runtimePython, '--cache-dir', $cache)
$heldExecutable = $null
if ($Action -in @('Repair', 'Update')) {
    $installArgs += @('--reinstall-package', 'moltengine')
    $runtimeMolt = Join-Path $runtime 'Scripts\molt.exe'
    if (Test-Path -LiteralPath $runtimeMolt) {
        # Windows locks a running console-script executable. Renaming is atomic
        # and lets uv create the replacement while the old process finishes.
        $heldExecutable = Join-Path $runtime 'Scripts\molt.previous.exe'
        if (Test-Path -LiteralPath $heldExecutable) {
            Remove-Item -LiteralPath $heldExecutable -Force
        }
        Move-Item -LiteralPath $runtimeMolt -Destination $heldExecutable
    }
}
$installArgs += @('--prerelease', 'allow', $package)
& $uv @installArgs
if ($LASTEXITCODE -ne 0) {
    $runtimeMolt = Join-Path $runtime 'Scripts\molt.exe'
    if ($heldExecutable -and -not (Test-Path -LiteralPath $runtimeMolt)) {
        Move-Item -LiteralPath $heldExecutable -Destination $runtimeMolt
    }
    throw 'MOLT package installation failed.'
}
if ($heldExecutable) {
    $cleanup = Join-Path $env:TEMP ("molt-update-cleanup-{0}.ps1" -f [guid]::NewGuid())
    $quotedExecutable = $heldExecutable.Replace("'", "''")
    $quotedCleanup = $cleanup.Replace("'", "''")
    [IO.File]::WriteAllText($cleanup, @"
for (`$attempt = 0; `$attempt -lt 60; `$attempt++) {
    try {
        if (Test-Path -LiteralPath '$quotedExecutable') {
            Remove-Item -LiteralPath '$quotedExecutable' -Force -ErrorAction Stop
        }
        break
    } catch {
        Start-Sleep -Seconds 1
    }
}
if (-not (Test-Path -LiteralPath '$quotedExecutable')) {
    Remove-Item -LiteralPath '$quotedCleanup' -Force
}
"@, [Text.UTF8Encoding]::new($false))
    Start-Process powershell.exe -WindowStyle Hidden -ArgumentList @(
        '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', $cleanup
    ) | Out-Null
}

& $uv pip check --python $runtimePython
if ($LASTEXITCODE -ne 0) { throw 'Dependency verification failed.' }
$check = @('-m', 'molt_stream.setup_check')
if (-not $SkipCompile) { $check += '--compile' }
& $runtimePython @check
if ($LASTEXITCODE -ne 0) { throw 'CUDA runtime verification failed.' }

Write-Launcher
if ($PSCommandPath -and ([IO.Path]::GetFullPath($PSCommandPath) -ne [IO.Path]::GetFullPath($savedInstaller))) {
    Copy-Item -LiteralPath $PSCommandPath -Destination $savedInstaller -Force
}
$actualVersion = & $runtimePython -c "import importlib.metadata as m; print(m.version('moltengine'))"
$state = [ordered]@{
    schema_version = 1
    status = 'verified'
    version = [string]$actualVersion.Trim()
    runtime = $runtime
    launcher = $launcher
    cache = $cache
    installed_at_utc = [DateTime]::UtcNow.ToString('o')
}
[IO.File]::WriteAllText(
    (Join-Path $runtimeHome 'install-state.json'),
    (($state | ConvertTo-Json) + "`n"),
    [Text.UTF8Encoding]::new($false)
)

$conflicts = Get-LauncherConflicts
Write-Host "MOLT $($state.version) installed and verified." -ForegroundColor Green
Write-Host "Launcher: $launcher"
Write-Host "CUDA package cache: $cache"
if ($conflicts) {
    Write-Warning ('Older MOLT launchers remain but are lower priority: ' + ($conflicts -join ', '))
}
Write-Host 'Open a new terminal, then run: molt doctor' -ForegroundColor Green
