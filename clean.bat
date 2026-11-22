@echo off
setlocal EnableExtensions DisableDelayedExpansion

if not "%~2"=="" goto usage
if "%~1"=="" goto clean
if /i "%~1"=="/all" goto clean
if "%~1"=="/?" goto help
goto usage

:clean
set "CONVECTION_CLEAN_ROOT=%~dp0"
set "CONVECTION_CLEAN_MODE=%~1"

powershell.exe -NoLogo -NoProfile -Command ^
    "$ErrorActionPreference = 'Stop';" ^
    "$root = (Resolve-Path -LiteralPath $env:CONVECTION_CLEAN_ROOT).ProviderPath;" ^
    "$rootItem = Get-Item -LiteralPath $root -Force;" ^
    "if ($rootItem.Attributes -band [IO.FileAttributes]::ReparsePoint) { throw 'The project directory is a link.' };" ^
    "if (-not (Test-Path -LiteralPath (Join-Path $root 'convection_3D.vcxproj') -PathType Leaf)) { throw 'Place clean.bat next to convection_3D.vcxproj.' };" ^
    "$targets = @('.vs', 'x64', 'convection_3D\x64', 'convection_3D.vcxproj.user', 'fields');" ^
    "if ($env:CONVECTION_CLEAN_MODE -ieq '/all') {" ^
    "    $targets += @('Ra.dat', 'integrals.dat', 'timer_report.dat', 'recovery.txt.tmp');" ^
    "    $parameters = Join-Path $root 'parameters.txt';" ^
    "    if (Test-Path -LiteralPath $parameters -PathType Leaf) {" ^
    "        $mode = @(Select-String -LiteralPath $parameters -Pattern '^\s*read_recovery(?:\s|=|$)');" ^
    "        if ($mode.Count -eq 1 -and $mode[0].Line -match '^\s*read_recovery(?:\s*=\s*|\s+)[+-]?0+\s*(?:#.*)?$') {" ^
    "            $targets += @('recovery.txt', 'recovery2.txt');" ^
    "        };" ^
    "    };" ^
    "};" ^
    "$prefix = $root.TrimEnd('\') + '\';" ^
    "$items = foreach ($relative in $targets) {" ^
    "    $candidate = Join-Path $root $relative;" ^
    "    if (-not (Test-Path -LiteralPath $candidate)) { continue };" ^
    "    $path = (Resolve-Path -LiteralPath $candidate).ProviderPath;" ^
    "    if (-not $path.StartsWith($prefix, [StringComparison]::OrdinalIgnoreCase)) { throw ('Path outside the project: ' + $path) };" ^
    "    $partPath = $root;" ^
    "    foreach ($part in ($relative -split '\\')) {" ^
    "        $partPath = Join-Path $partPath $part;" ^
    "        $item = Get-Item -LiteralPath $partPath -Force;" ^
    "        if ($item.Attributes -band [IO.FileAttributes]::ReparsePoint) { throw ('Link in cleanup path: ' + $partPath) };" ^
    "    };" ^
    "    if ($item.PSIsContainer -and (Get-ChildItem -LiteralPath $path -Force -Recurse -Attributes ReparsePoint | Select-Object -First 1)) {" ^
    "        throw ('Link inside cleanup directory: ' + $path);" ^
    "    };" ^
    "    $item;" ^
    "};" ^
    "foreach ($item in $items) {" ^
    "    Remove-Item -LiteralPath $item.FullName -Force -Recurse:($item.PSIsContainer);" ^
    "    Write-Host ('Removed ' + $item.FullName);" ^
    "};"
exit /b %errorlevel%

:usage
echo Usage: clean.bat [/all]
exit /b 2

:help
echo clean.bat       Remove build files, Visual Studio cache, and fields.
echo clean.bat /all  Also remove calculation logs and unused checkpoints.
exit /b 0
