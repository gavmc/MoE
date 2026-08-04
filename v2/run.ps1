# Supervisor loop (train.md §7). Relaunches train.py on nonzero exit, with a
# retry cap so a deterministic crash doesn't spin for 60 hours burning the window.
#
#   .\run.ps1                        # main config
#   .\run.ps1 -Config debug          # debug config
#   .\run.ps1 -MaxRetries 20

param(
    [string]$Config     = "main",
    [string]$Out        = "",
    [string]$ExtraArgs  = "",     # passed through to train.py, e.g. "--total-steps 300"
    [int]   $MaxRetries = 12,
    [int]   $CooldownSec = 30,
    [int]   $ResetAfterMin = 30   # a run that survived this long resets the retry counter
)

$ErrorActionPreference = "Stop"
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $here

$argsList = @("train.py", "--config", $Config)
if ($Out) { $argsList += @("--out", $Out) }
if ($ExtraArgs) { $argsList += ($ExtraArgs -split '\s+' | Where-Object { $_ }) }

$attempt = 0
$startedAll = Get-Date

while ($true) {
    $attempt++
    $t0 = Get-Date
    Write-Host ""
    Write-Host "=== attempt $attempt/$MaxRetries  $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ===" -ForegroundColor Cyan

    # -u: unbuffered stdout. Without it Python block-buffers when redirected and the
    # console shows nothing live -- the JSONL is still correct, but you cannot glance
    # at the window and see progress.
    & python -u @argsList
    $code = $LASTEXITCODE
    $mins = ((Get-Date) - $t0).TotalMinutes

    if ($code -eq 0) {
        Write-Host "train.py exited 0 after $([math]::Round($mins,1)) min - run complete." -ForegroundColor Green
        break
    }

    Write-Host "train.py exited $code after $([math]::Round($mins,1)) min." -ForegroundColor Yellow

    if ($mins -ge $ResetAfterMin) {
        Write-Host "survived >= $ResetAfterMin min; resetting retry counter." -ForegroundColor Yellow
        $attempt = 0
    }
    elseif ($attempt -ge $MaxRetries) {
        Write-Host "hit $MaxRetries consecutive fast failures - giving up." -ForegroundColor Red
        Write-Host "total wall clock: $([math]::Round(((Get-Date) - $startedAll).TotalHours,2)) h"
        exit 1
    }

    Write-Host "relaunching in $CooldownSec s (auto-resume picks up ckpt_last.pt)..."
    Start-Sleep -Seconds $CooldownSec
}

Write-Host "total wall clock: $([math]::Round(((Get-Date) - $startedAll).TotalHours,2)) h"
