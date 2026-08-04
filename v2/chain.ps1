# Wait for pre-training to finish, then run SFT and the effort-conditioning eval.
#
#   .\chain.ps1
#
# Safe to start while pre-training is already running in another window: it only
# watches, and only relaunches training if the process has been gone for two
# consecutive polls (longer than run.ps1's own cooldown, so the two cannot race).
#
# NOTE it cannot survive a system sleep -- nothing in a user session can. That is
# what keep_awake() in train.py is for; this script only covers ordinary crashes.

param(
    [string]$MainOut      = "D:\ml\runs\main",
    [string]$SftOut       = "D:\ml\runs\sft",
    [string]$SftData      = "D:\ml\data-sft",
    [int]   $PollSec      = 120,
    [double]$MaxWaitHours = 20,
    [switch]$SkipWait                       # go straight to SFT
)

$ErrorActionPreference = "Stop"
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $here

function Get-Progress {
    $cfgPath = Join-Path $MainOut "config.json"
    $logPath = Join-Path $MainOut "log.jsonl"
    if (-not (Test-Path $cfgPath) -or -not (Test-Path $logPath)) { return $null }
    $total = (Get-Content $cfgPath -Raw | ConvertFrom-Json).total_steps
    $last = Get-Content $logPath -Tail 1
    if (-not $last) { return $null }
    $step = ($last | ConvertFrom-Json).step
    return @{ step = $step; total = $total }
}

$start = Get-Date
$misses = 0

if (-not $SkipWait) {
    Write-Host "=== waiting for pre-training to reach total_steps ===" -ForegroundColor Cyan
    while ($true) {
        if (((Get-Date) - $start).TotalHours -gt $MaxWaitHours) {
            Write-Host "gave up waiting after $MaxWaitHours h - not starting SFT." -ForegroundColor Red
            exit 1
        }
        $p = Get-Progress
        $alive = [bool](Get-Process python -ErrorAction SilentlyContinue)

        if ($p -and $p.step -ge $p.total) {
            if (-not $alive) {
                Write-Host "pre-training complete at step $($p.step)." -ForegroundColor Green
                break
            }
            Write-Host "step $($p.step)/$($p.total) - waiting for final checkpoint write..."
        }
        elseif ($p) {
            $pct = [math]::Round(100 * $p.step / $p.total, 1)
            if ($alive) {
                $misses = 0
                Write-Host ("{0}  step {1:N0}/{2:N0} ({3}%)" -f (Get-Date -Format 'HH:mm'), $p.step, $p.total, $pct)
            }
            else {
                $misses++
                Write-Host "no python running at step $($p.step)/$($p.total) (miss $misses/2)" -ForegroundColor Yellow
                if ($misses -ge 2) {
                    Write-Host "relaunching pre-training..." -ForegroundColor Yellow
                    Start-Process powershell -ArgumentList @(
                        "-ExecutionPolicy", "Bypass", "-File", (Join-Path $here "run.ps1"),
                        "-Config", "main") -WorkingDirectory $here
                    $misses = 0
                    Start-Sleep -Seconds 90
                }
            }
        }
        Start-Sleep -Seconds $PollSec
    }
}

Write-Host ""
Write-Host "=== SFT ===" -ForegroundColor Cyan
& python -u sft.py --base (Join-Path $MainOut "ckpt_last.pt") --data $SftData --out $SftOut --require-complete
if ($LASTEXITCODE -ne 0) {
    Write-Host "SFT failed with $LASTEXITCODE - base model is untouched at $MainOut" -ForegroundColor Red
    exit $LASTEXITCODE
}

Write-Host ""
Write-Host "=== effort-conditioning eval (train.md sec 9) ===" -ForegroundColor Cyan
& python -u sample.py --ckpt (Join-Path $SftOut "ckpt_last.pt") --data $SftData `
    --out (Join-Path $SftOut "samples.md")
if ($LASTEXITCODE -ne 0) { Write-Host "sampling failed with $LASTEXITCODE" -ForegroundColor Yellow }

Write-Host ""
Write-Host "ALL DONE. Read $SftOut\samples.md" -ForegroundColor Green
Write-Host "total wall clock: $([math]::Round(((Get-Date) - $start).TotalHours,2)) h"
