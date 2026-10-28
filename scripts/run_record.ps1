# GRU data collection: runs PDM-Lite + feature logging (team_code\recording_agent.py)
# on one or more routes, several repetitions each, with a FRESH CARLA for every episode.
#
# Usage:
#   .\run_record.ps1 165_stalled                         (1 route, 5 repetitions)
#   .\run_record.ps1 165_broken,165_stalled,165_queue,normal05 -Reps 10
#
# Episode class (label) is decided from the route name:
#   contains "broken" or "stalled"  -> P  (stop that never resolves -> LLM needed)
#   anything else                   -> N  (stop resolves by itself / normal driving)
# Feature CSVs go to C:\carla_garage\results\features\<route>_<timestamp>.csv
param(
    [Parameter(Mandatory=$true)][string[]]$Ids,
    [int]$Reps = 5,
    [int]$MaxStopSec = 40,
    [string]$OutDir = "features"     # results\<OutDir>  (use e.g. features_newroads for held-out test data)
)

$CARLA_EXE = "C:\carla0915\WindowsNoEditor\CarlaUE4.exe"
$env:CARLA_ROOT = "C:\carla0915\WindowsNoEditor"
$env:PYTHONIOENCODING = "utf-8"
$env:WORK_DIR = "C:\carla_garage"
$env:SCENARIO_RUNNER_ROOT = "$env:WORK_DIR\scenario_runner_autopilot"
$env:LEADERBOARD_ROOT = "$env:WORK_DIR\leaderboard_autopilot"
$env:PYTHONPATH = "$env:CARLA_ROOT\PythonAPI\carla;$env:SCENARIO_RUNNER_ROOT;$env:LEADERBOARD_ROOT;$env:WORK_DIR\team_code"
$env:DATAGEN = "0"
$env:NO_BACKGROUND = "0"   # (may be left over from run_intervene.ps1 -NoTraffic)
$env:MAX_STOP_SEC = "$MaxStopSec"
$env:RECORD_DIR = "$env:WORK_DIR\results\$OutDir"
$PY = "$env:WORK_DIR\venv\Scripts\python.exe"
New-Item -ItemType Directory -Force "$env:RECORD_DIR" | Out-Null
New-Item -ItemType Directory -Force "$env:WORK_DIR\results\record_logs" | Out-Null

function Stop-Carla {
    Get-Process | Where-Object { $_.Name -like "CarlaUE4*" -or $_.Name -like "CrashReportClient*" } | Stop-Process -Force -ErrorAction SilentlyContinue
    Start-Sleep -Seconds 3
}

function Start-CarlaOnce($town) {
    Stop-Carla
    Start-Process $CARLA_EXE -ArgumentList "-quality-level=Low -dx11 -prefernvidia" | Out-Null
    $ok = $false
    for ($i = 0; $i -lt 30; $i++) {
        Start-Sleep -Seconds 5
        & $PY -c "import carla; c=carla.Client('localhost',2000); c.set_timeout(5); c.get_server_version()" 2>$null | Out-Null
        if ($LASTEXITCODE -eq 0) { $ok = $true; break }
    }
    if (-not $ok) { return $false }
    Start-Sleep -Seconds 15   # let the default map finish streaming before switching maps
    & $PY -c "import carla,time; c=carla.Client('localhost',2000); c.set_timeout(120); cur=c.get_world().get_map().name.split('/')[-1]; (cur=='$town') or c.load_world('$town'); time.sleep(5); print('map:', c.get_world().get_map().name)" | Out-Host
    if ($LASTEXITCODE -ne 0) { return $false }
    Start-Sleep -Seconds 5
    return $true
}

# The Windows build sometimes crashes while switching maps -> retry with a fresh CARLA (up to 3 times)
function Start-CarlaInMap($town) {
    for ($try = 1; $try -le 3; $try++) {
        if (Start-CarlaOnce $town) { return $true }
        Write-Host "!! CARLA crashed while loading $town (try $try/3) - restarting"
    }
    return $false
}

$summary = @()
foreach ($Id in $Ids) {
    $ROUTE = "$env:WORK_DIR\leaderboard\data\bench2drive_split\bench2drive_$Id.xml"
    if (-not (Test-Path $ROUTE)) { Write-Host "!! Route file not found: $ROUTE (skipped)"; continue }
    $TOWN = ([xml](Get-Content $ROUTE)).routes.route.town
    $CLASS = if ($Id -match 'broken|stalled') { 'P' } else { 'N' }
    $env:RECORD_TAG = $Id
    $env:RECORD_CLASS = $CLASS
    $env:END_AFTER_RELEASE = if ($Id -match 'queue') { '1' } else { '0' }

    for ($r = 1; $r -le $Reps; $r++) {
        Write-Host ""
        Write-Host "######## $Id  (class $CLASS, map $TOWN)  repetition $r / $Reps ########"
        if (-not (Start-CarlaInMap $TOWN)) {
            Write-Host "!! CARLA failed to start / load map - skipping this repetition"
            $summary += "$Id rep$r : CARLA FAILED"
            continue
        }
        $LOG = "$env:WORK_DIR\results\record_logs\${Id}_rep$r.log.txt"
        & $PY "$env:LEADERBOARD_ROOT\leaderboard\leaderboard_evaluator_local.py" `
            --agent "$env:WORK_DIR\team_code\recording_agent.py" `
            --routes $ROUTE `
            --checkpoint "$env:WORK_DIR\results\record_logs\${Id}_rep$r.json" `
            --track MAP 2>&1 | ForEach-Object { "$_" } | Tee-Object -FilePath $LOG
        $summary += "$Id rep$r : done"
        Stop-Carla
    }
}

Write-Host ""
Write-Host "================ SUMMARY ================"
$summary | ForEach-Object { Write-Host $_ }
Write-Host "Feature CSVs: $env:RECORD_DIR"
