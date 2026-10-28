# Usage:  .\run_route.ps1 174
# Restarts CARLA 0.9.15 fresh, runs ONE Bench2Drive route with PDM-Lite, then closes CARLA.
# (CARLA 0.9.15 on Windows crashes on the 2nd map load in the same session, so we restart every run.)
param([Parameter(Mandatory=$true)][string]$Id)

$CARLA_EXE = "C:\carla0915\WindowsNoEditor\CarlaUE4.exe"
$env:CARLA_ROOT = "C:\carla0915\WindowsNoEditor"
$env:PYTHONIOENCODING = "utf-8"
$env:WORK_DIR = "C:\carla_garage"
$env:SCENARIO_RUNNER_ROOT = "$env:WORK_DIR\scenario_runner_autopilot"
$env:LEADERBOARD_ROOT = "$env:WORK_DIR\leaderboard_autopilot"
$env:PYTHONPATH = "$env:CARLA_ROOT\PythonAPI\carla;$env:SCENARIO_RUNNER_ROOT;$env:LEADERBOARD_ROOT;$env:WORK_DIR\team_code"
$env:DATAGEN = "0"
$env:NO_BACKGROUND = "0"   # (may be left over from run_intervene.ps1 -NoTraffic)
$PY = "$env:WORK_DIR\venv\Scripts\python.exe"
$ROUTE = "$env:WORK_DIR\leaderboard\data\bench2drive_split\bench2drive_$Id.xml"
$OUT = "$env:WORK_DIR\results\b2d_$Id.json"
$LOG = "$env:WORK_DIR\results\b2d_$Id.log.txt"   # terminal output copy (readable later)
New-Item -ItemType Directory -Force "$env:WORK_DIR\results" | Out-Null

if (-not (Test-Path $ROUTE)) { Write-Host "Route file not found: $ROUTE"; exit 1 }
# Start CARLA directly in the route's map so the leaderboard does not need load_world (Windows crash workaround)
$TOWN = ([xml](Get-Content $ROUTE)).routes.route.town
Write-Host ">> Route map: $TOWN"

function Stop-Carla {
    Get-Process | Where-Object { $_.Name -like "CarlaUE4*" -or $_.Name -like "CrashReportClient*" } | Stop-Process -Force -ErrorAction SilentlyContinue
    Start-Sleep -Seconds 3
}

Write-Host ">> [1/4] Closing any running CARLA..."
Stop-Carla

Write-Host ">> [2/4] Starting CARLA 0.9.15 (low quality)..."
Start-Process $CARLA_EXE -ArgumentList "/Game/Carla/Maps/$TOWN -quality-level=Low -dx11 -prefernvidia"

Write-Host ">> [3/4] Waiting for CARLA server..."
$ready = $false
for ($i = 0; $i -lt 30; $i++) {
    Start-Sleep -Seconds 5
    & $PY -c "import carla; c=carla.Client('localhost',2000); c.set_timeout(5); print(c.get_server_version())" 2>$null
    if ($LASTEXITCODE -eq 0) { $ready = $true; break }
}
if (-not $ready) { Write-Host "CARLA did not start in time."; Stop-Carla; exit 1 }

# Map launch argument is ignored by the Windows build, so switch the map ONCE here in plain async mode
# (before the leaderboard enables synchronous mode / Traffic Manager), then the leaderboard skips load_world.
Write-Host ">> Switching CARLA map to $TOWN (async, before leaderboard)..."
& $PY -c "import carla,time; c=carla.Client('localhost',2000); c.set_timeout(120); w=c.get_world(); cur=w.get_map().name.split('/')[-1]; print('current map:', cur); (cur=='$TOWN') or c.load_world('$TOWN'); time.sleep(5); print('now:', c.get_world().get_map().name)"
if ($LASTEXITCODE -ne 0) { Write-Host "Map switch failed (CARLA may have crashed)."; Stop-Carla; exit 1 }
Start-Sleep -Seconds 5
Start-Sleep -Seconds 5

Write-Host ">> [4/4] Running route $Id -> $OUT"
& $PY "$env:LEADERBOARD_ROOT\leaderboard\leaderboard_evaluator_local.py" `
    --agent "$env:WORK_DIR\team_code\data_agent.py" `
    --routes $ROUTE `
    --checkpoint $OUT `
    --track MAP 2>&1 | ForEach-Object { "$_" } | Tee-Object -FilePath $LOG

Write-Host ">> Done. Closing CARLA."
Stop-Carla
