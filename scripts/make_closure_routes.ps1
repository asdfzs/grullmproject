# Find straight MIDDLE-lane roads in Town05 and write lane-closure routes with a left / right arrow board.
# Usage:  .\make_closure_routes.ps1          (2 spots = 4 route files)
param([int]$N = 2, [int]$Seed = 3, [string]$Town = "Town05")
$CARLA_EXE = "C:\carla0915\WindowsNoEditor\CarlaUE4.exe"
$env:CARLA_ROOT = "C:\carla0915\WindowsNoEditor"
$env:PYTHONIOENCODING = "utf-8"
$env:WORK_DIR = "C:\carla_garage"
$env:PYTHONPATH = "$env:CARLA_ROOT\PythonAPI\carla"
$PY = "$env:WORK_DIR\venv\Scripts\python.exe"

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

if (-not (Start-CarlaInMap $Town)) { Write-Host "!! CARLA failed"; exit 1 }
& $PY "$env:WORK_DIR\tools\make_closure_routes.py" --n $N --seed $Seed
Stop-Carla
