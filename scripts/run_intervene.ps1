# Step 4: intervention test = PDM-Lite + deadlock trigger + LLM advisor (team_code\intervention_agent.py)
#
# Usage:
#   .\run_intervene.ps1 165_broken,165_stalled,165_queue              (rule trigger, stub LLM, 1 run each)
#   .\run_intervene.ps1 165_stalled -Trigger gru                       (GRU trigger, model gru\out_long)
#   .\run_intervene.ps1 165_stalled -Llm claude                        (real LLM; set $env:ANTHROPIC_API_KEY first)
#   .\run_intervene.ps1 165_stalled -Llm gemini                        (free Gemini; set $env:GEMINI_API_KEY first)
#
# What to expect:
#   165_broken / 165_stalled : trigger -> LLM says BYPASS -> PDM-Lite changes lane and finishes the route
#   165_queue                : trigger (maybe) -> LLM says WAIT -> queue drives off -> route finished without bypass
# Logs: results\intervention\<route>_<time>\events.jsonl (+ camera JPEG per LLM call)
param(
    [Parameter(Mandatory=$true)][string[]]$Ids,
    [ValidateSet('rule','gru')][string]$Trigger = 'rule',
    [ValidateSet('stub','claude','gemini')][string]$Llm = 'stub',
    [ValidateSet('full','plain','photo')][string]$LlmInput = 'full',   # full = hints + cue line | plain = raw numbers, no hints | photo = plain + no debris size
    [int]$Reps = 1,
    [double]$TriggerStopSec = 10,
    [string]$GruDir = "C:\carla_garage\gru\out_long_w20s_4hz_final",
    [double]$GruThresh = 0.99,
    [double]$FallbackSec = 12,
    [ValidateSet('perfect','noisy')][string]$Perception = 'noisy',   # noisy = camera-like (no object name, size +-30%)
    [ValidateSet('llm','always_bypass','off')][string]$DebrisPolicy = 'llm',   # off = PDM-Lite alone ignores debris
    [ValidateSet(0,1)][int]$SafeMerge = 0,   # 0 = PDM-Lite handles the lane change itself (same as the vehicle bypass); 1 = our experimental gap check
    [switch]$NoTraffic,
    [double]$MaxStopSec = 150,               # end the run after this long standing still
    [double]$SpeedCap = 0,                   # km/h, 0 = off (e.g. 30 for the prop tests)
    [switch]$PropPhotos,                     # photos of the object ahead at 40/20/10/5 m (events folder)
    [switch]$PropPhysics,                    # debris object can be pushed (movable) instead of an immovable block
    [string]$Suffix = ""   # added to log names, e.g. "_final"
)

$CARLA_EXE = "C:\carla0915\WindowsNoEditor\CarlaUE4.exe"
$env:CARLA_ROOT = "C:\carla0915\WindowsNoEditor"
$env:PYTHONIOENCODING = "utf-8"
$env:WORK_DIR = "C:\carla_garage"
$env:SCENARIO_RUNNER_ROOT = "$env:WORK_DIR\scenario_runner_autopilot"
$env:LEADERBOARD_ROOT = "$env:WORK_DIR\leaderboard_autopilot"
$env:PYTHONPATH = "$env:CARLA_ROOT\PythonAPI\carla;$env:SCENARIO_RUNNER_ROOT;$env:LEADERBOARD_ROOT;$env:WORK_DIR\team_code"
$env:DATAGEN = "0"
$env:MAX_STOP_SEC = "$MaxStopSec"
$env:SPEED_CAP_KMH = "$SpeedCap"
$env:PROP_PHYSICS = if ($PropPhysics) { '1' } else { '0' }
$env:PROP_PHOTOS = if ($PropPhotos) { '1' } else { '0' }                                         # give the bypass time to happen
$env:RECORD_DIR = "$env:WORK_DIR\results\intervention_features"   # keep GRU training data clean
$env:TRIGGER_MODE = $Trigger
$env:TRIGGER_STOP_SEC = "$TriggerStopSec"
$env:GRU_DIR = $GruDir
$env:GRU_THRESH = "$GruThresh"
$env:FALLBACK_STOP_SEC = "$FallbackSec"
$env:PERCEPTION = $Perception
$env:SAFE_MERGE = "$SafeMerge"
$env:NO_BACKGROUND = if ($NoTraffic) { '1' } else { '0' }
$env:DEBRIS = if ($DebrisPolicy -eq 'off') { '0' } else { '1' }
$env:DEBRIS_POLICY = if ($DebrisPolicy -eq 'off') { 'llm' } else { $DebrisPolicy }
$env:END_AFTER_RELEASE = "0"                                      # (may be left over from run_record.ps1)
$env:LLM_MODE = $Llm
$env:LLM_INPUT = $LlmInput
$PY = "$env:WORK_DIR\venv\Scripts\python.exe"
New-Item -ItemType Directory -Force "$env:WORK_DIR\results\intervention_logs" | Out-Null

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
    $env:RECORD_TAG = "${Id}_$Trigger"
    $env:RECORD_CLASS = if ($Id -match 'broken|stalled') { 'P' } else { 'N' }

    for ($r = 1; $r -le $Reps; $r++) {
        Write-Host ""
        Write-Host "######## INTERVENE $Id  (trigger=$Trigger, llm=$Llm)  run $r / $Reps ########"
        if (-not (Start-CarlaInMap $TOWN)) { $summary += "$Id run$r : CARLA FAILED"; continue }
        $TAG = "${Id}_${Trigger}_${Llm}${Suffix}_run$r"
        $env:RECORD_TAG = "${Id}_${Trigger}${Suffix}_run$r"
        & $PY "$env:LEADERBOARD_ROOT\leaderboard\leaderboard_evaluator_local.py" `
            --agent "$env:WORK_DIR\team_code\intervention_agent.py" `
            --routes $ROUTE `
            --checkpoint "$env:WORK_DIR\results\intervention_logs\$TAG.json" `
            --track MAP 2>&1 | ForEach-Object { "$_" } | Tee-Object -FilePath "$env:WORK_DIR\results\intervention_logs\$TAG.log.txt"
        $summary += $TAG
        Stop-Carla
    }
}

Write-Host ""
Write-Host "================ SUMMARY ================"
& $PY "$env:WORK_DIR\tools\summarize_intervention.py" $summary
