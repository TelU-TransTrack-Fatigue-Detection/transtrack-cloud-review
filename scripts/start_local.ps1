# start_local.ps1
# Launches all TransTrack local dev services in separate windows.
# Usage:
#   .\scripts\start_local.ps1                # normal mode (callback returns 200)
#   .\scripts\start_local.ps1 -MockMode 500  # callback returns 500 (retry test)
#   .\scripts\start_local.ps1 -Stop          # kill all services

param(
    [string]$MockMode = "200",
    [switch]$Stop
)

$Root    = Split-Path $PSScriptRoot -Parent
$Python  = "C:\Users\lulay\.conda\envs\transtrack_test\python.exe"
$Celery  = "C:\Users\lulay\.conda\envs\transtrack_test\Scripts\celery.exe"

if ($Stop) {
    Write-Host "Stopping all TransTrack services..." -ForegroundColor Yellow
    foreach ($port in @(8000, 8001, 8002, 5555)) {
        $pids = netstat -ano | Select-String ":$port\s" | Select-String "LISTENING" |
                ForEach-Object { ($_.ToString().Trim() -split '\s+')[-1] }
        foreach ($p in $pids) {
            if ($p -and $p -ne "0") {
                taskkill /PID $p /F 2>$null | Out-Null
                Write-Host "  Killed PID $p on port $port"
            }
        }
    }
    Write-Host "Done." -ForegroundColor Green
    exit 0
}

# Verify conda env exists
if (-not (Test-Path $Python)) {
    Write-Error "Python not found at $Python — is transtrack_test conda env created?"
    exit 1
}

Write-Host "Starting Redis via Docker..." -ForegroundColor Cyan
docker run -d --name transtrack-redis -p 6379:6379 redis:7-alpine 2>$null
if ($LASTEXITCODE -ne 0) {
    # Already running or Docker not available — check if redis is up
    $redisCheck = docker ps --filter "name=transtrack-redis" --format "{{.Names}}" 2>$null
    if ($redisCheck -ne "transtrack-redis") {
        Write-Warning "Redis container not running. Start it manually or install Redis."
    } else {
        Write-Host "  Redis already running." -ForegroundColor Green
    }
}

Start-Sleep -Seconds 2

Write-Host "Starting TransTrack local services (MOCK_MODE=$MockMode)..." -ForegroundColor Cyan
Write-Host "  API      -> http://localhost:8000"
Write-Host "  Callback -> http://localhost:8001  (mock, returns $MockMode)"
Write-Host "  Video    -> http://localhost:8002"
Write-Host ""

function Start-Service($title, $cmd) {
    Start-Process powershell -ArgumentList "-NoExit", "-Command",
        "cd '$Root'; `$host.UI.RawUI.WindowTitle = 'transtrack: $title'; $cmd"
}

# 1. Mock video server
Start-Service "video-server" "& '$Python' mock/video_server.py"

# 2. Mock callback server
Start-Service "callback-server" "`$env:MOCK_MODE='$MockMode'; & '$Python' mock/callback_server.py"

# 3. Celery worker
Start-Service "worker" "`$env:CALLBACK_URL='http://localhost:8001/callback'; & '$Celery' -A app.worker.celery_app worker --pool=solo --loglevel=info"

# 4. FastAPI
Start-Service "api" "& '$Python' -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000"

Write-Host "All services launched in separate windows." -ForegroundColor Green
Write-Host ""
Write-Host "Send a test alarm (copy-paste this):" -ForegroundColor Yellow
Write-Host @'
Invoke-RestMethod -Method Post -Uri http://localhost:8000/review -ContentType "application/json" -Body '{"id":"TEST-001","imei":"TEST001","time":"2025-01-10T02:00:00Z","alarm":"fatigue","dms_video_url":"http://localhost:8002/videos/test.mp4"}'
'@
Write-Host ""
Write-Host "To stop everything:  .\scripts\start_local.ps1 -Stop"
