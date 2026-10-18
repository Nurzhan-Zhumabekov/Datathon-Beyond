# Windows: starts the model server (llama.cpp), the service, and optionally a public tunnel.
# usage: powershell -ExecutionPolicy Bypass -File start.ps1        (Ctrl+C stops everything)
#
# Needs:  .venv with the dependencies of README.md (without mlx-vlm),
#         llama-server.exe (set LLAMA_SERVER, or put it in PATH),
#         models\Qwen3VL-4B-Instruct-Q4_K_M.gguf and models\mmproj-Qwen3VL-4B-Instruct-F16.gguf
#         (scripts\download_gguf.ps1 downloads them once; after that no internet is needed).
# While this script runs it asks Windows not to go to sleep (like caffeinate on the Mac); this does not change the
# power settings and ends with the script. Closing the laptop lid still follows your power settings.
# The script also watches the model server and the service and restarts either one if it exits.
# The tunnel is OFF by default: the Mac may be serving the fixed ngrok domain, and two machines must not use it
# at the same time. Set START_NGROK=1 only when the Mac service is stopped.
# START_CLOUDFLARED=1 opens a Cloudflare quick tunnel instead (tools\cloudflared.exe or cloudflared in PATH): a
# random https://*.trycloudflare.com address, written to data\public_url.txt; it changes if the tunnel restarts,
# so a tunnel that exits is only reported, never restarted silently.
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
New-Item -ItemType Directory -Force data | Out-Null

Add-Type -Namespace Win -Name Power -MemberDefinition '[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint f);'
# ES_CONTINUOUS | ES_SYSTEM_REQUIRED = 0x80000001; as a decimal UInt32, because PowerShell 5.1 reads the hex literal as a negative Int32
[void][Win.Power]::SetThreadExecutionState([uint32]2147483649)

$llama  = if ($env:LLAMA_SERVER) { $env:LLAMA_SERVER } else { "llama-server" }
$model  = if ($env:GGUF_MODEL)   { $env:GGUF_MODEL }   else { "models\Qwen3VL-4B-Instruct-Q4_K_M.gguf" }
$mmproj = if ($env:GGUF_MMPROJ)  { $env:GGUF_MMPROJ }  else { "models\mmproj-Qwen3VL-4B-Instruct-F16.gguf" }
$env:VLM_BACKEND = "openai"
if (-not $env:VLM_URL) { $env:VLM_URL = "http://127.0.0.1:8081" }

function Wait-Http($url, $seconds) {
    for ($i = 0; $i -lt $seconds; $i++) {
        try { if ((Invoke-WebRequest $url -UseBasicParsing -TimeoutSec 2).StatusCode -eq 200) { return $true } } catch { }
        Start-Sleep 1
    }
    return $false
}

function Start-Model {
    Start-Process -FilePath $llama -PassThru -WindowStyle Hidden `
        -RedirectStandardOutput data\llama.out.log -RedirectStandardError data\llama.err.log `
        -ArgumentList @("-m", $model, "--mmproj", $mmproj, "--host", "127.0.0.1", "--port", "8081",
                        "-ngl", "99", "-c", "8192", "--alias", "qwen3-vl-4b")
}

function Start-BeyondService {
    Start-Process -FilePath ".venv\Scripts\python.exe" -PassThru -WindowStyle Hidden `
        -RedirectStandardOutput data\server.log -RedirectStandardError data\server.err.log `
        -ArgumentList @("-m", "uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", "8000", "--log-level", "warning")
}

$procs = @()
try {
    $modelProc = Start-Model
    $procs += $modelProc
    if (-not (Wait-Http "http://127.0.0.1:8081/health" 180)) { throw "llama-server did not start, see data\llama.err.log" }
    Write-Host "Model server is up on http://127.0.0.1:8081"

    $serviceProc = Start-BeyondService
    $procs += $serviceProc
    if (-not (Wait-Http "http://127.0.0.1:8000/health" 60)) { throw "service did not start, see data\server.err.log" }
    Write-Host "Service is up on http://127.0.0.1:8000"
    Write-Host "API key: see .env (API_KEY=...)"

    if ($env:START_NGROK -eq "1" -and (Get-Command ngrok -ErrorAction SilentlyContinue)) {
        $procs += Start-Process -FilePath "ngrok" -PassThru -WindowStyle Hidden `
            -RedirectStandardOutput data\ngrok.log -ArgumentList @("http", "8000", "--log", "stdout")
        Start-Sleep 5
        $t = (Invoke-RestMethod http://127.0.0.1:4040/api/tunnels).tunnels
        if ($t) { $t[0].public_url | Set-Content data\public_url.txt; Write-Host "PUBLIC URL: $($t[0].public_url)" }
    }

    $tunnelProc = $null
    if ($env:START_CLOUDFLARED -eq "1") {
        $cf = if (Test-Path "tools\cloudflared.exe") { "tools\cloudflared.exe" } else { "cloudflared" }
        Remove-Item data\cloudflared.err.log, data\cloudflared.out.log -ErrorAction SilentlyContinue
        $tunnelProc = Start-Process -FilePath $cf -PassThru -WindowStyle Hidden `
            -RedirectStandardOutput data\cloudflared.out.log -RedirectStandardError data\cloudflared.err.log `
            -ArgumentList @("tunnel", "--no-autoupdate", "--url", "http://127.0.0.1:8000")
        $procs += $tunnelProc
        $url = ""
        for ($i = 0; $i -lt 60 -and -not $url; $i++) {
            Start-Sleep 1
            $m = Select-String -Path data\cloudflared.err.log, data\cloudflared.out.log -Pattern 'https://[a-z0-9-]+\.trycloudflare\.com' -ErrorAction SilentlyContinue | Select-Object -First 1
            if ($m) { $url = [regex]::Match($m.Line, 'https://[a-z0-9-]+\.trycloudflare\.com').Value }
        }
        if ($url) { $url | Set-Content data\public_url.txt; Write-Host "PUBLIC URL (cloudflare, changes on restart): $url" }
        else { Write-Host "Cloudflare tunnel did not report an address, see data\cloudflared.err.log" }
    }

    $tunnelReported = $false
    while ($true) {
        Start-Sleep 5
        if ($modelProc.HasExited) {
            Write-Host "$(Get-Date -Format s) llama-server exited, restarting"
            $modelProc = Start-Model
            $procs += $modelProc
            [void](Wait-Http "http://127.0.0.1:8081/health" 180)
        }
        if ($serviceProc.HasExited) {
            Write-Host "$(Get-Date -Format s) service exited, restarting"
            $serviceProc = Start-BeyondService
            $procs += $serviceProc
        }
        if ($tunnelProc -and $tunnelProc.HasExited -and -not $tunnelReported) {
            Write-Host "$(Get-Date -Format s) cloudflared exited: the public address no longer works, restart start.ps1 to get a new one"
            $tunnelReported = $true
        }
    }
}
finally {
    $procs | ForEach-Object { if ($_ -and -not $_.HasExited) { Stop-Process -Id $_.Id -Force } }
}
