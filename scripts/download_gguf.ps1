# Downloads Qwen3-VL weights for llama.cpp (once).
#   -Size 4b (default): Qwen3VL-4B-Instruct-Q4_K_M.gguf ~2.50 GB + mmproj-Qwen3VL-4B-Instruct-F16.gguf ~0.84 GB
#   -Size 8b:           Qwen3VL-8B-Instruct-Q4_K_M.gguf ~5.03 GB + mmproj-Qwen3VL-8B-Instruct-F16.gguf ~1.16 GB
# Source: https://huggingface.co/Qwen/Qwen3-VL-<size>-Instruct-GGUF (Apache-2.0).
# usage: powershell -ExecutionPolicy Bypass -File scripts\download_gguf.ps1 [-Size 8b]
param([ValidateSet("4b", "8b")][string]$Size = "4b")
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")
New-Item -ItemType Directory -Force models | Out-Null
$s = $Size.ToUpper()
$base = "https://huggingface.co/Qwen/Qwen3-VL-$s-Instruct-GGUF/resolve/main"
foreach ($f in @("Qwen3VL-$s-Instruct-Q4_K_M.gguf", "mmproj-Qwen3VL-$s-Instruct-F16.gguf")) {
    # -C - resumes an interrupted download
    curl.exe -L -C - --fail -o "models\$f" "$base/$f"
    if ($LASTEXITCODE -ne 0) { throw "download of $f failed (curl exit code $LASTEXITCODE)" }
}
Get-ChildItem models\*.gguf | Select-Object Name, @{n = "MB"; e = { [int]($_.Length / 1MB) }}
