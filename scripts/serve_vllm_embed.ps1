<#
.SYNOPSIS
Start multilingual-e5-large as a vLLM OpenAI-compatible embeddings server.

.DESCRIPTION
The vLLM image and the weights already live on G:, so this only drives the WSL
docker daemon that owns the `wsl-g-drive` context:

    docker context use wsl-g-drive        # Docker CE in WSL Ubuntu-24.04
    dasboard: wsl -d Ubuntu-24.04 -e bash -lc "systemctl is-active docker"

Embeddings are served at http://127.0.0.1:18000/v1 (OpenAI schema), which is what
`RAGV2_EMBED_BACKEND=vllm` points at.

.EXAMPLE
powershell -File agentic_rag_v2/scripts/serve_vllm_embed.ps1 -Detach
powershell -File agentic_rag_v2/scripts/serve_vllm_embed.ps1 -Stop
#>
param(
    [int]$Port = 18000,
    [string]$Name = "vllm-e5",
    [switch]$Detach,
    [switch]$Stop
)

$ErrorActionPreference = "Stop"
$distro = "Ubuntu-24.04"

function Invoke-Wsl([string]$command) {
    & wsl -d $distro -e bash -lc $command
    if ($LASTEXITCODE -ne 0) { throw "wsl command failed: $command" }
}

if ($Stop) {
    Invoke-Wsl "docker rm -f $Name >/dev/null 2>&1 || true"
    Write-Host "stopped $Name"
    return
}

# make sure the daemon that owns the G: vhdx is up
$active = (& wsl -d $distro -e bash -lc "systemctl is-active docker" 2>&1) -join ""
if ($active.Trim() -ne "active") {
    Write-Host "starting docker in WSL..."
    Invoke-Wsl "systemctl start docker"
}

$args = "--detach --port $Port --model /path/to/models/multilingual-e5-large"
$script = "/path/to/repo/scripts/serve_vllm_embed.sh"
$name = $Name
if ($Detach) {
    Invoke-Wsl "VLLM_NAME=$name bash $script $args"
} else {
    Write-Host "running in foreground (Ctrl+C to stop); use -Detach for background"
    Invoke-Wsl "VLLM_NAME=$name bash $script $args"
}
