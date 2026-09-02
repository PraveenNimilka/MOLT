param(
    [Parameter(Mandatory = $true)]
    [string]$Prompt,
    [int]$MaxNewBytes = 128,
    [double]$Temperature = 0.8,
    [int]$TopK = 40,
    [int]$Seed = 1337
)

$ErrorActionPreference = "Stop"
$engine = "D:\Projects\MOLT\.venv\Scripts\ai-local.exe"
$run = "D:\Projects\MOLT AI\runs\20260901-023504-MOLT-AI-1B-TinyStories-v1-fa3fdd64"

& $engine generate `
    --run $run `
    --prompt $Prompt `
    --max-new-bytes $MaxNewBytes `
    --temperature $Temperature `
    --top-k $TopK `
    --seed $Seed

if ($LASTEXITCODE -ne 0) {
    throw "MOLT AI generation failed with exit code $LASTEXITCODE"
}
