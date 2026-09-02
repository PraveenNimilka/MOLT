param(
    [Parameter(Mandatory = $true)]
    [ValidateSet("story", "qa")]
    [string]$Mode,
    [Parameter(Mandatory = $true)]
    [string]$Prompt,
    [int]$MaxNewTokens = 0,
    [double]$Temperature = -1,
    [int]$TopK = 40,
    [int]$Seed = 1337
)

$ErrorActionPreference = "Stop"
$engine = "D:\Projects\MOLT\.venv\Scripts\ai-local.exe"
$baseRun = "D:\Projects\MOLT AI\50M\runs\20260901-093650-MOLT-AI-50M-BPE-pretrain-v1-68ba7279"
$qaRun = "D:\Projects\MOLT AI\50M\runs\20260901-094628-MOLT-AI-50M-BPE-QA-masked-v2-1d782a20"

if ($Mode -eq "qa") {
    $run = $qaRun
    $actualPrompt = "<|question|>$Prompt<|answer|>"
    if ($MaxNewTokens -le 0) { $MaxNewTokens = 40 }
    if ($Temperature -lt 0) { $Temperature = 0 }
} else {
    $run = $baseRun
    $actualPrompt = $Prompt
    if ($MaxNewTokens -le 0) { $MaxNewTokens = 120 }
    if ($Temperature -lt 0) { $Temperature = 0.8 }
}

& $engine generate `
    --run $run `
    --prompt $actualPrompt `
    --max-new-tokens $MaxNewTokens `
    --temperature $Temperature `
    --top-k $TopK `
    --seed $Seed

if ($LASTEXITCODE -ne 0) {
    throw "MOLT AI 50M generation failed with exit code $LASTEXITCODE"
}
