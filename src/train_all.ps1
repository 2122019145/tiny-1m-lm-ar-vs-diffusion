param(
  [int]$Steps = 4000,
  [int]$Batch = 32,
  [int]$SeqLen = 256,
  [int]$Threads = 8,
  [int]$LR = 3000,      # micro-lr: 3000 => 0.003 (divided by 1e6 below)
  [string]$LRMin = "0.0003",
  [int]$Warmup = 250,
  [int]$EvalEvery = 250,
  [int]$CkptEvery = 500,
  [string]$Suffix = "",  # appended to run dir, e.g. _long
  [switch]$Resume
)
$ErrorActionPreference = "Continue"
$root = "D:\dsh1\modeltest"
$py = "$root\.venv\Scripts\python.exe"
$env:PYTHONIOENCODING = "utf-8"
Set-Location "$root\src"

# 3000 -> 0.003 ; keep 6 decimals so the value survives the CLI round trip
$lrStr = ([double]$LR / 1e6).ToString("0.000000", [System.Globalization.CultureInfo]::InvariantCulture)
Write-Output "peak lr = $lrStr  lr_min = $LRMin"

$common = @(
  "--data_dir", "$root\data",
  "--seq_len", "$SeqLen",
  "--batch_size", "$Batch",
  "--eval_batch", "8",
  "--n_layer", "5", "--n_head", "4", "--d_model", "112", "--d_ff", "320",
  "--lr", $lrStr, "--lr_min", $LRMin,
  "--warmup", "$Warmup",
  "--max_steps", "$Steps",
  "--log_every", "50",
  "--eval_every", "$EvalEvery",
  "--ckpt_every", "$CkptEvery",
  "--threads", "$Threads"
)
if ($Resume) { $common += "--resume" }

$arDir = "$root\runs\ar$Suffix"
$dfDir = "$root\runs\diffusion$Suffix"
New-Item -ItemType Directory -Force -Path $arDir, $dfDir | Out-Null

$tokensPerStep = $Batch * $SeqLen
Write-Output "=== AR + diffusion | $Steps steps | $tokensPerStep tok/step | $($Steps * $tokensPerStep) tokens | $Threads threads each ==="

$arJob = Start-Process -FilePath $py -ArgumentList (@("train.py", "--arch", "ar", "--run_dir", $arDir) + $common) `
  -RedirectStandardOutput "$root\runs\ar$Suffix.log" -RedirectStandardError "$root\runs\ar$Suffix.err" `
  -NoNewWindow -PassThru

$dfJob = Start-Process -FilePath $py -ArgumentList (@("train.py", "--arch", "diffusion", "--run_dir", $dfDir) + $common) `
  -RedirectStandardOutput "$root\runs\diffusion$Suffix.log" -RedirectStandardError "$root\runs\diffusion$Suffix.err" `
  -NoNewWindow -PassThru

Write-Output "AR pid=$($arJob.Id)  Diffusion pid=$($dfJob.Id)"
$arJob.WaitForExit()
$dfJob.WaitForExit()
Write-Output "=== both finished: AR exit=$($arJob.ExitCode) Diff exit=$($dfJob.ExitCode) ==="
