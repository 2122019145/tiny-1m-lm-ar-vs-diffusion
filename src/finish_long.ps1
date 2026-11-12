param(
  [string]$Suffix = "_long",
  [int]$PollSeconds = 60
)
$ErrorActionPreference = "Continue"
$root = "D:\dsh1\modeltest"
$py = "$root\.venv\Scripts\python.exe"
$env:PYTHONIOENCODING = "utf-8"
Set-Location "$root\src"

Write-Output "[finish] waiting for train.py processes (suffix $Suffix) ..."
while ($true) {
  $procs = Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
    Where-Object { $_.CommandLine -like "*train.py*" }
  if (-not $procs -or $procs.Count -eq 0) { break }
  Start-Sleep -Seconds $PollSeconds
}
Write-Output "[finish] no train.py left; running evaluation"

& $py evaluate.py `
  --data_dir "$root\data" `
  --ar_ckpt "$root\runs\ar$Suffix\model.pt" `
  --diff_ckpt "$root\runs\diffusion$Suffix\model.pt" `
  --out "$root\out\eval$Suffix.json" --threads 16 2>&1 | Write-Output

Write-Output "[finish] sampling"
& $py sample.py `
  --ar_ckpt "$root\runs\ar$Suffix\model.pt" `
  --diff_ckpt "$root\runs\diffusion$Suffix\model.pt" `
  --out "$root\out\samples$Suffix.md" --n_samples 3 --ar_len 220 --diff_len 220 --diff_steps 64 2>&1 | Write-Output

Write-Output "[finish] plotting"
& $py plot_curves.py --runs "$root\runs" --ar_name "ar$Suffix" --diff_name "diffusion$Suffix" --out "$root\out\curves$Suffix.png" 2>&1 | Write-Output

Write-Output "[finish] DONE"
