# Wait until both training processes exit, then run the final pipeline.
$ErrorActionPreference = "Continue"
$root = "D:\dsh1\modeltest"
$py = "$root\.venv\Scripts\python.exe"
$env:PYTHONIOENCODING = "utf-8"
Set-Location "$root\src"

Write-Output "[wait] polling for training processes..."
while ($true) {
  $n = @(Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
         Where-Object { $_.CommandLine -like "*train.py*" }).Count
  if ($n -eq 0) { break }
  Start-Sleep -Seconds 20
}
Write-Output "[wait] no train.py processes left"

Write-Output "=== tail AR ==="
Get-Content "$root\runs\ar.log" -Tail 4
Write-Output "=== tail DIFF ==="
Get-Content "$root\runs\diffusion.log" -Tail 6

Write-Output "=== evaluate ==="
& $py evaluate.py --out "$root\out\eval.json" 2>&1 | Tee-Object -FilePath "$root\out\eval.log"

Write-Output "=== sample ==="
& $py sample.py --out "$root\out\samples.md" 2>&1 | Tee-Object -FilePath "$root\out\sample.log"

Write-Output "=== plot ==="
& $py plot_curves.py --runs "$root\runs" --out "$root\out\curves.png" 2>&1

Write-Output "=== PIPELINE DONE ==="
