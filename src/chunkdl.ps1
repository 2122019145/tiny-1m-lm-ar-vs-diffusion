param(
  [Parameter(Mandatory=$true)][string]$Url,
  [Parameter(Mandatory=$true)][string]$Dest,
  [int]$Chunks = 8
)
$ProgressPreference = 'SilentlyContinue'
# Discover total size
$head = curl.exe -sIL $Url --max-time 60
$len = $null
foreach ($line in $head) {
  if ($line -match '^Content-Length:\s*(\d+)') { $len = [int64]$Matches[1] }
}
if (-not $len) { Write-Output "NOLEN"; exit 1 }
Write-Output "size=$len chunks=$Chunks"

$dir = Split-Path $Dest -Parent
$tmp = Join-Path $dir ("parts_" + [IO.Path]::GetFileName($Dest))
New-Item -ItemType Directory -Force -Path $tmp | Out-Null

$chunkSize = [int64][math]::Ceiling($len / $Chunks)
$jobs = @()
for ($i = 0; $i -lt $Chunks; $i++) {
  $start = $i * $chunkSize
  $end = [math]::Min($start + $chunkSize - 1, $len - 1)
  if ($start -gt $end) { break }
  $part = Join-Path $tmp ("part{0:D3}" -f $i)
  if ((Test-Path $part) -and ((Get-Item $part).Length -eq ($end - $start + 1))) { continue }
  $jobs += Start-Job -ScriptBlock {
    param($u, $s, $e, $p)
    curl.exe -s -r "$s-$e" -o $p $u --max-time 1800 --retry 5 --retry-delay 2
    if (Test-Path $p) { (Get-Item $p).Length } else { 0 }
  } -ArgumentList $Url, $start, $end, $part
}
if ($jobs.Count -gt 0) {
  $null = Wait-Job -Job $jobs -Timeout 1800
  foreach ($j in $jobs) { $null = Receive-Job $j; Remove-Job $j -Force }
}
# Verify and concatenate
$parts = Get-ChildItem $tmp -Filter "part*" | Sort-Object Name
$total = ($parts | Measure-Object Length -Sum).Sum
Write-Output "parts=$($parts.Count) total=$total expected=$len"
if ($total -ne $len) { Write-Output "INCOMPLETE"; exit 2 }

$stage = "$Dest.staging"
if (Test-Path $stage) { Remove-Item $stage -Force }
$out = [IO.File]::Create($stage)
foreach ($p in $parts) {
  $in = [IO.File]::OpenRead($p.FullName)
  $in.CopyTo($out)
  $in.Close()
}
$out.Close()
Move-Item -Path $stage -Destination $Dest -Force
Remove-Item $tmp -Recurse -Force
Write-Output "DONE $Dest $((Get-Item $Dest).Length)"
