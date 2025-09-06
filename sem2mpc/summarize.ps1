$rows = @()
Get-ChildItem -Recurse -Filter *_metrics.json .\exp | ForEach-Object {
  $m = Get-Content $_.FullName -Raw | ConvertFrom-Json
  $case = Split-Path $_.DirectoryName -Leaf
  $kind = if ($_.BaseName -like '*_llm_metrics') { 'LLM' } else { 'Manual' }
  $rows += [pscustomobject]@{
    Case   = $case
    Kind   = $kind
    N      = $m.N
    EndErr = '{0:N4}' -f $m.end_position_error
    MinObs = if ($m.min_obstacle_distance) { '{0:N4}' -f $m.min_obstacle_distance } else { '' }
    SolveS = '{0:N3}' -f $m.solve_time_sec
  }
}

$rows | Sort-Object Case,Kind | Format-Table -AutoSize
$rows | ConvertTo-Csv -NoTypeInformation | Set-Content -Encoding utf8NoBOM .\exp\summary.csv
Write-Host "✅ Saved exp\summary.csv"
