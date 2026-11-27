$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
python (Join-Path $PSScriptRoot "holo.py") @args
