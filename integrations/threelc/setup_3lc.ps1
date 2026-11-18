<#
Create a SEPARATE virtual environment for the 3LC integration so its pinned
Ultralytics version cannot change the main .venv.

  powershell -ExecutionPolicy Bypass -File .\integrations\threelc\setup_3lc.ps1
#>
param([string]$TorchIndex = "https://download.pytorch.org/whl/cu128")
$ErrorActionPreference = "Stop"
Set-Location (Resolve-Path "$PSScriptRoot\..\..")

$Py = ".\.venv-3lc\Scripts\python.exe"
if (-not (Test-Path $Py)) {
    $created = $false
    foreach ($v in @("-3.12", "-3.11", "-3.13")) {
        $ErrorActionPreference = "Continue"
        & py $v -m venv .venv-3lc 2>$null
        $ok = ($LASTEXITCODE -eq 0)
        $ErrorActionPreference = "Stop"
        if ($ok) { $created = $true; break }
    }
    if (-not $created) { python -m venv .venv-3lc }
}
& $Py -m pip install --upgrade pip
& $Py -m pip install torch torchvision --index-url $TorchIndex
& $Py -m pip install 3lc-ultralytics --extra-index-url https://pypi.3lc.ai/public/repositories/releases-public
if ($LASTEXITCODE -ne 0) { throw "3lc-ultralytics install failed" }
& $Py -m pip install pandas pyyaml pillow
& $Py -c "import tlc, tlc_ultralytics, torch; print('3lc', tlc.__version__, '| CUDA', torch.cuda.is_available())"
Write-Host "`nNext: .\.venv-3lc\Scripts\3lc login <YOUR_3LC_API_KEY>" -ForegroundColor Green
