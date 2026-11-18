<#
VisionOps end-to-end pipeline for Windows (PowerShell 5.1+).

  powershell -ExecutionPolicy Bypass -File .\run_pipeline.ps1
  powershell -ExecutionPolicy Bypass -File .\run_pipeline.ps1 -DatasetZip C:\Downloads\pcb.zip
  powershell -ExecutionPolicy Bypass -File .\run_pipeline.ps1 -SkipExperiment

Resumable: every step is skipped if its output already exists, so re-running after
a failure continues where it stopped. Training settings live in configs/baseline.yaml.
Log: logs\pipeline.log   Current step: logs\status.txt
#>
param(
    [string]$DatasetZip = "",
    [switch]$SkipExperiment,
    [switch]$SkipOnnx,
    [string]$TorchIndex = "",          # e.g. https://download.pytorch.org/whl/cu126 ; default: cu128 if NVIDIA GPU else cpu
    [double]$CurationFrac = 0.05
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
New-Item -ItemType Directory -Force -Path logs | Out-Null
$Log = Join-Path $PSScriptRoot "logs\pipeline.log"
$Status = Join-Path $PSScriptRoot "logs\status.txt"
$env:PYTHONUNBUFFERED = "1"
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"
$env:PIN_MEMORY = "False"   # pinned host memory ran out on a low-RAM PC

function Write-Status($msg) {
    $line = "$(Get-Date -Format s) $msg"
    Set-Content -Path $Status -Value $line -Encoding UTF8
    Add-Content -Path $Log -Value "=== $line" -Encoding UTF8
    Write-Host "=== $msg" -ForegroundColor Cyan
}

function Invoke-Native([string]$Name, [string]$Exe, [string[]]$Arguments) {
    Write-Status "RUNNING $Name"
    $old = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    & $Exe @Arguments 2>&1 | ForEach-Object {
        $l = "$_"
        Write-Host $l
        Add-Content -Path $Log -Value $l -Encoding UTF8
    }
    $code = $LASTEXITCODE
    $ErrorActionPreference = $old
    if ($code -ne 0) {
        Write-Status "FAILED $Name (exit $code)"
        throw "Step '$Name' failed with exit code $code. See $Log"
    }
    Write-Status "DONE $Name"
}

# ---------- 1. Python + venv ----------
$Py = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $Py)) {
    $base = $null
    $ErrorActionPreference = "Continue"
    if (Get-Command py -ErrorAction SilentlyContinue) {
        foreach ($v in @("-3.12", "-3.11", "-3.13", "-3.10")) {
            & py $v -c "import sys" 2>$null
            if ($LASTEXITCODE -eq 0) { $base = @("py", $v); break }
        }
    }
    $ErrorActionPreference = "Stop"
    if (-not $base -and (Get-Command python -ErrorAction SilentlyContinue)) { $base = @("python") }
    if (-not $base) { throw "Python not found. Install Python 3.10-3.12 from python.org (tick 'Add to PATH')." }
    $exe = $base[0]
    $pre = @()
    if ($base.Count -gt 1) { $pre = $base[1..($base.Count - 1)] }
    Invoke-Native "create venv" $exe ($pre + @("-m", "venv", ".venv"))
}
Invoke-Native "python version" $Py @("-c", "import sys; print(sys.version); assert sys.version_info < (3, 14), 'Use Python 3.10-3.13 for PyTorch wheels'")

# ---------- 2. Dependencies ----------
$torchOk = $false
$ErrorActionPreference = "Continue"
& $Py -c "import torch, ultralytics, streamlit, fastapi, roboflow, onnx" 2>$null
if ($LASTEXITCODE -eq 0) { $torchOk = $true }
$ErrorActionPreference = "Stop"
if (-not $torchOk) {
    Invoke-Native "pip upgrade" $Py @("-m", "pip", "install", "--upgrade", "pip")
    if (-not $TorchIndex) {
        if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
            $TorchIndex = "https://download.pytorch.org/whl/cu128"
        } else {
            $TorchIndex = "https://download.pytorch.org/whl/cpu"
            Write-Host "nvidia-smi not found: installing CPU PyTorch (training will be slow)." -ForegroundColor Yellow
        }
    }
    Invoke-Native "install torch ($TorchIndex)" $Py @("-m", "pip", "install", "torch", "torchvision", "--index-url", $TorchIndex)
    Invoke-Native "install requirements" $Py @("-m", "pip", "install", "-r", "requirements.txt")
}
Invoke-Native "check GPU" $Py @("-c", "import torch; print('torch', torch.__version__, '| CUDA available:', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')")

# ---------- 3. Unit tests ----------
Invoke-Native "unit tests" $Py @("-m", "pytest", "-q")

# ---------- 4. Dataset ----------
if (-not (Test-Path "data\raw\roboflow\data.yaml")) {
    if ($DatasetZip) {
        Invoke-Native "extract dataset zip" $Py @("-m", "visionops.data.download", "--zip", $DatasetZip)
    } else {
        $hasKey = (Test-Path ".env") -and (Select-String -Path ".env" -Pattern "^ROBOFLOW_API_KEY=\S+" -Quiet)
        if (-not $hasKey) {
            Write-Status "BLOCKED dataset: no ROBOFLOW_API_KEY in .env and no -DatasetZip"
            Write-Host @"

Dataset needed. Either:
  1) Copy .env.example to .env and set ROBOFLOW_API_KEY (free: app.roboflow.com -> Settings -> API Keys), or
  2) Download the YOLOv8 zip from https://universe.roboflow.com/pcbdataset/pcb-defect-detection-9ewqw
     and run:  powershell -ExecutionPolicy Bypass -File .\run_pipeline.ps1 -DatasetZip <path-to-zip>
"@ -ForegroundColor Yellow
            exit 2
        }
        Invoke-Native "download dataset" $Py @("-m", "visionops.data.download")
    }
}
# Always (re)build split lists: cheap and deterministic. The provided test split (31 images) misses 2 of 6
# classes, so provided valid+test become the held-out test set and validation is carved from train (grouped).
Invoke-Native "prepare splits" $Py @("-m", "visionops.data.prepare", "--merge-val-into-test", "--val-frac", "0.1")
Invoke-Native "validate dataset" $Py @("-m", "visionops.data.validate")

# ---------- 5. Baseline ----------
if (-not (Test-Path "runs\baseline\weights\best.pt")) {
    Invoke-Native "train baseline" $Py @("-m", "visionops.train", "--config", "configs/baseline.yaml", "--name", "baseline")
}
if (-not (Test-Path "reports\baseline\metrics.json")) {
    Invoke-Native "evaluate baseline on test" $Py @("-m", "visionops.evaluate", "--weights", "runs/baseline/weights/best.pt", "--name", "baseline")
}

if (-not (Test-Path "reports\threshold\threshold.json")) {
    Invoke-Native "tune confidence threshold on validation" $Py @("-m", "visionops.threshold", "--weights", "runs/baseline/weights/best.pt", "--name", "baseline")
}

# ---------- 6. Review queue ----------
if (-not (Test-Path "reports\review\review_queue.csv")) {
    Invoke-Native "rank training samples for review" $Py @("-m", "visionops.review", "--weights", "runs/baseline/weights/best.pt")
}

# ---------- 7. Curation experiment ----------
if (-not $SkipExperiment) {
    if (-not (Test-Path "reports\experiments\manifest.json")) {
        Invoke-Native "build curation variants" $Py @("-m", "visionops.experiment", "build", "--frac", "$CurationFrac")
    }
    Invoke-Native "run curation experiment" $Py @("-m", "visionops.experiment", "run")
}

# ---------- 8. ONNX ----------
if (-not $SkipOnnx -and -not (Test-Path "reports\onnx_consistency.json")) {
    Invoke-Native "export ONNX + consistency check" $Py @("-m", "visionops.export_onnx", "--weights", "runs/baseline/weights/best.pt")
}

# ---------- 9. Results ----------
Invoke-Native "render README figures" $Py @("-m", "visionops.figures")
Invoke-Native "build results report" $Py @("-m", "visionops.report")
Write-Status "PIPELINE COMPLETE"
Write-Host "`nDashboard:  .\.venv\Scripts\streamlit run app\dashboard.py" -ForegroundColor Green
Write-Host "API:        .\.venv\Scripts\uvicorn api.main:app --port 8000" -ForegroundColor Green
