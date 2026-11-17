<#
Create the local git repo with YOUR identity only and push it to GitHub.

  powershell -ExecutionPolicy Bypass -File .\publish_github.ps1 -Name "Your Name" -Email "you@users.noreply.github.com" -RepoUrl https://github.com/<you>/visionops.git

- Identity is set for THIS repository only (git config --local).
- Commit messages contain no co-author or tool attribution lines.
- First create an EMPTY public repo on github.com (no README/license), then pass its URL.
  On the first push, Git for Windows opens a browser window to sign in to GitHub.
#>
param(
    [Parameter(Mandatory = $true)][string]$Name,
    [Parameter(Mandatory = $true)][string]$Email,
    [string]$RepoUrl = "",
    [string]$Message = "VisionOps: PCB defect detection, failure analysis and data curation pipeline"
)
$ErrorActionPreference = "Continue"
Set-Location $PSScriptRoot
if (-not (Get-Command git -ErrorAction SilentlyContinue)) { throw "git not found. Install Git for Windows: https://git-scm.com/download/win" }

if (-not (Test-Path ".git")) { git init -b main | Out-Host }
git config --local user.name $Name
git config --local user.email $Email

# safety: never commit secrets or data
foreach ($f in @(".env")) { if (Test-Path $f) { git rm --cached --quiet --ignore-unmatch $f 2>$null } }
git add -A
$staged = git diff --cached --name-only
if ($staged | Select-String -Pattern "(^|/)\.env$|\.pt$|\.onnx$|^data/|^runs/") { throw "Refusing: secrets/data/weights are staged. Check .gitignore." }
if ($staged) { git commit -m $Message | Out-Host; if ($LASTEXITCODE -ne 0) { throw "git commit failed" } } else { Write-Host "Nothing new to commit." }

git log -1 --format="Author: %an <%ae>%nCommitter: %cn <%ce>%n%n%B" | Out-Host

if ($RepoUrl) {
    if (git remote | Select-String -Pattern "^origin$" -Quiet) { git remote set-url origin $RepoUrl } else { git remote add origin $RepoUrl }
    git push -u origin main | Out-Host
    if ($LASTEXITCODE -ne 0) { throw "git push failed (see message above)." }
    Write-Host "Pushed to $RepoUrl" -ForegroundColor Green
}
