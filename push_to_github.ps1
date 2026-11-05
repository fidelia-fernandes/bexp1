# push_to_github.ps1 - Helper to push ECommerceGPT to GitHub
param (
    [string]$RepoUrl
)

$env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path","User")
$gitExe = "C:\Users\fidel\AppData\Local\Programs\Git\cmd\git.exe"

if (-not (Test-Path $gitExe)) {
    $gitExe = "git"
}

if (-not $RepoUrl) {
    Write-Host "==========================================================" -ForegroundColor Cyan
    Write-Host "     ECommerceGPT - Push to GitHub Setup" -ForegroundColor Cyan
    Write-Host "==========================================================" -ForegroundColor Cyan
    Write-Host ""
    $RepoUrl = Read-Host "Enter your GitHub Repository URL (e.g. https://github.com/<username>/<repo>.git)"
}

if (-not $RepoUrl) {
    Write-Host "Error: No repository URL provided." -ForegroundColor Red
    exit 1
}

Write-Host "Configuring remote origin: $RepoUrl" -ForegroundColor Yellow
& $gitExe remote remove origin 2>$null
& $gitExe remote add origin $RepoUrl

Write-Host "Setting main branch..." -ForegroundColor Yellow
& $gitExe branch -M main

Write-Host "Pushing to GitHub..." -ForegroundColor Green
& $gitExe push -u origin main

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "SUCCESS: Repository pushed to GitHub successfully!" -ForegroundColor Green
} else {
    Write-Host ""
    Write-Host "Push failed. If this is a private repo or requires authentication, please sign in via browser prompt or GitHub CLI ('gh auth login')." -ForegroundColor Yellow
}
