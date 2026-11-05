@echo off
echo ==========================================================
echo        ECommerceGPT - Push to GitHub
echo ==========================================================
set GIT_PATH="C:\Users\fidel\AppData\Local\Programs\Git\cmd\git.exe"

set /p REPO_URL="Enter your GitHub Repository URL: "
if "%REPO_URL%"=="" (
    echo Error: No repository URL provided.
    pause
    exit /b 1
)

%GIT_PATH% remote remove origin 2>nul
%GIT_PATH% remote add origin %REPO_URL%
%GIT_PATH% branch -M main
echo Pushing to GitHub...
%GIT_PATH% push -u origin main

if %ERRORLEVEL% equ 0 (
    echo.
    echo SUCCESS: Project pushed to GitHub successfully!
) else (
    echo.
    echo Push failed. Please check your credentials or run 'gh auth login'.
)
pause
