@echo off
REM LoC-LIC Windows Desktop App — Build Script
REM
REM Prerequisites:
REM   pip install pyinstaller PyQt6 torch torchvision compressai Pillow
REM
REM Run from:  Optimization\Windows Desktop App\
REM   build\build_windows.bat

setlocal EnableDelayedExpansion

set "APP_DIR=%~dp0.."
set "DIST_DIR=%APP_DIR%\dist"
set "BUILD_DIR=%APP_DIR%\build_tmp"
set "PYTHON=python"

echo.
echo  LoC-LIC Compressor ^| Windows Build
echo  ====================================

REM Check Python
%PYTHON% --version >nul 2>&1
if errorlevel 1 (
    echo  [error] Python not found. Add it to PATH.
    exit /b 1
)

REM Check PyInstaller
%PYTHON% -m PyInstaller --version >nul 2>&1
if errorlevel 1 (
    echo  [error] PyInstaller not found.
    echo  Install: pip install pyinstaller
    exit /b 1
)

REM Check PyQt6
%PYTHON% -c "import PyQt6" >nul 2>&1
if errorlevel 1 (
    echo  [error] PyQt6 not found.  pip install PyQt6
    exit /b 1
)

REM Clean previous output
if exist "%DIST_DIR%\LoC-LIC Compressor" (
    echo  Removing previous dist ...
    rmdir /s /q "%DIST_DIR%\LoC-LIC Compressor"
)
if exist "%BUILD_DIR%" rmdir /s /q "%BUILD_DIR%"

echo  Building ...
%PYTHON% -m PyInstaller "%APP_DIR%\build\loclic.spec" ^
    --distpath "%DIST_DIR%" ^
    --workpath "%BUILD_DIR%" ^
    --noconfirm

if errorlevel 1 (
    echo.
    echo  [FAILED] Build exited with an error. See output above.
    exit /b 1
)

echo.
echo  Build complete!
echo  Output: %DIST_DIR%\LoC-LIC Compressor\
echo.
echo  ─────────────────────────────────────────────
echo  NEXT STEPS — copy model checkpoints into dist:
echo.
echo    mkdir "%DIST_DIR%\LoC-LIC Compressor\checkpoints\loc-lic"
echo    copy  ^<repo^>\checkpoints\loc-lic\*.pth.tar  ^<above folder^>
echo.
echo    mkdir "%DIST_DIR%\LoC-LIC Compressor\results\gptq_0035"
echo    (repeat for gptq_0067, gptq_025, qat_mixed_phase3, ...)
echo.
echo  Then run: "%DIST_DIR%\LoC-LIC Compressor\LoC-LIC Compressor.exe"
echo  ─────────────────────────────────────────────

endlocal
