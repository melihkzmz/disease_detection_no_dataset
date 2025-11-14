@echo off
REM ============================================================
REM TensorFlow GPU Environment Setup Script
REM Requirements: Miniconda installed
REM ============================================================

echo.
echo ============================================================
echo TENSORFLOW GPU ENVIRONMENT SETUP
echo ============================================================
echo.

echo [STEP 1/6] Creating Python 3.11 environment...
call conda create -n tf_gpu python=3.11 -y
if errorlevel 1 (
    echo [ERROR] Failed to create conda environment
    pause
    exit /b 1
)

echo.
echo [STEP 2/6] Activating environment...
call conda activate tf_gpu
if errorlevel 1 (
    echo [ERROR] Failed to activate environment
    pause
    exit /b 1
)

echo.
echo [STEP 3/6] Installing CUDA Toolkit and cuDNN...
call conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0 -y
if errorlevel 1 (
    echo [ERROR] Failed to install CUDA
    pause
    exit /b 1
)

echo.
echo [STEP 4/6] Installing TensorFlow GPU...
call pip install tensorflow==2.10.0
if errorlevel 1 (
    echo [ERROR] Failed to install TensorFlow
    pause
    exit /b 1
)

echo.
echo [STEP 5/6] Installing other requirements...
call pip install pillow matplotlib seaborn scikit-learn pandas numpy
if errorlevel 1 (
    echo [ERROR] Failed to install requirements
    pause
    exit /b 1
)

echo.
echo [STEP 6/6] Testing GPU detection...
python test_gpu.py

echo.
echo ============================================================
echo SETUP COMPLETE!
echo ============================================================
echo.
echo [NEXT STEPS]
echo   1. Close this window
echo   2. Open NEW PowerShell/CMD
echo   3. Run: conda activate tf_gpu
echo   4. Run: cd Skin-Disease-Classifier
echo   5. Run: python train_mendeley_eye.py
echo.
echo ============================================================
pause

