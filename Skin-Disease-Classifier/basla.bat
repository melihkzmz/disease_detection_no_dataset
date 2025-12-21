@echo off
chcp 65001 >nul
echo ======================================================================
echo HASTALIK ANALIZ SISTEMI - HIZLI BASLANGIC
echo ======================================================================
echo.
echo Bu script projeyi baslatmak icin gerekli adimlari kontrol eder.
echo.
echo ======================================================================
echo.

REM Python kontrolü
echo [1/4] Python kontrol ediliyor...
python --version >nul 2>&1
if errorlevel 1 (
    echo [HATA] Python bulunamadi! Lutfen Python 3.8+ yukleyin.
    echo Indirme: https://www.python.org/downloads/
    pause
    exit /b 1
)
python --version
echo [OK] Python bulundu
echo.

REM Paket kontrolü
echo [2/4] Gerekli paketler kontrol ediliyor...
python -c "import flask" >nul 2>&1
if errorlevel 1 (
    echo [UYARI] Flask bulunamadi. Paketler yukleniyor...
    pip install -r requirements.txt
    if errorlevel 1 (
        echo [HATA] Paket yukleme basarisiz! Lutfen manuel olarak yukleyin:
        echo pip install -r requirements.txt
        pause
        exit /b 1
    )
)
echo [OK] Gerekli paketler yuklu
echo.

REM Model kontrolü
echo [3/4] Model dosyasi kontrol ediliyor...
if exist "models\bone_disease_model_4class_densenet121_macro_f1_savedmodel\saved_model.pb" (
    echo [OK] SavedModel formatinda model bulundu
) else if exist "models\bone_disease_model_4class_densenet121_macro_f1.keras" (
    echo [OK] Keras formatinda model bulundu
) else (
    echo [UYARI] Model dosyasi bulunamadi!
    echo Model dosyasi olmadan API calismayacaktir.
    echo Lutfen model dosyasinin models\ klasorunde oldugundan emin olun.
    echo.
    set /p continue="Yine de devam etmek istiyor musunuz? (E/H): "
    if /i not "%continue%"=="E" (
        exit /b 1
    )
)
echo.

REM Başlatma
echo [4/4] Backend API baslatiliyor...
echo.
echo ======================================================================
echo ONEMLI: Bu pencereyi KAPATMAYIN!
echo API calisiyor: http://localhost:5002
echo.
echo Frontend'i baslatmak icin YENI bir terminal penceresi acin ve:
echo   cd Skin-Disease-Classifier
echo   start_server.bat
echo.
echo Tarayicida acin: http://localhost:8000/analyze.html
echo ======================================================================
echo.
echo API baslatiliyor...
echo.

cd /d "%~dp0"
python bone_disease_api.py

pause

