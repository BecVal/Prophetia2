@echo off
echo ======================================
echo      INSTALADOR DE PROPHETIA
echo ======================================

echo.
echo [1/3] Instalando dependencias Python...
pip install -r requirements.txt

echo.
echo [2/3] Instalando dependencias Frontend...
cd frontend
npm install

echo.
echo [3/3] Instalacion completada.
echo.

pause