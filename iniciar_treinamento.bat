@echo off
chcp 65001 > nul
cd /d "%~dp0"

echo Iniciando o treinamento de deteccao de EPIs...
echo.

"%~dp0ultralytics\Scripts\python.exe" "%~dp0treinar_modelo.py"

echo.
if errorlevel 1 (
    echo O treinamento nao foi iniciado ou terminou com erro.
) else (
    echo Processo finalizado com sucesso.
)

pause
