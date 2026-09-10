@echo off
chcp 65001 > nul
cd /d "%~dp0"

echo Iniciando a avaliacao final no conjunto test...
echo.

"%~dp0ultralytics\Scripts\python.exe" "%~dp0avaliar_modelo.py"

echo.
if errorlevel 1 (
    echo A avaliacao nao foi iniciada ou terminou com erro.
) else (
    echo Avaliacao finalizada com sucesso.
)

pause
