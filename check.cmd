:: Run all static analysis checks.
@echo off
cls
call .venv\Scripts\activate.bat
ruff format src test
ruff check --fix src test
where ty >nul 2>nul
if %ERRORLEVEL%==0 (
	ty check
) else (
	echo Skipping ty: command not found
)

where pyright >nul 2>nul
if %ERRORLEVEL%==0 (
	pyright
) else (
	echo Skipping pyright: command not found
)

python -m pytest test/ -v
echo.
echo All checks passed!