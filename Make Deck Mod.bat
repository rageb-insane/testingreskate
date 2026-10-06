@echo off
rem Drag a finished deck image onto this file to turn it into a ReSkate deck mod.
setlocal
if "%~1"=="" (
  echo Drag your finished deck image ^(PNG/JPG^) onto this file.
  pause
  exit /b 1
)
set /p NAME=Deck name as it should appear in game:
if "%NAME%"=="" set NAME=%~n1
python "%~dp0tool\make_deck_mod.py" "%~1" --name "%NAME%" --install
echo.
echo If skate. is running, open the ReSkate menu (Insert) ^> MODS and apply, or just restart the game.
pause
