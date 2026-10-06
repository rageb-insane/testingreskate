@echo off
rem Black crop top with your logo on the chest, for skate. (ReSkate).
rem  - Drag a logo image (a PNG with a transparent background works best) onto this file.
setlocal
cd /d "%~dp0"
if "%~1"=="" (
  echo Drag a logo image onto this file to put it on a black crop top.
  echo A PNG with a transparent background works best.
  goto done
)
set /p NAME=Top name as it should appear in game:
if "%NAME%"=="" set NAME=%~n1 Crop Top
python "tool\make_top_mod.py" "%~1" --name "%NAME%" --install

:done
echo.
echo Restart skate. to see the top (if it was running, close it and run this again to install).
pause
