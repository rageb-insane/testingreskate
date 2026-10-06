@echo off
rem Custom grip tape for skate. (ReSkate).
rem  - Paint your design on grip-template.png (nose at the top), save it, and drag it onto this file.
rem  - Or double-click this file and type a colour for a plain grip.
setlocal
cd /d "%~dp0"
if not exist "grip-template.png" python "tool\make_grip_mod.py" --template "grip-template.png"
if "%~1"=="" goto colour

set /p NAME=Grip name as it should appear in game:
if "%NAME%"=="" set NAME=%~n1
python "tool\make_grip_mod.py" "%~1" --name "%NAME%" --install
goto done

:colour
echo No image dropped on this file, so this makes a plain colour grip.
echo (To use your own design, paint it on grip-template.png and drag the image onto this file.)
echo.
set /p COLOUR=Grip colour (a name like red, navy, hotpink, or a hex code like #ff8800):
if "%COLOUR%"=="" (
  echo No colour given.
  goto done
)
set /p NAME=Grip name as it should appear in game:
if "%NAME%"=="" set NAME=%COLOUR% Grip
python "tool\make_grip_mod.py" --color "%COLOUR%" --name "%NAME%" --install

:done
echo.
echo Restart skate. to see the grip (if it was running, close it and run this again to install).
pause
