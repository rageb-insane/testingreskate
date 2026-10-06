@echo off
rem Skinny jeans in any colour, for skate. (ReSkate). Double-click and type a colour.
setlocal
cd /d "%~dp0"
set /p COLOUR=Jeans colour (black, or a name like navy, white, olive, or a hex code like #3a4a6b) [black]:
if "%COLOUR%"=="" set COLOUR=black
set /p NAME=Name as it should appear in game [%COLOUR% Skinny Jeans]:
if "%NAME%"=="" set NAME=%COLOUR% Skinny Jeans
python "tool\make_jeans_mod.py" --color "%COLOUR%" --name "%NAME%" --install
echo.
echo Restart skate. to see the jeans (if it was running, close it and run this again to install).
pause
