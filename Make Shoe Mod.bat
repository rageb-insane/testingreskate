@echo off
rem Drag a shoe model (scene.gltf from a Sketchfab glTF download, a .glb, or an .obj with its .mtl) onto this file.
setlocal
if "%~1"=="" (
  echo Drag a shoe model ^(.gltf, .glb or .obj^) onto this file.
  pause
  exit /b 1
)
set /p NAME=Shoe name as it should appear in game:
if "%NAME%"=="" set NAME=%~n1
set /p CREDIT=Credit line for the model's author (paste from its license.txt, or leave empty):
python "%~dp0tool\make_shoe_mod.py" "%~1" --name "%NAME%" --credit "%CREDIT%" --install
echo.
echo Restart skate. to see the shoes (if it was running, close it and run this again to install).
pause
