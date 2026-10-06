@echo off
rem Merges installed cosmetic mods (shoes, clothing, decks...) into one mod you can install or share.
rem Close skate. and the ReSkate launcher first if you want it installed right away.
setlocal
python "%~dp0tool\make_modpack.py" --interactive
echo.
pause
