@echo off
rem Builds ebxtool.exe from ReSkate's Engine/Resource sources (needs Visual Studio's C++ tools).
setlocal
set HERE=%~dp0
set RS=%~1
if "%RS%"=="" set RS=%HERE%..\ReSkate
call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat" >nul || exit /b 1
cd /d "%HERE%"
cl /nologo /std:c++latest /EHsc /O2 /MT /utf-8 /I"%RS%" ebxtool.cpp ^
  "%RS%\Engine\Resource\binary_io.cpp" ^
  "%RS%\Engine\Resource\ebx_document.cpp" ^
  "%RS%\Engine\Resource\ebx_writer.cpp" ^
  "%RS%\Engine\Resource\ebx_merge.cpp" ^
  "%RS%\Engine\Resource\bundle_ref_table.cpp" ^
  /Fe:ebxtool.exe /Fo:obj\ || exit /b 1
