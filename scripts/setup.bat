@echo off

REM Copyright (c) 2020 Agenium Scale
REM
REM Permission is hereby granted, free of charge, to any person obtaining a copy
REM of this software and associated documentation files (the "Software"), to deal
REM in the Software without restriction, including without limitation the rights
REM to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
REM copies of the Software, and to permit persons to whom the Software is
REM furnished to do so, subject to the following conditions:
REM
REM The above copyright notice and this permission notice shall be included in all
REM copies or substantial portions of the Software.
REM
REM THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
REM IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
REM FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
REM AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
REM LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
REM OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
REM SOFTWARE.

REM ###########################################################################

setlocal EnableDelayedExpansion
pushd "%~dp0"

REM ###########################################################################
REM Init

set NSTOOLS_DIR="%CD%\..\nstools"

REM ###########################################################################
REM Pull nstools

if exist "%NSTOOLS_DIR%\README.md" (
  pushd %NSTOOLS_DIR%
  git pull
  popd
) else (
  if exist "..\.git" (
    git remote get-url origin >_tmp-nsimd-url.txt
    set /P NSIMD_URL=<_tmp-nsimd-url.txt
    set NSTOOLS_URL=!NSIMD_URL:nsimd=nstools!
    del /F /Q _tmp-nsimd-url.txt
    pushd ".."
    git clone !NSTOOLS_URL! nstools
    popd
  ) else (
    pushd ".."
    git clone "https://github.com/agenium-scale/nstools.git" nstools
    popd
  )
)

if "%NSIMD_NSTOOLS_CHECKOUT_LATER%" == "" (
  git -C %NSTOOLS_DIR% checkout v1.0
) else (
  git -C %NSTOOLS_DIR% checkout master
)

REM ###########################################################################
REM Create bin directory

if not exist %NSTOOLS_DIR%\bin (
  md %NSTOOLS_DIR%\bin
)

REM ###########################################################################
REM Build nsconfig (if not already built)

pushd %NSTOOLS_DIR%\nsconfig
nmake /F Makefile.win nsconfig.exe
nmake /F Makefile.win nstest.exe
copy /Y "nsconfig.exe" %NSTOOLS_DIR%\bin
copy /Y "nstest.exe" %NSTOOLS_DIR%\bin
popd

popd
endlocal
exit /B 0
