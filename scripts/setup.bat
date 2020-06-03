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

pushd "%~dp0"

REM ###########################################################################
REM Init

set NSTOOLS_DIR="%CD%\..\nstools"
set NSTOOLS_URL="git@github.com:agenium-scale/nstools.git"
set NSTOOLS_URL2="https://github.com/agenium-scale/nsimd.git"

REM ###########################################################################
REM Build nsconfig (if not already built)

if exist "%NSTOOLS_DIR%\README.md" (
  pushd %NSTOOLS_DIR%
  git pull
  popd
) else (
  pushd ".."
  git clone %NSTOOLS_URL% || git clone %NSTOOLS_URL2%
  popd
)

if not exist %NSTOOLS_DIR%\bin (
  md %NSTOOLS_DIR%\bin
)

if not exist %NSTOOLS_DIR%\bin\nsconfig.exe (
  pushd %NSTOOLS_DIR%\nsconfig
  nmake /F Makefile.win all
  copy "nsconfig.exe" %NSTOOLS_DIR%\bin
  copy "nstest.exe" %NSTOOLS_DIR%\bin
  popd
)

popd
