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

set SETUP_BAT="%CD%\setup.bat"
set NSCONFIG="%CD%\..\nstools\bin\nsconfig.exe"
set HATCH_PY="%CD%\..\egg\hatch.py"
set BUILD_ROOT="%CD%\.."

REM ###########################################################################
REM Run setup

call %SETUP_BAT%
if errorlevel 1 goto end_nok

REM ###########################################################################
REM Generate NSIMD

python %HATCH_PY% -lf
if errorlevel 1 goto end_nok

REM ###########################################################################
REM Check/parse command line arguments

if "%1" == "" (
  echo %0: usage: %0 for simd_ext1/.../simd_ext2 [with compiler1/.../compiler2]
  goto end_ok
)

if not "%1" == "for" (
  echo ERROR: expected 'for' as first argument
  goto end_nok
)

if "%2" == "" (
  echo "ERROR: no SIMD extension given"
  goto end_nok
)

set SIMD_EXTS_ARG=%2
set SIMD_EXTS=%SIMD_EXTS_ARG:/=,%

if "%3" == "" (
  set COMPILER_ARG=cl
) else ( if "%3" == "with" (
  if "%4" == "" (
    echo "ERROR: no compiler given after with"
    goto end_nok
  )
  set COMPILER_ARG=%4
) else (
  echo ERROR: expected 'with' as fourth argument
  goto end_nok
) )

set COMPILERS=%COMPILER_ARG:/=,%

for %%g in (%COMPILERS%) do (
  %%g 1>nul 2>nul
  if errorlevel 1 (
    echo ERROR: compiler %%g not found in PATH
    goto end_nok
  )
)

REM ###########################################################################
REM Build NSIMD : one build directory per SIMD extension per compiler

for %%g in (%COMPILERS%) do (
  for %%h in (%SIMD_EXTS%) do (
    set BUILD_DIR=%BUILD_ROOT%\build-%%h-%%g
    if exist !BUILD_DIR! rd /Q /S !BUILD_DIR!
    md !BUILD_DIR!
    pushd !BUILD_DIR!
      %NSCONFIG% .. -Dsimd=%%h -comp=%%g
      ninja
    popd
  )
)

REM ###########################################################################

:end_ok
popd
endlocal
exit /B 0

:end_nok
popd
endlocal
exit /B 1
