echo %INCLUDE%

REM Make sure to load MSVC compiler env
call "C:\Program Files (x86)\Microsoft Visual Studio 14.0\VC\vcvarsall.bat" amd64

echo %INCLUDE%

REM
set "INCLUDE=%INCLUDE%;C:\Program Files (x86)\Microsoft Visual Studio\2017\BuildTools\VC\Tools\MSVC\14.16.27023\include"

REM
REM set "PATH=C:\Python37\python;%PATH%"

REM Now we can call cmake now that cl.exe can be found
cmake -DPYTHON_EXECUTABLE=C:\\Python37\\python %*
cmake --build .
