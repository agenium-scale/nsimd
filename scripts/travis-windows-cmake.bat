REM Make sure to load MSVC compiler env
call "C:\Program Files (x86)\Microsoft Visual Studio 14.0\VC\vcvarsall.bat" x64

REM Now we can call cmake now that cl.exe can be found
cmake -DPYTHON_EXECUTABLE=C:\\Python37\\python -DCMAKE_CXX_FLAGS="/IC:\\Program Files (x86)\\Microsoft Visual Studio\\2017\\BuildTools\\VC\\Tools\\MSVC\\14.16.27023\\include" %*
