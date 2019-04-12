call "C:\Program Files (x86)\Microsoft Visual Studio 14.0\VC\vcvarsall.bat" x64
dir /S "C:\Program Files (x86)\Microsoft Visual Studio 14.0\Common7\IDE\Extensions\CppSDK/SDK/include/"
cmake .. -DPYTHON_EXECUTABLE=C:\\Python37\\python -GNinja -DCMAKE_C_COMPILER="cl.exe" -DCMAKE_CXX_COMPILER="cl.exe" %*
