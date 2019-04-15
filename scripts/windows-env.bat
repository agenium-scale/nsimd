REM Load MSVC compiler env
call "C:\Program Files (x86)\Microsoft Visual Studio 14.0\VC\vcvarsall.bat" amd64

REM Convert fresh env to bash env
bash -c "env | sed 's/^/export /' | sed ""s/=/='/"" | sed ""s/$/'/""" > _env.sh
