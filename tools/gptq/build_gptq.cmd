@echo off
setlocal
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1 || exit /b 1
set "CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3"
set "PATH=%CUDA_PATH%\bin;%PATH%"
cd /d D:\Projects\basalt-hy4-gptq || exit /b 1
if not exist build\build.ninja cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DBASALT_MODELS=hy4flash -DCMAKE_CUDA_ARCHITECTURES=120 || exit /b 1
cmake --build build -j 16 || exit /b 1
exit /b 0
