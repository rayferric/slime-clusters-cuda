# MSVC 2015 linker doesn't work with runtime toolkit 142 (version required by BOINC libraries),
# but CL allows checking for compilation errors on a limited system:

# NVCC=C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v8.0
# VCVARS=C:/Program Files (x86)/Microsoft Visual Studio 14.0/VC/bin/amd64/vcvars64.bat

NVCC=C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.0
VCVARS=C:/Program Files (x86)/Microsoft Visual Studio/2019/Community/VC/Auxiliary/Build/vcvars64.bat

PLATFORM=windows

msvc:
	IF NOT EXIST "bin" MD "bin"
	"$(VCVARS)" && "$(NVCC)/bin/nvcc" "src/main.cu" -o="bin/slime-clusters" -Wno-deprecated-gpu-targets -I"include/boinc" -I"include/boinc/$(PLATFORM)" -L"lib" -lboinc_api -lboinc -luser32 -m64 -O3

msvc-debug:
	IF NOT EXIST "bin" MD "bin"
	"$(VCVARS)" && "$(NVCC)/bin/nvcc" "src/main.cu" -o="bin/slime-clusters" -Wno-deprecated-gpu-targets -I"include/boinc" -I"include/boinc/$(PLATFORM)" -L"lib" -lboinc_api -lboinc -luser32 -m64 -O0 -Xcompiler /Zi -Xcompiler /Fdbin// -Xlinker /DEBUG:FULL

mingw:
	IF NOT EXIST "bin" MD "bin"
	"$(NVCC)/bin/nvcc" "src/main.cu" -o="bin/slime-clusters" -Wno-deprecated-gpu-targets -I"include/boinc" -I"include/boinc/$(PLATFORM)" -L"lib" -lboinc_api -lboinc -luser32 -m64 -O3 -Xcompiler -static-libgcc -Xcompiler -static-libstdc+

mingw-debug:
	IF NOT EXIST "bin" MD "bin"
	"$(NVCC)/bin/nvcc" "src/main.cu" -o="bin/slime-clusters" -Wno-deprecated-gpu-targets -I"include/boinc" -I"include/boinc/$(PLATFORM)" -L"lib" -lboinc_api -lboinc -luser32 -m64 -O0 -Xcompiler -static-libgcc -Xcompiler -static-libstdc+ -Xcompiler -g

clean:
	IF EXIST "bin" RD /S /Q "bin"