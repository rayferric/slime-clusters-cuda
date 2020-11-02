VCVARS="C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Auxiliary\Build\vcvars64.bat"

default: build

build:
	IF NOT EXIST "bin" MD "bin"
	$(VCVARS) && nvcc "src\main.cu" -o="bin\slime-clusters" -Wno-deprecated-gpu-targets --disable-warnings

test: build
	bin/slime-clusters

clean:
	IF EXIST "bin" RD /S /Q "bin"