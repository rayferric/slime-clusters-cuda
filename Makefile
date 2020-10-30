VCVARS="C:\Program Files (x86)\Microsoft Visual Studio 14.0\VC\bin\vcvars32.bat"

default: build

build:
    IF NOT EXIST "bin" MD "bin"
    $(VCVARS) && nvcc "src\main.cu" -o="bin\slime-clusters" -Wno-deprecated-gpu-targets --disable-warnings

run: build
    bin/slime-clusters

clean:
    IF EXIST "bin" RD /S /Q "bin"