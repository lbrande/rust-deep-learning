#!/bin/bash -x
for file in /opt/intel/compilers_and_libraries_2018.2.199/linux/mkl/include/*.h; do 
	~/.cargo/bin/bindgen $file -o `basename ${file%.h}`.rs
done
