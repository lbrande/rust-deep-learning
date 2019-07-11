#!/bin/bash
for file in /opt/intel/compilers_and_libraries_2018.2.199/linux/mkl/include/*.h; do 
	basename=`basename ${file%.h}`
	newname=src/${basename#mkl_}.rs
	~/.cargo/bin/bindgen $file -o $newname
done
