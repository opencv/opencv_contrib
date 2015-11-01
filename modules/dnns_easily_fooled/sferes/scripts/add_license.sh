#!/bin/sh
set -x
PATHS='sferes scripts tests modules/nn modules/cartpole'
FILES=`find $PATHS |egrep "(.hpp|.cpp)$"`
for i in $FILES; do
    grep -v "//|" $i >/tmp/file
    cp scripts/license_cpp.txt $i
    cat /tmp/file >> $i
done

FILES=`find $PATHS |egrep "(wscript|.py)$"`
FILES="$FILES wscript"
for i in $FILES; do
    grep -v "#|" $i|grep -v "#!" >/tmp/file
    cat scripts/license_py.txt >> $i
    cat /tmp/file >> $i
done
