#!/bin/bash

# Uninstall caffe from Sferes
rm -rf ~/src/sferes/include/caffe/
rm ~/src/sferes/lib/libcaffe.*
echo "Removed old installation in ~/src/sferes/"

# Reinstall caffe to Sferes

# Include files
cp -R ~/src/caffe/include/caffe/ ~/src/sferes/include/
echo "Installed header files from ~/src/caffe/include/caffe/"

cp -R ~/src/caffe/build/src/caffe/ ~/src/sferes/include/
echo "Installed header files from ~/src/caffe/build/src/caffe/"

# Library files
cp ~/src/caffe/build/lib/libcaffe.* ~/src/sferes/lib/
echo "Installed library files from ~/src/caffe/build/lib/"

echo "Done."
