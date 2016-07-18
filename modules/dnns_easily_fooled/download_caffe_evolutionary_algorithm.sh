#!/bin/bash

if [ -d "./caffe" ]; then
  echo "Please remove the existing [caffe] folder and re-run this script."
  exit 1
fi

# Download the version of Caffe that can be used for generating fooling images via EAs.
echo "Downloading Caffe ..."
wget https://github.com/Evolving-AI-Lab/fooling/archive/master.zip

echo "Extracting into ./caffe"
unzip master.zip
mv ./fooling-master/caffe ./

# Clean up
rm -rf fooling-master master.zip

echo "Done."