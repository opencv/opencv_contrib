#!/bin/bash

if [ -d "./caffe" ]; then
  echo "Please remove the existing [caffe] folder and re-run this script."
  exit 1
fi

# Download the version of Caffe that can be used for generating fooling images via EAs.
echo "Downloading Caffe ..."
wget https://github.com/Evolving-AI-Lab/fooling/archive/ascent.zip

echo "Extracting into ./caffe"
unzip ascent.zip
mv ./fooling-ascent/caffe ./

# Clean up
rm -rf fooling-ascent ascent.zip

echo "Done."