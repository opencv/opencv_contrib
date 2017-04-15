#!/bin/bash
path="./sferes"

if [ -d "${path}" ]; then
  echo "Please remove the existing [${path}] folder and re-run this script."
  exit 1
fi

# Download the version of Sferes that can be used for generating fooling images via EAs.
echo "Downloading Sferes ..."
wget https://github.com/Evolving-AI-Lab/fooling/archive/master.zip

echo "Extracting into ${path}"
unzip master.zip
mv ./fooling-master/sferes ./

# Clean up
rm -rf fooling-master master.zip

echo "Done."
