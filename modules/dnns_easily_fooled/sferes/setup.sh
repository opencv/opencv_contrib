#!/bin/bash

# Setup the exp/images experiment
cd exp/images/
./build_wscript.sh
cd ../../

dir=$(echo $(pwd))
#echo "Please set LOCAL_RUN in $dir/exp/images/settings.h"
#vim $dir/exp/images/settings.h
echo "Done setting up."
