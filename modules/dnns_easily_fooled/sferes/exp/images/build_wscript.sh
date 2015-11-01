#!/bin/bash
home=$(echo ~)
echo "You are building from: $home"

# Check the building folder, either on local or Moran
if [ "$home" == "/home/anh" ]
then
    echo "Enabled local settings.."
    cp ./wscript.local ./wscript
  
  
else
  if [ "$home" == "/home/anguyen8" ]
  then 
      echo "Enabled Moran settings.."
      cp ./wscript.moran ./wscript
  else
    echo "Unknown environment. Building stopped."    
  fi
fi


