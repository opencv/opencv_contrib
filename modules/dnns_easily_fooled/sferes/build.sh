#!/bin/bash
home=$(echo ~)

quit=0

# Remove the build folder
rm -rf ./build
echo "Build folder removed."

# Check the building folder, either on local or Moran
if [ "$home" == "/home/anh" ]
then    
    echo "Configuring sferes for local.."    
    echo "..."
    ./waf clean
    ./waf distclean
    #./waf configure --boost-include=/home/anh/src/sferes/include --boost-lib=/home/anh/src/sferes/lib --eigen3=/home/anh/src/sferes/include --mpi=/home/anh/openmpi
    ./waf configure --boost-include=/home/anh/src/sferes/include --boost-lib=/home/anh/src/sferes/lib --eigen3=/home/anh/src/sferes/include
    
    quit=1
    
else
  if [ "$home" == "/home/anguyen8" ]
  then 
      echo "Configuring sferes for Moran.."
      echo "..."
      ./waf clean
      ./waf distclean

      # TBB
      # ./waf configure --boost-include=/project/RIISVis/anguyen8/sferes/include/ --boost-libs=/project/RIISVis/anguyen8/sferes/lib/ --eigen3=/home/anguyen8/local/include --mpi=/apps/OPENMPI/gnu/4.8.2/1.6.5 --tbb=/home/anguyen8/sferes --libs=/home/anguyen8/local/lib
      
      # MPI (No TBB)
      ./waf configure --boost-include=/project/RIISVis/anguyen8/sferes/include/ --boost-libs=/project/RIISVis/anguyen8/sferes/lib/ --eigen3=/home/anguyen8/local/include --mpi=/apps/OPENMPI/gnu/4.8.2/1.6.5 --libs=/home/anguyen8/local/lib
      
      quit=1

  else
    echo "Unknown environment. Building stopped."    
  fi
fi

if [ "$quit" -eq "1" ]
then
    echo "Building sferes.."    
    echo "..."
    echo "..."
    ./waf build
    
    echo "Building exp/images.."
    echo "..."
    echo "..."
    ./waf --exp images
fi

