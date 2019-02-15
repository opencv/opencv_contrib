Tesseract (master) installation by using git-bash (version>=2.14.1) and cmake (version >=3.9.1){#tutorial_install_tesseract}
===============================================================

-#  We assume you installed opencv and opencv_contrib in c:/lib using [this tutorials](http://docs.opencv.org/master/d3/d52/tutorial_windows_install.html#tutorial_windows_gitbash_build]

-# You must download [png lib](https://sourceforge.net/projects/libpng/files/libpng16/1.6.32/lpng1632.zip/download) and [zlib](https://sourceforge.net/projects/libpng/files/zlib/1.2.11/zlib1211.zip/download).
Uncompress lpngx.y.zz in folder lpng and zlib in folder zlib. lpng and zlib must be in same folder as opencv and opencv_contrib.
save this script with name installpngzlib.sh in c:/lib
@code{.bash}
#!/bin/bash
myRepo=$(pwd)
CMAKE_CONFIG_GENERATOR="Visual Studio 14 2015 Win64"
RepoSource=zlib
mkdir Build/$RepoSource
pushd Build/$RepoSource
cmake . -G"Visual Studio 14 2015 Win64" \
-DCMAKE_INSTALL_PREFIX:PATH="$myRepo"/install/zlib -DINSTALL_BIN_DIR:PATH="$myRepo"/install/zlib/bin \
-DINSTALL_INC_DIR:PATH="$myRepo"/install/zlib/include -DINSTALL_LIB_DIR:PATH="$myRepo"/install/zlib/lib "$myRepo"/"$RepoSource"
cmake --build . --config release
cmake --build . --target install --config release
cmake --build . --config debug
cmake --build . --target install --config debug
popd
RepoSource=lpng
mkdir Build/$RepoSource
pushd Build/$RepoSource
cp "$myRepo"/"$RepoSource"/scripts/pnglibconf.h.prebuilt "$myRepo"/"$RepoSource"/pnglibconf.h
cmake . -G"Visual Studio 14 2015 Win64" \
-DZLIB_INCLUDE_DIR:PATH="$myRepo"/install/zlib/include -DZLIB_LIBRARY_DEBUG:FILE="$myRepo"/install/zlib/lib/zlibstaticd.lib \
-Dld-version-script:BOOL=OFF -DPNG_TESTS:BOOL=OFF -DAWK:STRING= \
-DZLIB_LIBRARY_RELEASE:FILE="$myRepo"/install/zlib/lib/zlibstatic.lib -DCMAKE_INSTALL_PREFIX="$myRepo"/Install/"$RepoSource" \
"$myRepo"/"$RepoSource"
cmake --build . --config release
cmake --build . --target install --config release
cmake --build . --config debug
cmake --build . --target install --config debug
popd
@endcode
-#  In git command line enter the following command :
@code{.bash}
./installpngzlib.sh
@endcode

-#  save this script with name installTesseract.sh in c:/lib
@code{.bash}
#!/bin/bash
function MAJGitRepo
{
if [  ! -d "$myRepo/$1"  ]; then
  echo "clonning ${1}"
  git clone $2
  mkdir Build/$1
else
  echo "update $1"
  cd $1
  git pull --rebase
  cd ..
fi
}
echo "Installing leptonica and tesseract"
myRepo=$(pwd)
CMAKE_CONFIG_GENERATOR="Visual Studio 14 2015 Win64"

MAJGitRepo leptonica https://github.com/DanBloomberg/leptonica.git
RepoSource=leptonica
pushd Build/$RepoSource
cmake -G"$CMAKE_CONFIG_GENERATOR" -DCMAKE_INSTALL_PREFIX="$myRepo"/Install/leptonica "$myRepo/$RepoSource"
echo "************************* $Source_DIR -->debug"
cmake --build .  --config release
cmake --build $RepoSource  --target install --config release
popd

RepoSource=tesseract
MAJGitRepo $RepoSource https://github.com/tesseract-ocr/tesseract.git
pushd Build/$RepoSource
cmake -G"$CMAKE_CONFIG_GENERATOR"  -DBUILD_TRAINING_TOOLS:BOOL=OFF -DCMAKE_INSTALL_PREFIX="$myRepo"/Install/tesseract -DLeptonica_DIR:PATH="$myRepo"/Install/leptonica/cmake -DPKG_CONFIG_EXECUTABLE:BOOL=OFF "$myRepo"/"$RepoSource"
echo "************************* $Source_DIR -->release"
cmake --build . --config release
cmake --build .  --target install --config release

popd
RepoSource=opencv
pushd Build/$RepoSource
CMAKE_OPTIONS='-DBUILD_PERF_TESTS:BOOL=OFF -DBUILD_TESTS:BOOL=OFF -DBUILD_DOCS:BOOL=OFF -DWITH_CUDA:BOOL=OFF'
cmake -G"$CMAKE_CONFIG_GENERATOR"  \
-DTesseract_INCLUDE_DIR:PATH="${myRepo}"/Install/tesseract/include -DTesseract_LIBRARY="${myRepo}"/Install/tesseract/lib/tesseract400.lib -DLept_LIBRARY="${myRepo}"/Install/leptonica/lib/leptonica-1.74.4.lib \
$CMAKE_OPTIONS -DOPENCV_EXTRA_MODULES_PATH=../../opencv_contrib/modules \
-DINSTALL_CREATE_DISTRIB=ON -DCMAKE_INSTALL_PREFIX="$myRepo"/install/"$RepoSource"  "$myRepo/$RepoSource"
echo "************************* $Source_DIR -->devenv debug"
cmake --build .  --config debug
echo "************************* $Source_DIR -->devenv release"
cmake --build .  --config release
cmake --build .  --target install --config release
cmake --build .  --target install --config debug
popd

@endcode
    In this script I suppose you use VS 2015 in 64 bits
@code{.bash}
CMAKE_CONFIG_GENERATOR="Visual Studio 14 2015 Win64"
@endcode
    and leptonica, tesseract will be installed in c:/lib/install
@code{.bash}
-DCMAKE_INSTALL_PREFIX="$myRepo"/install/"$RepoSource" "$myRepo/$RepoSource"
@endcode
    with no Perf tests, no tests, no doc, no CUDA and no example
@code{.bash}
CMAKE_OPTIONS='-DBUILD_PERF_TESTS:BOOL=OFF -DBUILD_TESTS:BOOL=OFF -DBUILD_DOCS:BOOL=OFF  -DWITH_CUDA:BOOL=OFF -DBUILD_EXAMPLES:BOOL=OFF'
@endcode
-#  In git command line enter the following command :
@code{.bash}
./installTesseract.sh
@endcode
-# now we need the language files from tesseract. either clone https://github.com/tesseract-ocr/tessdata, or copy only those language files you need to a folder (example c:\\lib\\install\\tesseract\\tessdata). If you don't want to add a new folder you must copy language file in same folder than your executable
-# if you created a new folder, then you must add a new variable, TESSDATA_PREFIX with the value c:\\lib\\install\\tessdata to your system's environment
-# add c:\\Lib\\install\\leptonica\\bin and c:\\Lib\\install\\tesseract\\bin to your PATH environment. If you don't want to modify the PATH then copy tesseract400.dll and leptonica-1.74.4.dll to the same folder than your exe file.
