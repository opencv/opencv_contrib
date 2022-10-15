# GCV
OpenGL/OpenCL/VAAPI interop demo using my OpenCV-4.x fork (https://github.com/kallaballa/opencv/tree/GCV)

# Instructions
While I try to get the necessary changes into the official OpenCV repo (https://github.com/opencv/opencv/issues/22607 and https://github.com/opencv/opencv/issues/22622) you need to build the OpenCV fork yourself.

## Build FFMPEG intel-cartwheel (only required for Intel Tiger Lake)
* First follow the instructions in README.cartwheel
* then configure with the following options:

```bash
./configure --disable-static --enable-shared --enable-vaapi --enable-libvorbis -enable-libvpx --enable-gpl --cpu=native --enable-opengl --enable-libfdk-aac --enable-libx264 --enable-libx265 --enable-nonfree
make -j8
sudo make install
```

## Build OpenCV

```bash
git clone --branch GCV https://github.com/kallaballa/opencv.git
cd opencv
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release -DWITH_OPENGL=ON -DWITH_VA=ON -DWITH_VA_INTEL=ON -DWITH_QT=ON -DBUILD_PERF_TESTS=OFF -DBUILD_TESTS=OFF ..
make -j8
sudo make install
```

## Build demo code

```bash
git clone https://github.com/kallaballa/GCV.git
cd GCV
make -j2
```

## Run the demo tetra-demo:

```bash
src/tetra/tetra-demo
```

## Run the demo camera-demo:

```bash
src/tetra/camera-demo
```
