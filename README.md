# GCV
OpenGL/OpenCL/VAAPI interop demo using the 4.x branch of OpenCV (https://github.com/opencv/opencv/tree/4.x)

# Instructions
You need to build the most recent 4.x branch of OpenCV.

## Build OpenCV

```bash
git clone --branch 4.x https://github.com/opencv/opencv.git
cd opencv
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release -DHAVE_EGL_INTEROP=ON -DWITH_OPENGL=ON -DWITH_VA=ON -DWITH_VA_INTEL=ON -DWITH_QT=ON -DBUILD_PERF_TESTS=OFF -DBUILD_TESTS=OFF ..
make -j8
sudo make install
```

## Build demo code

```bash
git clone https://github.com/kallaballa/GCV.git
cd GCV
make -j2
```
## Download the example file
```bash
wget -O bunny.ogg https://download.blender.org/peach/bigbuckbunny_movies/big_buck_bunny_1080p_stereo.ogg
```
## Run the tetra-demo:

```bash
src/tetra/tetra-demo
```

## Run the video-demo:

```bash
src/video/video-demo ogg.webm
```

## Run the nanovg-demo:

```bash
src/nanovg/nanovg-demo ogg.webm
```
