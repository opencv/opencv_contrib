# GCV
OpenGL/OpenCL/VAAPI interop demo using my OpenCV-4.x fork (https://github.com/kallaballa/opencv/tree/GCV)

# Instructions
While I try to get the necessary changes into the official OpenCV repo (https://github.com/opencv/opencv/issues/22607) you need to build the OpenCV fork yourself. More detail instructions will follow.

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
make
```

## Run the demo

```bash
src/tetra/tetra-demo
```
