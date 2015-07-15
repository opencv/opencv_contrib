#!/usr/bin/env sh
# This script converts the lfw data into leveldb format.

git clone https://github.com/Wangyida/caffe/tree/cnn_triplet
cd caffe
mkdir build
cd build
cmake -DCMAKE_INSTALL_PREFIX=/usr/local ..
make -j4
make test
sudo make install
cd ..
cmake ..
make -j4

./sphereview_test -ite_depth=2 -plymodel=../3Dmodel/ape.ply -imagedir=../data/images_ape/ -labeldir=../data/label_ape.txt -num_class=4 -label_class=0
./sphereview_test -ite_depth=2 -plymodel=../3Dmodel/ant.ply -imagedir=../data/images_ant/ -labeldir=../data/label_ant.txt -num_class=4 -label_class=1
./sphereview_test -ite_depth=2 -plymodel=../3Dmodel/cow.ply -imagedir=../data/images_cow/ -labeldir=../data/label_cow.txt -num_class=4 -label_class=2
./sphereview_test -ite_depth=2 -plymodel=../3Dmodel/plane.ply -imagedir=../data/images_plane/ -labeldir=../data/label_plane.txt -num_class=4 -label_class=3

echo "Creating leveldb..."

rm -rf ./linemod_triplet_train_leveldb
rm -rf ./linemod_triplet_test_leveldb

convert_lfw_triplet_data \
    ./binary_image_train \
    ./binary_label_train \
    ./linemod_triplet_train_leveldb
convert_lfw_triplet_data \
    ./binary_image_test \
    ./binary_image_test \
    ./linemod_triplet_test_leveldb

echo "Done."

caffe train --solver=examples/triplet/lfw_triplet_solver.prototxt
