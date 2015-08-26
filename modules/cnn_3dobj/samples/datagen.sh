rm -rf ../data/binary_image
rm -rf ../data/binary_label
rm -rf ../data/header_for_image
rm -rf ../data/header_for_label

./sphereview_test -plymodel=../data/3Dmodel/ape.ply -label_class=0 -cam_head_x=0 -cam_head_y=0 -cam_head_z=1
./sphereview_test -plymodel=../data/3Dmodel/ant.ply -label_class=1 -cam_head_x=0 -cam_head_y=-1 -cam_head_z=0
./sphereview_test -plymodel=../data/3Dmodel/cow.ply -label_class=2 -cam_head_x=0 -cam_head_y=-1 -cam_head_z=0
./sphereview_test -plymodel=../data/3Dmodel/plane.ply -label_class=3 -cam_head_x=0 -cam_head_y=-1 -cam_head_z=0
./sphereview_test -plymodel=../data/3Dmodel/bunny.ply -label_class=4 -cam_head_x=0 -cam_head_y=-1 -cam_head_z=0
./sphereview_test -plymodel=../data/3Dmodel/horse.ply -label_class=5 -cam_head_x=0 -cam_head_y=0 -cam_head_z=-1