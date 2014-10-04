export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:../../../../../opencv/build/lib

../../../../../opencv/build/bin/example_graphmodels_train_convnet --board=0 --model=net.pbtxt --train=train_plus_val_data.pbtxt --val=test_data.pbtxt

