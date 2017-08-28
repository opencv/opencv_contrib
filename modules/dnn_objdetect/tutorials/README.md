# Building

Download the weights file and model definition file from `opencv_extra/dnn_objdetect`

```bash
cd opencv_contrib/modules/dnn_objdetect/samples
```

- Image Classification
```bash
g++ image_classification.cpp -o classifier -lopencv_core -lopencv_imgcodecs -lopencv_dnn

./classifier <model-definition-file>  <model-weights-file>  <test-image>
```

- Object Detection

```bash
g++ obj_detect.cpp -o detect -lopencv_imgcodecs -lopencv_imgproc -lopencv_dnn -lopencv_dnn_objdetect -lopencv_core -lopencv_highgui

./detect <model-definition-file>  <model-weights-file>  <test-image>
```
# Examples - Object Detection

All the following examples were run on a laptop with `Intel(R) Core(TM)2 i3-4005U CPU @ 1.70GHz` (without GPU).

The model is incredibly fast taking just `0.172091` seconds on an average to predict multiple bounding boxes.

```bash
./detect SqueezeDet_deploy.prototxt  SqueezeDet.caffemodel  ../tutorials/images/aeroplane.jpg

Total objects detected: 1 in 0.168792 seconds
------
Class: aeroplane
Probability: 0.845181
Co-ordinates: 41 116 415 254
------
```

![Train_Dets](https://github.com/kvmanohar22/opencv_contrib/blob/GSoC17_dnn_objdetect/modules/dnn_objdetect/tutorials/images/aero_det.png?raw=true)


```bash
./detect SqueezeDet_deploy.prototxt  SqueezeDet.caffemodel  ../tutorials/images/bus.jpg

Total objects detected: 1 in 0.201276 seconds
------
Class: bus
Probability: 0.701829
Co-ordinates: 0 32 415 244
------
```

![Train_Dets](https://github.com/kvmanohar22/opencv_contrib/blob/GSoC17_dnn_objdetect/modules/dnn_objdetect/tutorials/images/bus_det.png?raw=true)

```bash
./detect SqueezeDet_deploy.prototxt  SqueezeDet.caffemodel  ../tutorials/images/cat.jpg

Total objects detected: 1 in 0.190335 seconds
------
Class: cat
Probability: 0.703465
Co-ordinates: 34 0 381 282
------
```

![Train_Dets](https://github.com/kvmanohar22/opencv_contrib/blob/GSoC17_dnn_objdetect/modules/dnn_objdetect/tutorials/images/cat_det.png?raw=true)

```bash
./detect SqueezeDet_deploy.prototxt  SqueezeDet.caffemodel  ../tutorials/images/persons_mutli.png

Total objects detected: 2 in 0.169152 seconds
------
Class: person
Probability: 0.737349
Co-ordinates: 160 67 313 363
------
Class: person
Probability: 0.720328
Co-ordinates: 187 198 222 323
------
```

![Train_Dets](https://github.com/kvmanohar22/opencv_contrib/blob/GSoC17_dnn_objdetect/modules/dnn_objdetect/tutorials/images/person_multi_det.png?raw=true)

Go ahead and run the model with other images !


## Changing threshold

By default this model thresholds the detections at confidence of `0.65`. While filtering there are number of bounding boxes which are predicted, you can manually control what gets thresholded by setting the value of `thresh` in `samples/obj_detect.cpp` while calling the function `filter`

```cpp
double threshold = 0.5;
inf.filter(thresh = threshold);
```

Changing the threshold to say `0.0`, produces the following:

![Train_Dets](https://github.com/kvmanohar22/opencv_contrib/blob/GSoC17_dnn_objdetect/modules/dnn_objdetect/tutorials/images/aero_thresh_det.png?raw=true)

That doesn't seem to be that helpful !

# Examples - Image Classification

The size of the model being **4.9MB**, just takes a time of **0.136401** seconds to classify the image.

Running the model on examples produces the following results:

```bash
./detect SqueezeNet_deploy.prototxt  SqueezeNet.caffemodel  ../tutorials/images/aeroplane.jpg
Best class Index: 404
Time taken: 0.137722
Probability: 77.1757
```
Looking at [synset_words.txt](https://raw.githubusercontent.com/opencv/opencv/master/samples/data/dnn/synset_words.txt), the predicted class belongs to `airliner`


```bash
./detect SqueezeNet_deploy.prototxt  SqueezeNet.caffemodel  ../tutorials/images/cat.jpg
Best class Index: 285
Time taken: 0.136401
Probability: 40.7111
```

This belongs to the class: `Egyptian cat`

```bash
./detect SqueezeNet_deploy.prototxt  SqueezeNet.caffemodel  ../tutorials/images/space_shuttle.jpg
Best class Index: 812
Time taken: 0.137792
Probability: 15.8467
```

This belongs to the class: `space shuttle`
