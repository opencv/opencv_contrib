# Building

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

Download the weights file and model definition file from `opencv_extra/dnn_objdetect` and run the detector like this:

```bash
./detect SqueezeDet_deploy.prototxt  SqueezeDet.caffemodel  data/aeroplane.jpg
```

which produces:

![Train_Dets](https://github.com/kvmanohar22/opencv_contrib/blob/GSoC17_dnn_objdetect/modules/dnn_objdetect/sample/data/aero_det.png?raw=true)

- Other detection examples:

![Train_Dets](https://github.com/kvmanohar22/opencv_contrib/blob/GSoC17_dnn_objdetect/modules/dnn_objdetect/sample/data/bus_det.png?raw=true)

![Train_Dets](https://github.com/kvmanohar22/opencv_contrib/blob/GSoC17_dnn_objdetect/modules/dnn_objdetect/sample/data/cat_det.png?raw=true)

![Train_Dets](https://github.com/kvmanohar22/opencv_contrib/blob/GSoC17_dnn_objdetect/modules/dnn_objdetect/sample/data/person_multi_det.png?raw=true)

Go ahead and run the model with other images !


##### Changing threshold

By default this model thresholds the detections at confidence of `0.65`. While filtering there are number of bounding boxes which are predicted, you can manually control what gets thresholded by setting the value of thresh while calling the function `filter`

```cpp
double threshold = 0.5;
inf.filter(thresh = threshold);
```

Changing the threshold to say `0.0`, produces the following:

![Train_Dets](https://github.com/kvmanohar22/opencv_contrib/blob/GSoC17_dnn_objdetect/modules/dnn_objdetect/sample/data/aero_thresh_det.jpg?raw=true)

That doesn't seem to be that helpful !

# Examples - Image Classification

The size of the model being **4.9MB**, just takes a time of **0.136401** seconds to classify the image.

Running the model on examples produces the following results:

```bash
./detect SqueezeNet_deploy.prototxt  SqueezeNet.caffemodel  data/aeroplane.jpg
Best class Index: 404
Time taken: 0.137722
Probability: 77.1757
```
Looking at [synset_words.txt](https://raw.githubusercontent.com/opencv/opencv/master/samples/data/dnn/synset_words.txt), the predicted class belongs to `airliner`


```bash
./detect SqueezeNet_deploy.prototxt  SqueezeNet.caffemodel  data/cat.jpg
Best class Index: 285
Time taken: 0.136401
Probability: 40.7111
```

This belongs to the class: `Egyptian cat`

```bash
./detect SqueezeNet_deploy.prototxt  SqueezeNet.caffemodel  data/space_shuttle.jpg
Best class Index: 812
Time taken: 0.137792
Probability: 15.8467
```

This belongs to the class: `space shuttle`
