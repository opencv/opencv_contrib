Object Detection using CNNs {#tutorial_dnn_objdetect}
===========================

# Building

Build samples of "dnn_objectect" module. Refer to OpenCV build tutorials for details.
Enable `BUILD_EXAMPLES=ON` CMake option and build these targets (Linux):
- example_dnn_objdetect_image_classification
- example_dnn_objdetect_obj_detect

Download the weights file and model definition file from `opencv_extra/dnn_objdetect`


# Object Detection

```bash
example_dnn_objdetect_obj_detect  <model-definition-file>  <model-weights-file>  <test-image>
```

All the following examples were run on a laptop with `Intel(R) Core(TM)2 i3-4005U CPU @ 1.70GHz` (without GPU).

The model is incredibly fast taking just `0.172091` seconds on an average to predict multiple bounding boxes.

```bash
<bin_path>/example_dnn_objdetect_obj_detect  SqueezeDet_deploy.prototxt  SqueezeDet.caffemodel  tutorials/images/aeroplane.jpg

Total objects detected: 1 in 0.168792 seconds
------
Class: aeroplane
Probability: 0.845181
Co-ordinates: 41 116 415 254
------
```

![Train_Dets](images/aero_det.jpg)


```bash
<bin_path>/example_dnn_objdetect_obj_detect  SqueezeDet_deploy.prototxt  SqueezeDet.caffemodel  tutorials/images/bus.jpg

Total objects detected: 1 in 0.201276 seconds
------
Class: bus
Probability: 0.701829
Co-ordinates: 0 32 415 244
------
```

![Train_Dets](images/bus_det.jpg)

```bash
<bin_path>/example_dnn_objdetect_obj_detect  SqueezeDet_deploy.prototxt  SqueezeDet.caffemodel  tutorials/images/cat.jpg

Total objects detected: 1 in 0.190335 seconds
------
Class: cat
Probability: 0.703465
Co-ordinates: 34 0 381 282
------
```

![Train_Dets](images/cat_det.jpg)

```bash
<bin_path>/example_dnn_objdetect_obj_detect  SqueezeDet_deploy.prototxt  SqueezeDet.caffemodel  tutorials/images/persons_mutli.jpg

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

![Train_Dets](images/person_multi_det.jpg)

Go ahead and run the model with other images !


## Changing threshold

By default this model thresholds the detections at confidence of `0.53`. While filtering there are number of bounding boxes which are predicted, you can manually control what gets thresholded by passing the value of optional arguement `threshold` like:

```bash
<bin_path>/example_dnn_objdetect_obj_detect  <model-definition-file>  <model-weights-file>  <test-image> <threshold>
```

Changing the threshold to say `0.0`, produces the following:

![Train_Dets](images/aero_thresh_det.jpg)

That doesn't seem to be that helpful !

# Image Classification

```bash
example_dnn_objdetect_image_classification  <model-definition-file>  <model-weights-file>  <test-image>
```

The size of the model being **4.9MB**, just takes a time of **0.136401** seconds to classify the image.

Running the model on examples produces the following results:

```bash
<bin_path>/example_dnn_objdetect_image_classification  SqueezeNet_deploy.prototxt  SqueezeNet.caffemodel  tutorials/images/aeroplane.jpg
Best class Index: 404
Time taken: 0.137722
Probability: 77.1757
```

Looking at [synset_words.txt](https://raw.githubusercontent.com/opencv/opencv/3.4.0/samples/data/dnn/synset_words.txt), the predicted class belongs to `airliner`


```bash
<bin_path>/example_dnn_objdetect_image_classification  SqueezeNet_deploy.prototxt  SqueezeNet.caffemodel  tutorials/images/cat.jpg
Best class Index: 285
Time taken: 0.136401
Probability: 40.7111
```

This belongs to the class: `Egyptian cat`

```bash
<bin_path>/example_dnn_objdetect_image_classification  SqueezeNet_deploy.prototxt  SqueezeNet.caffemodel  tutorials/images/space_shuttle.jpg
Best class Index: 812
Time taken: 0.137792
Probability: 15.8467
```

This belongs to the class: `space shuttle`
