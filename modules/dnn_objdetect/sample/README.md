## Building

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
