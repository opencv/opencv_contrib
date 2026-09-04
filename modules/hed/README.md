# hed — Holistically-nested Edge Detection

Implements HED edge detection (Xie & Tu, 2015) as an OpenCV contrib module.
Unlike Canny, HED uses a pretrained VGG network to detect semantically
meaningful edges rather than raw pixel contrast changes.

## Paper
Xie, S., & Tu, 2015. Holistically-nested edge detection.
https://arxiv.org/abs/1504.06375

## Model files
Download from: https://vcl.ucsd.edu/hed/
- hed_pretrained_bsds.caffemodel
- deploy.prototxt

## Python usage
```python
detector = cv2.hed.HEDDetector.create("model.caffemodel", "deploy.prototxt")
edges = detector.detectEdges(image)  # float32, values 0..1
```

## C++ usage
```cpp
auto det = cv::hed::HEDDetector::create("model.caffemodel", "deploy.prototxt");
cv::Mat edges = det->detectEdges(image);
```