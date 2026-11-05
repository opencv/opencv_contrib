# Dehaze Module

Single image dehazing using the Dark Channel Prior (He, Sun, Tang, CVPR 2009),
with guided filter refinement of the transmission map and basic sky-region
protection to reduce over-correction in sky areas.

## Algorithm

1. **Dark Channel** — For each pixel, the lowest value between the three
   channels (B, G, R). Haze-free outdoor image patches will always have
   a very low value in at least one of the channels, while a high value
   means haze is present.
2. **Atmospheric Light Estimation** — implemented using quad-tree search
   (descending recursively into the image region that has the largest
   mean minus standard deviation of the dark channel value). It is
   more robust to colored haze outliers compared to simple brightest
   pixel averaging.
3. **Transmission Map** — estimates how much of the original scene radiance
   reaches the camera at each pixel.
4. **Guided Filter** — edge-preserving refinement of the transmission map
   based on the grayscale input image, minimizing any blocky effects or
   edge halos in the objects.
5. **Sky Region Protection** — identifies potential sky pixels (higher value,
   lower saturation using the HSV color space) in the upper part of the image
   and weakens the dehazing process on them, as the sky does not comply with
   the dark channel assumption.
6. **Scene Radiance Recovery** — recovers the haze-free image using the
   standard atmospheric scattering model inversion.

## API

```cpp
cv::dehazeImage(src, dst);
```

Or call each stage individually for more control:

```cpp
cv::Mat dark, transmission, refined, dst;
cv::Vec3d A;

cv::computeDarkChannel(src, dark, 15);
cv::estimateAtmosphericLight(src, dark, A);
cv::computeTransmission(src, A, transmission, 15, 0.95);

cv::Mat gray;
cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);
cv::guidedFilter(gray, transmission, refined, 60, 0.0001);

cv::recoverSceneRadiance(src, refined, A, dst, 0.1);
```

## Known Limitations

- Works best with **uniform haze, with neutral colors** (regular fog/mist,
  white smog). For images containing **highly colored and spatially varying
  haze** (like orange/brown bands of smog), the output will contain some
  amount of color cast in the respective area. This is one of the drawbacks
  of the simple Dark Channel Prior haze removal algorithm since it works
  with only a single global atmospheric light color.
- Small color issues may be present in the sky areas even when sky
  protection technique is applied to highly hazy images.
- No benchmarks have been done against GPU/SIMD implementations;
  this is a CPU implementation only.

## Reference

He, K., Sun, J., & Tang, X. (2009). _Single Image Haze Removal Using Dark
Channel Prior._ IEEE CVPR.
