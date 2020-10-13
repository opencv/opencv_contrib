##   OpenCV Tree-Based Morse REgions

Author and maintainers: Steffen Ehrle (steffen.ehrle@myestro.de).

Tree Based Morse Regions (TBMR) is a topological approach to local invariant feature detection motivated by Morse theory.

The algorithm can be seen as a variant of [MSER](https://docs.opencv.org/3.4.2/d3/d28/classcv_1_1MSER.html) as both algorithms rely on Min/Max-tree.
However TBMRs being purely topological are invariant to affine illumination change.

For more details about the algorithm, please refer to the original paper: [6940260]

### usage

c++ interface:

```c++
using namespace tbmr;
// read a image
Mat img = imread(image_path), res;

// create engine
// TBMR uses only 2 parameters: MinSize and MaxSizeRel.
// MinSize, pruning areas smaller than MinSize
// MaxSizeRel, pruning areas bigger than MaxSize = MaxSizeRel * img.size
Ptr<TBMR> alg = TBMR::create(30, 0.01);

// perform Keypoint Extraction
// TBMR features provide Point, diameter and angle
std::vector<Keypoint> tbmrs;
res = alg->detect(img, tbmrs);
```

### Reference

[6940260]: Y. Xu and P. Monasse and T. Géraud and L. Najman Tree-Based Morse Regions: A Topological Approach to Local Feature Detection.