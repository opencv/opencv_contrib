import cv2
import sys

if len(sys.argv) < 2:
    print('Usage: affine_detector.py img_path')
    exit()

img = cv2.imread(sys.argv[1])

detector = cv2.xfeatures2d.AffineFeature2D_create(
    cv2.xfeatures2d.HarrisLaplaceFeatureDetector_create(maxCorners=10000),
    cv2.xfeatures2d.SIFT_create()
)

kps = detector.detect(img, mask=None)
#kps, descrs = detector.detectAndCompute(img, mask=None)

print('Detected %u affine keypoints' % len(kps))
