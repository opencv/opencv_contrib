import cv2
import sys
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--image', required=True)
parser.add_argument('--model', required=True)
parser.add_argument('--proto', required=True)
args = parser.parse_args()

image = cv2.imread(args.image)
if image is None:
    print(f"Cannot load image: {args.image}")
    sys.exit(1)

detector = cv2.hed.HEDDetector.create(args.model, args.proto)
edges = detector.detectEdges(image)
edges_u8 = (edges * 255).astype('uint8')

gray  = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
canny = cv2.Canny(gray, 100, 200)

cv2.imshow("Original",        image)
cv2.imshow("Canny (classic)", canny)
cv2.imshow("HED (yours)",     edges_u8)
cv2.imwrite("hed_result.png", edges_u8)
print("Saved hed_result.png")
cv2.waitKey(0)