import cv2

# Read the image
img = cv2.imread('opencv-4.5.2/samples/data/fruits.jpg')

# Execute letterbox resize
test = cv2.ximgproc.letterboxResize(img, (300, 600), 1, 0, (128, 0, 0))

# Display the image
cv2.imshow('test', test)

# Wait till keypress
cv2.waitKey(0)