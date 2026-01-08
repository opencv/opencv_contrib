using OpenCV
const cv = OpenCV

img = rand(UInt8, 3, 500, 500)
filter = rand(Float32, 1, 5, 5)/25

out = OpenCV.filter2D(img, Int32(-1), filter)

cv.namedWindow("orig")
cv.namedWindow("out")

cv.imshow("orig", img)
cv.imshow("out", out)

cv.waitKey(Int32(0))