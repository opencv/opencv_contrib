using OpenCV

const cv = OpenCV

file = download("https://raw.githubusercontent.com/opencv/opencv_extra/master/testdata/cv/cameracalibration/chess1.png")
img = cv.imread(file,cv.IMREAD_GRAYSCALE)
climg = cv.cvtColor(img, cv.COLOR_GRAY2BGR)

# Find the chess board corners
ret, corners = cv.findChessboardCorners(img, cv.Size{Int32}(7,5))

# If found, add object points, image points (after refining them)
if ret
    climg = cv.drawChessboardCorners(climg, cv.Size{Int32}(7,5), corners,ret)
    cv.imshow("img",climg)
    cv.waitKey(Int32(0))

    cv.destroyAllWindows()
end
