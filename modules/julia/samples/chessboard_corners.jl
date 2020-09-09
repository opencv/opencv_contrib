using OpenCV

const cv = OpenCV


# chess1.png is at https://raw.githubusercontent.com/opencv/opencv_extra/master/testdata/cv/cameracalibration/chess1.png
img = cv.imread("chess1.png",cv.IMREAD_GRAYSCALE)

# Find the chess board corners
ret, corners = cv.findChessboardCorners(img, cv.Size{Int32}(7,5))

# If found, add object points, image points (after refining them)
if ret
    img = cv.drawChessboardCorners(img, cv.Size{Int32}(7,5), corners,ret)
    cv.imshow("img",img)
    cv.waitKey(Int32(0))

    cv.destroyAllWindows()
end
