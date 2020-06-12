# Create a random image
img = rand(UInt8 , 3, 500, 500)

# Test input as AbstractArray and cvtColor
img_gray = OpenCV.cvtColor(img, OpenCV.COLOR_RGB2GRAY)

@test size(img_gray, 1) == 1 && size(img_gray, 2) == size(img, 2) && size(img_gray, 3) == size(img, 3)

# Exception test
try
    # This should throw an error
    OpenCV.cvtColor(img_gray, OpenCV.COLOR_RGB2GRAY)
    exit(1)
catch
    # Error caught so we can continue
end

ve = view(img, :,200:300, 200:300)

# Auto-conversion from-to OpenCV types
ve_gray = OpenCV.cvtColor(ve, OpenCV.COLOR_RGB2GRAY)

# Shape check
@test size(ve_gray)[1] == 1 && size(img_gray)[1] == 1



print("imgproc test passed\n")
