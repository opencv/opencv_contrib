using Test

print("Loading module")

@show @time using OpenCV

# Test waitkey
t = @timed OpenCV.waitKey(Int32(500))

@show @test t[2]> 0.5

# Create a random image
@show img = rand(UInt8 , 3, 500, 500)

# Test input as AbstractArray and cvtColor
@show img_gray = OpenCV.cvtColor(img, OpenCV.COLOR_RGB2GRAY)

@show size(img_gray)

# Exception test
@show @test_throws ErrorException OpenCV.cvtColor(img_gray, OpenCV.COLOR_RGB2GRAY)

# Non-continous memory test
@show ve = view(img, :,200:300, 200:300)

# Auto-conversion from-to OpenCV types
@show ve_gray = OpenCV.cvtColor(ve, OpenCV.COLOR_RGB2GRAY)

# Sanity check
@show @test size(ve_gray)[1] == 1 && size(img_gray)[1] == 1

# If we got this far everything has passed. 
exit(0)