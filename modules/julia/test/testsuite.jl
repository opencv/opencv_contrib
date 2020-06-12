print("Loading module\n")

using OpenCV
using Test


test_dir = joinpath(ENV["OPENCV_TEST_DATA_PATH"], "cv")

include("test_feature2d.jl")
include("test_imgproc.jl")
include("test_objdetect.jl")

exit(0)
