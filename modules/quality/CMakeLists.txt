set(the_description "Image Quality Analysis API")
ocv_define_module(quality opencv_core opencv_imgproc opencv_ml WRAP python)
ocv_add_testdata(samples/ contrib/quality
    FILES_MATCHING PATTERN "*.yml"
)