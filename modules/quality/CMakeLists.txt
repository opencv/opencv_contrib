set(the_description "Image Quality Analysis API")
ocv_define_module(quality opencv_core opencv_imgproc opencv_ml WRAP python)

# add test data from samples dir to contrib/quality
ocv_add_testdata(samples/ contrib/quality FILES_MATCHING PATTERN "*.yml")

# add brisque model, range files to installation
file(GLOB QUALITY_MODEL_DATA samples/*.yml)
install(FILES ${QUALITY_MODEL_DATA} DESTINATION ${OPENCV_OTHER_INSTALL_PATH}/quality COMPONENT libs)