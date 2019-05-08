set(the_description "Image Quality Analysis API")
ocv_define_module(quality opencv_core opencv_imgproc opencv_ml WRAP python)

# add brisque model, range files to installation.  the tests and quality module use these files unless otherwise directed
file(GLOB QUALITY_MODEL_DATA samples/*.yml)
install(FILES ${QUALITY_MODEL_DATA} DESTINATION ${OPENCV_OTHER_INSTALL_PATH}/quality COMPONENT libs)