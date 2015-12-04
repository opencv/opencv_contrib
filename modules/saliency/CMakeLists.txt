set(the_description "Saliency API")
set(OPENCV_MODULE_IS_PART_OF_WORLD OFF)
add_definitions(-DOBJECTNESS_MODELS_PATH="${CMAKE_INSTALL_PREFIX}/${OPENCV_OTHER_INSTALL_PATH}/Objectness")

file(GLOB OBJECTNESS_TRAINED_MODELS samples/ObjectnessTrainedModel/*.yml.gz)

install(FILES ${OBJECTNESS_TRAINED_MODELS} DESTINATION ${OPENCV_OTHER_INSTALL_PATH}/Objectness COMPONENT libs)
install(DIRECTORY DESTINATION ${OPENCV_OTHER_INSTALL_PATH}/Objectness/Results)


ocv_define_module(saliency opencv_imgproc opencv_highgui opencv_features2d WRAP python)
