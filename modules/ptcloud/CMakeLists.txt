set(the_description "Point Cloud Object Fitting API")
ocv_define_module(ptcloud opencv_core opencv_highgui opencv_viz opencv_surface_matching WRAP python)

# add test data from samples dir to contrib/ptcloud
ocv_add_testdata(samples/ contrib/ptcloud FILES_MATCHING PATTERN "*.ply")

# add data point cloud files to installation
file(GLOB POINTCLOUD_DATA samples/*.ply)
install(FILES ${POINTCLOUD_DATA} DESTINATION ${OPENCV_OTHER_INSTALL_PATH}/ptcloud COMPONENT libs)