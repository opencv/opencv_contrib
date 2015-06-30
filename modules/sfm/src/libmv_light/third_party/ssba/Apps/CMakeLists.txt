include_directories(../Math ../Geometry)

add_executable(bundle_common bundle_common.cpp)
add_executable(bundle_varying bundle_varying.cpp)
target_link_libraries(bundle_common V3D ldl colamd)
target_link_libraries(bundle_varying V3D ldl colamd)
