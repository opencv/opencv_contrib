add_subdirectory(correspondence)
add_subdirectory(multiview)
add_subdirectory(numeric)

if ( Ceres_FOUND )
    add_subdirectory(simple_pipeline)
endif ()