if(IOS)
    ocv_module_disable(adas)
endif()

set(the_description "Automatic driver assistance algorithms")
ocv_define_module(adas opencv_xobjdetect)

add_subdirectory(tools)
