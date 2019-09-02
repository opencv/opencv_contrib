cmake_minimum_required (VERSION 3.0)
project (myproject)
set (CMAKE_CXX_STANDARD 11)

add_executable (alpha src/alphac.cpp)

find_package(PkgConfig)
if (PKG_CONFIG_FOUND)
  pkg_check_modules(GTK "gtk+-2.0")
  if (GTK_FOUND)    
    target_link_libraries(alpha ${GTK_LIBRARIES})
    add_definitions(${GTK_CFLAGS} ${GTK_CFLAGS_OTHER})
  endif()
endif()
