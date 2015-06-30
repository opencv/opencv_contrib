#Install macro for libmv libraries
MACRO (LIBMV_INSTALL_LIB name)

  set_target_properties( ${name}
    PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/lib
  )

ENDMACRO (LIBMV_INSTALL_LIB)


#Install macro for third parties libraries
MACRO (LIBMV_INSTALL_THIRD_PARTY_LIB name)

  set_target_properties( ${name}
    PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/lib
  )

ENDMACRO (LIBMV_INSTALL_THIRD_PARTY_LIB)
