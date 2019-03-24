# Tesseract OCR
if(COMMAND pkg_check_modules)
  pkg_check_modules(Tesseract tesseract lept)
endif()
if(NOT Tesseract_FOUND)
  find_path(Tesseract_INCLUDE_DIRS tesseract/baseapi.h
    HINTS
    /usr/include
    /usr/local/include)

  find_library(Tesseract_LIBRARIES NAMES tesseract
    HINTS
    /usr/lib
    /usr/local/lib)

  find_library(Tesseract_Lept_LIBRARY NAMES lept
    HINTS
    /usr/lib
    /usr/local/lib)

  if(Tesseract_INCLUDE_DIRS AND Tesseract_LIBRARIES AND Tesseract_Lept_LIBRARY)
    set(Tesseract_INCLUDE_DIRS ${Tesseract_INCLUDE_DIRS} CACHE INTERNAL)
    set(Tesseract_LIBRARIES ${Tesseract_LIBRARIES} ${Tesseract_Lept_LIBRARY} CACHE INTERNAL)
    set(Tesseract_FOUND 1)
  endif()
endif()
