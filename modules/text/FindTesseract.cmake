# Tesseract OCR
unset(Tesseract_FOUND)

find_path(Tesseract_INCLUDE_DIR tesseract/baseapi.h
  HINTS
  /usr/include
  /usr/local/include)

find_library(Tesseract_LIBRARY NAMES tesseract
  HINTS
  /usr/lib
  /usr/local/lib)

find_library(Lept_LIBRARY NAMES lept
  HINTS
  /usr/lib
  /usr/local/lib)

set(Tesseract_LIBS ${Tesseract_LIBRARY} ${Lept_LIBRARY})
if(Tesseract_LIBS AND Tesseract_INCLUDE_DIR)
    set(Tesseract_FOUND 1)
endif()

        
