# OpenNL
unset(OPENNL_FOUND)

find_path(OPENNL_INCLUDE_DIR NL/nl.h
  HINTS
  /usr/include
  /usr/local/include)

find_library(OPENNL_LIBRARY NAMES nl
  HINTS
  /usr/lib
  /usr/local/lib)

set(OPENNL_LIBS ${OPENNL_LIBRARY})
if(OPENNL_LIBS AND OPENNL_INCLUDE_DIR)
    set(OPENNL_FOUND 1)
endif()

        
