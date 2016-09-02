#Caffe
unset(QT5_FOUND)

find_path(QT5_INCLUDE_DIR NAMES qt5/QtGui/QFontMetrics qt5/QtGui/QFont qt5/QtGui/QFontDatabase qt5/QtGui/QGuiApplication
  HINTS
  /usr/local/include)

find_library(Caffe_LIBS NAMES caffe
  HINTS
  /usr/local/lib)

if(Caffe_LIBS AND Caffe_INCLUDE_DIR)
    set(Caffe_FOUND 1)
endif()
