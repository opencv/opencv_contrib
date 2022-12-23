CXX      := g++
CXXFLAGS := -std=c++20 -Wno-deprecated-enum-enum-conversion -fno-strict-aliasing -pedantic -Wall -flto -I/usr/local/include/opencv4/ -I/usr/local/include/nanovg -I/usr/local/include/
ifdef EMSDK
LDFLAGS  := -flto -L/usr/local/lib/ -L../common
LIBS     := -lopencv_core -lopencv_imgproc -lnanogui
else
LDFLAGS  := -L/opt/local/lib -flto -L/usr/local/lib64 -L../common/ -L/usr/local/lib
LIBS     := -lnanogui
endif
.PHONY: all release debian-release info debug asan clean debian-clean distclean 
DESTDIR := /
PREFIX := /usr/local

ifndef EMSDK
LIBS += `pkg-config --libs glfw3 opencv4 glew`
endif

ifdef EMSDK
CXX     := em++
EMCXXFLAGS += -flto -sDISABLE_EXCEPTION_CATCHING=0 -sDISABLE_EXCEPTION_THROWING=0 -fexceptions
EMLDFLAGS += -s USE_GLFW=3 -s WASM=1 -s -s WASM_BIGINT -s LLD_REPORT_UNDEFINED=1 -s ALLOW_MEMORY_GROWTH=1 -sDISABLE_EXCEPTION_CATCHING=0 -sDISABLE_EXCEPTION_THROWING=0 -sEXCEPTION_DEBUG=1 -fexceptions
#LIBS += -lzlib -lopencv_calib3d -lopencv_core -lopencv_dnn -lopencv_features2d -lopencv_flann -lopencv_imgproc -lopencv_objdetect -lopencv_photo -lopencv_video -lopencv_objdetect -lopencv_face
EMCXXFLAGS += -msimd128
CXXFLAGS += $(EMCXXFLAGS) -c
LDFLAGS += $(EMLDFLAGS)
endif

all: release

release: CXXFLAGS += -g0 -O3
release: dirs

info: CXXFLAGS += -g3 -O0
info: dirs

debug: CXXFLAGS += -g3 -O0 -rdynamic
debug: dirs

profile: CXXFLAGS += -g3 -O1
profile: dirs

unsafe: CXXFLAGS += -g0 -Ofast -DNDEBUG -ffast-math -ftree-vectorizer-verbose=1 -funroll-loops -ftree-vectorize -fno-signed-zeros -fno-trapping-math -frename-registers
unsafe: dirs

asan: CXXFLAGS += -g3 -O0 -fno-omit-frame-pointer -fsanitize=address
asan: LDFLAGS += -fsanitize=address
asan: LIBS+= -lbfd -ldw
asan: dirs

clean: dirs

docs:
	doxygen Doxyfile

export LDFLAGS
export CXXFLAGS
export LIBS
export EMSDK

dirs: docs
	${MAKE} -C src/common/ ${MAKEFLAGS} CXX=${CXX} ${MAKECMDGOALS}
	${MAKE} -C src/tetra/ ${MAKEFLAGS} CXX=${CXX} ${MAKECMDGOALS}
	${MAKE} -C src/video/ ${MAKEFLAGS} CXX=${CXX} ${MAKECMDGOALS}
	${MAKE} -C src/nanovg/ ${MAKEFLAGS} CXX=${CXX} ${MAKECMDGOALS}
	${MAKE} -C src/optflow/ ${MAKEFLAGS} CXX=${CXX} ${MAKECMDGOALS}
	${MAKE} -C src/beauty/ ${MAKEFLAGS} CXX=${CXX} ${MAKECMDGOALS}
	${MAKE} -C src/font/ ${MAKEFLAGS} CXX=${CXX} ${MAKECMDGOALS}
	${MAKE} -C src/pedestrian/ ${MAKEFLAGS} CXX=${CXX} ${MAKECMDGOALS}

debian-release:
	${MAKE} -C src/common/ ${MAKEFLAGS} CXX=${CXX} release
	${MAKE} -C src/tetra/ ${MAKEFLAGS} CXX=${CXX} release
	${MAKE} -C src/video/ ${MAKEFLAGS} CXX=${CXX} release
	${MAKE} -C src/nanovg/ ${MAKEFLAGS} CXX=${CXX} release
	${MAKE} -C src/optflow/ ${MAKEFLAGS} CXX=${CXX} release
	${MAKE} -C src/beauty/ ${MAKEFLAGS} CXX=${CXX} release
	${MAKE} -C src/font/ ${MAKEFLAGS} CXX=${CXX} release
	${MAKE} -C src/pedestrian/ ${MAKEFLAGS} CXX=${CXX} release

debian-clean:
	${MAKE} -C src/common/ ${MAKEFLAGS} CXX=${CXX} clean
	${MAKE} -C src/tetra/ ${MAKEFLAGS} CXX=${CXX} clean
	${MAKE} -C src/video/ ${MAKEFLAGS} CXX=${CXX} clean
	${MAKE} -C src/nanovg/ ${MAKEFLAGS} CXX=${CXX} clean
	${MAKE} -C src/optflow/ ${MAKEFLAGS} CXX=${CXX} clean
	${MAKE} -C src/beauty/ ${MAKEFLAGS} CXX=${CXX} clean
	${MAKE} -C src/font/ ${MAKEFLAGS} CXX=${CXX} clean
	${MAKE} -C src/pedestrian/ ${MAKEFLAGS} CXX=${CXX} clean

install: ${TARGET}
	true

distclean:
	true

