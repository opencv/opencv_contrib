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
LIBS += `pkg-config --libs glfw3 glew` -lopencv_face -lopencv_gapi -lopencv_ml -lopencv_objdetect -lopencv_shape -lopencv_stitching -lopencv_superres -lopencv_optflow -lopencv_tracking -lopencv_highgui -lopencv_plot -lopencv_videostab -lopencv_videoio -lopencv_photo -lopencv_ximgproc -lopencv_video -lopencv_calib3d -lopencv_imgcodecs -lopencv_features2d -lopencv_dnn -lopencv_imgproc -lopencv_flann -lopencv_core -lGL -lOpenCL
endif
CXXFLAGS += -DCL_TARGET_OPENCL_VERSION=120
# -DVIZ2D_USE_ES3=1
ifdef EMSDK
CXX     := em++
EMCXXFLAGS += -flto -s USE_PTHREADS=1 -pthread -msimd128
EMLDFLAGS += -sUSE_GLFW=3 -sMIN_WEBGL_VERSION=2 -sMAX_WEBGL_VERSION=2 -sUSE_ZLIB=1 -sWASM=1 -sWASM_BIGINT -sINITIAL_MEMORY=512MB -sTOTAL_MEMORY=512MB -sUSE_PTHREADS=1 -pthread -sPTHREAD_POOL_SIZE=navigator.hardwareConcurrency -sLLD_REPORT_UNDEFINED --bind
CXXFLAGS += $(EMCXXFLAGS) -c
LDFLAGS += $(EMLDFLAGS)
endif

all: release

ifneq ($(UNAME_S), Darwin)
release: LDFLAGS += -s
endif
ifdef EMSDK
release: CXXFLAGS += -DNDEBUG -g0 -O3
release: LDFLAGS += -s STACK_OVERFLOW_CHECK=0 -s ASSERTIONS=0 -s SAFE_HEAP=0
endif
release: CXXFLAGS += -g0 -O3 -c
release: dirs

shrink: CXXFLAGS += -Os -w
shrink: LDFLAGS += -s
shrink: dirs

info: CXXFLAGS += -g3 -O0
info: LDFLAGS += -Wl,--export-dynamic -rdynamic
info: dirs

ifndef EMSDK
debug: CXXFLAGS += -rdynamic
debug: LDFLAGS += -rdynamic
else
debug: CXXFLAGS += -sDISABLE_EXCEPTION_CATCHING=0 -sDISABLE_EXCEPTION_THROWING=0 -fexceptions
debug: LDFLAGS += -s ASSERTIONS=2 -sLLD_REPORT_UNDEFINED=1 -sDISABLE_EXCEPTION_CATCHING=0 -sDISABLE_EXCEPTION_THROWING=0 -sEXCEPTION_DEBUG=1 -fexceptions
endif
debug: CXXFLAGS += -g3 -O0
debug: LDFLAGS += -Wl,--export-dynamic
debug: dirs

profile: CXXFLAGS += -g3 -O3
profile: LDFLAGS += -Wl,--export-dynamic
ifdef EMSDK
profile: LDFLAGS += --profiling
profile: CXXFLAGS += --profiling
endif
ifndef EMSDK
profile: CXXFLAGS += -rdynamic
endif
profile: dirs

ifdef EMSDK
unsafe: CXXFLAGS += -DNDEBUG -g0 -O3  --closure 1 -ffp-contract=fast -freciprocal-math -fno-signed-zeros
unsafe: LDFLAGS += -s STACK_OVERFLOW_CHECK=0 -s ASSERTIONS=0 -s SAFE_HEAP=0 --closure 1 -menable-unsafe-fp-math
else
unsafe: CXXFLAGS += -DNDEBUG -g0 -Ofast
endif
#ifeq ($(UNAME_S), Darwin)
unsafe: LDFLAGS += -s
#endif
unsafe: dirs

ifdef EMSDK
asan: CXXFLAGS += -fsanitize=address
asan: LDFLAGS += -s STACK_OVERFLOW_CHECK=2 -s ASSERTIONS=2 -s NO_DISABLE_EXCEPTION_CATCHING=1 -s EXCEPTION_DEBUG=1 -fsanitize=address
else
asan: CXXFLAGS += -rdynamic -fsanitize=address
asan: LDFLAGS += -rdynamic -fsanitize=address
endif
asan: CXXFLAGS += -g3 -O0 -fno-omit-frame-pointer
asan: LDFLAGS += -Wl,--export-dynamic -rdynamic
ifndef EMSDK
asan: LIBS+= -lbfd -ldw
endif
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
	${MAKE} -C src/shader/ ${MAKEFLAGS} CXX=${CXX} ${MAKECMDGOALS}
ifndef EMSDK
	${MAKE} -C src/tetra/ ${MAKEFLAGS} CXX=${CXX} ${MAKECMDGOALS}
	${MAKE} -C src/video/ ${MAKEFLAGS} CXX=${CXX} ${MAKECMDGOALS}
	${MAKE} -C src/nanovg/ ${MAKEFLAGS} CXX=${CXX} ${MAKECMDGOALS}
endif
	${MAKE} -C src/beauty/ ${MAKEFLAGS} CXX=${CXX} ${MAKECMDGOALS}
	${MAKE} -C src/optflow/ ${MAKEFLAGS} CXX=${CXX} ${MAKECMDGOALS}
	${MAKE} -C src/font/ ${MAKEFLAGS} CXX=${CXX} ${MAKECMDGOALS}
ifndef EMSDK
	${MAKE} -C src/pedestrian/ ${MAKEFLAGS} CXX=${CXX} ${MAKECMDGOALS}
endif

debian-release:
	${MAKE} -C src/common/ ${MAKEFLAGS} CXX=${CXX} release
	${MAKE} -C src/shader/ ${MAKEFLAGS} CXX=${CXX} release
	${MAKE} -C src/video/ ${MAKEFLAGS} CXX=${CXX} release
	${MAKE} -C src/nanovg/ ${MAKEFLAGS} CXX=${CXX} release
	${MAKE} -C src/optflow/ ${MAKEFLAGS} CXX=${CXX} release
	${MAKE} -C src/beauty/ ${MAKEFLAGS} CXX=${CXX} release
	${MAKE} -C src/font/ ${MAKEFLAGS} CXX=${CXX} release
	${MAKE} -C src/pedestrian/ ${MAKEFLAGS} CXX=${CXX} release

debian-clean:
	${MAKE} -C src/common/ ${MAKEFLAGS} CXX=${CXX} clean
	${MAKE} -C src/shader/ ${MAKEFLAGS} CXX=${CXX} clean
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

