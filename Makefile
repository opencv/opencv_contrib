CXX      := g++
CXXFLAGS := -std=c++20 -pthread -fno-strict-aliasing -pedantic -Wall -march=native -flto -I/usr/local/include/opencv4/
LDFLAGS  := -L/opt/local/lib -flto -L/usr/local/lib64
LIBS     := -lnanovg
.PHONY: all release debian-release info debug asan clean debian-clean distclean 
DESTDIR := /
PREFIX := /usr/local

ifdef GCV_ONLY_X11
CXXFLAGS += -D_GCV_ONLY_X11
else
LIBS += `pkg-config --libs glfw3`
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

export LDFLAGS
export CXXFLAGS
export LIBS

dirs:
	${MAKE} -C src/tetra/ ${MAKEFLAGS} CXX=${CXX} ${MAKECMDGOALS}
	${MAKE} -C src/video/ ${MAKEFLAGS} CXX=${CXX} ${MAKECMDGOALS}
	${MAKE} -C src/nanovg/ ${MAKEFLAGS} CXX=${CXX} ${MAKECMDGOALS}
	${MAKE} -C src/optflow/ ${MAKEFLAGS} CXX=${CXX} ${MAKECMDGOALS}
	${MAKE} -C src/beauty/ ${MAKEFLAGS} CXX=${CXX} ${MAKECMDGOALS}
	${MAKE} -C src/font/ ${MAKEFLAGS} CXX=${CXX} ${MAKECMDGOALS}
	${MAKE} -C src/pedestrian/ ${MAKEFLAGS} CXX=${CXX} ${MAKECMDGOALS}

debian-release:
	${MAKE} -C src/tetra/ ${MAKEFLAGS} CXX=${CXX} release
	${MAKE} -C src/video/ ${MAKEFLAGS} CXX=${CXX} release
	${MAKE} -C src/nanovg/ ${MAKEFLAGS} CXX=${CXX} release
	${MAKE} -C src/optflow/ ${MAKEFLAGS} CXX=${CXX} release
	${MAKE} -C src/beauty/ ${MAKEFLAGS} CXX=${CXX} release
	${MAKE} -C src/font/ ${MAKEFLAGS} CXX=${CXX} release
	${MAKE} -C src/pedestrian/ ${MAKEFLAGS} CXX=${CXX} release

debian-clean:
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

