CXX      := g++
CXXFLAGS := -I/usr/local/include/ -std=c++20 -pthread -fno-strict-aliasing -pedantic -Wall -march=native -flto
LDFLAGS  := -L/usr/local/lib64/ -L/opt/local/lib -flto
.PHONY: all release debian-release info debug asan clean debian-clean distclean 
DESTDIR := /
PREFIX := /usr/local

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

dirs:
	${MAKE} -C src/tetra/ ${MAKEFLAGS} CXX=${CXX} ${MAKECMDGOALS}

debian-release:
	${MAKE} -C src/tetra/ ${MAKEFLAGS} CXX=${CXX} release

debian-clean:
	${MAKE} -C src/tetra/ ${MAKEFLAGS} CXX=${CXX} clean

install: ${TARGET}
	true

distclean:
	true

