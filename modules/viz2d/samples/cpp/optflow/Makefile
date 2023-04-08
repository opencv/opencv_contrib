ifdef EMSDK
TARGET := optflow-demo.js
else
TARGET := optflow-demo
endif

SRCS := optflow-demo.cpp

#precompiled headers
HEADERS := 
OBJS := ${SRCS:.cpp=.o}
DEPS := ${SRCS:.cpp=.dep} 

CXXFLAGS += -fpic
LIBS += -lm -lopencv_optflow -lopencv_video -lopencv_features2d -lviz2d -lz
ifndef EMSDK
LIBS += -lOpenCL
endif

.PHONY: all release debug clean distclean 

all: release
release: ${TARGET}
debug: ${TARGET}
info: ${TARGET}
profile: ${TARGET}
unsafe: ${TARGET}
asan: ${TARGET}

${TARGET}: ${OBJS}
	${CXX} ${LDFLAGS} -o $@ $^ ${LIBS}

${OBJS}: %.o: %.cpp %.dep ${GCH}
	${CXX} ${CXXFLAGS} -o $@ -c $<

${DEPS}: %.dep: %.cpp Makefile 
	${CXX} ${CXXFLAGS} -MM $< > $@ 

${GCH}: %.gch: ${HEADERS} 
	${CXX} ${CXXFLAGS} -o $@ -c ${@:.gch=.hpp}

install:
	mkdir -p ${DESTDIR}/${PREFIX}
	cp ${TARGET} ${DESTDIR}/${PREFIX}

uninstall:
	rm ${DESTDIR}/${PREFIX}/${TARGET}

clean:
	rm -f *~ ${DEPS} ${OBJS} ${CUO} ${GCH} ${TARGET} 

distclean: clean

