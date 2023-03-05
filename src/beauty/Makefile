ifdef EMSDK
TARGET := beauty-demo.js
else
TARGET := beauty-demo
endif

SRCS    := beauty-demo.cpp

#precompiled headers
HEADERS := 
OBJS    := ${SRCS:.cpp=.o}
DEPS    := ${SRCS:.cpp=.dep} 

CXXFLAGS += -fpic
ifdef EMSDK
LDFLAGS += --preload-file assets/
endif
LIBS += -lm -lopencv_dnn -lopencv_face -lopencv_objdetect -lopencv_stitching -lviz2d
ifndef EMSDK
LIBS += -lOpenCL
else
LIBS += -lopencv_video -lopencv_tracking -llibprotobuf
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

