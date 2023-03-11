ifdef EMSDK
TARGET := libviz2d.so
else
TARGET := libviz2d.so
endif

SRCS    := detail/clglcontext.cpp detail/clvacontext.cpp detail/nanovgcontext.cpp viz2d.cpp util.cpp nvg.cpp source.cpp sink.cpp formhelper.cpp dialog.cpp

#precompiled headers
HEADERS := 
OBJS    := ${SRCS:.cpp=.o}
DEPS    := ${SRCS:.cpp=.dep} 

CXXFLAGS += -fpic
LDFLAGS += -shared
LIBS += -lm
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

