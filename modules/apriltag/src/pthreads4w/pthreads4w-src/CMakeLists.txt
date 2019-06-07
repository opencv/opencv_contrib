set(PTHREAD_API
    _ptw32.h
    pthread.h
    sched.h
    semaphore.h)

source_group(PTHREAD_API FILES ${PTHREAD_API})

set(PTHREAD_HEADERS
    config.h
    context.h
    implement.h
    need_errno.h)

set(PTHREAD_STATIC_SOURCES
    pthread.c
)

add_definitions( -DPTW32_BUILD_INLINED )

add_library(pthreads4w STATIC ${PTHREAD_STATIC_SOURCES} ${PTHREAD_API} ${PTHREAD_HEADERS})

target_compile_definitions(pthreads4w PUBLIC __PTW32_STATIC_LIB HAVE_CONFIG_H)

target_include_directories(pthreads4w PUBLIC
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/>"
    "$<INSTALL_INTERFACE:$<INSTALL_PREFIX>/${CMAKE_INSTALL_INCLUDEDIR}>/pthreads4w")
