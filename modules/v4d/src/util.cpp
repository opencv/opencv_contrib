// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright Amir Hassan (kallaballa) <amir@viel-zu.org>

#include "opencv2/v4d/v4d.hpp"
#include <opencv2/imgcodecs.hpp>
#include "opencv2/v4d/util.hpp"
#include "opencv2/v4d/nvg.hpp"
#include "detail/framebuffercontext.hpp"


#ifdef __EMSCRIPTEN__
#  include <emscripten.h>
#  include <emscripten/bind.h>
#  include <SDL/SDL.h>
#  include <SDL/SDL_image.h>
#  include <SDL/SDL_stdinc.h>
#else
# include <opencv2/core/ocl.hpp>
#endif

#include <csignal>
#include <thread>
#include <unistd.h>
#include <chrono>
#include <mutex>
#include <functional>
#include <iostream>
#include <cmath>

namespace cv {
namespace v4d {
namespace detail {
size_t cnz(const cv::UMat& m) {
    cv::UMat grey;
    if(m.channels() == 1) {
        grey = m;
    } else if(m.channels() == 3) {
        cvtColor(m, grey, cv::COLOR_BGR2GRAY);
    } else if(m.channels() == 4) {
        cvtColor(m, grey, cv::COLOR_BGRA2GRAY);
    } else {
        assert(false);
    }
    return cv::countNonZero(grey);
}
}
#ifdef __EMSCRIPTEN__
Mat read_embedded_image(const string &path) {
    SDL_Surface *loadedSurface = IMG_Load(path.c_str());
    Mat result;
    if (loadedSurface == NULL) {
        printf("Unable to load image %s! SDL_image Error: %s\n", path.c_str(),
        IMG_GetError());
    } else {
        if (loadedSurface->w == 0 && loadedSurface->h == 0) {
            std::cerr << "Empty image loaded" << std::endl;
            SDL_FreeSurface(loadedSurface);
            return Mat();
        }
        if(loadedSurface->format->BytesPerPixel == 1) {
            result = Mat(loadedSurface->h, loadedSurface->w, CV_8UC1, (unsigned char*) loadedSurface->pixels, loadedSurface->pitch).clone();
            cvtColor(result,result, COLOR_GRAY2BGR);
        } else if(loadedSurface->format->BytesPerPixel == 3) {
            result = Mat(loadedSurface->h, loadedSurface->w, CV_8UC3, (unsigned char*) loadedSurface->pixels, loadedSurface->pitch).clone();
            if(loadedSurface->format->Rmask == 0x0000ff)
                cvtColor(result,result, COLOR_RGB2BGR);
        } else if(loadedSurface->format->BytesPerPixel == 4) {
            result = Mat(loadedSurface->h, loadedSurface->w, CV_8UC4, (unsigned char*) loadedSurface->pixels, loadedSurface->pitch).clone();
            if(loadedSurface->format->Rmask == 0x000000ff)
                cvtColor(result,result, COLOR_RGBA2BGR);
            else
                cvtColor(result,result, COLOR_RGBA2RGB);
        } else {
            std::cerr << "Unsupported image depth" << std::endl;
            SDL_FreeSurface(loadedSurface);
            return Mat();
        }
        SDL_FreeSurface(loadedSurface);
    }
    return result;
}
#endif

unsigned int initShader(const char* vShader, const char* fShader, const char* outputAttributeName) {
    struct Shader {
        GLenum type;
        const char* source;
    } shaders[2] = { { GL_VERTEX_SHADER, vShader }, { GL_FRAGMENT_SHADER, fShader } };

    GLuint program = glCreateProgram();

    for (int i = 0; i < 2; ++i) {
        Shader& s = shaders[i];
        GLuint shader = glCreateShader(s.type);
        glShaderSource(shader, 1, (const GLchar**) &s.source, NULL);
        glCompileShader(shader);

        GLint compiled;
        glGetShaderiv(shader, GL_COMPILE_STATUS, &compiled);
        if (!compiled) {
            std::cerr << " failed to compile:" << std::endl;
            GLint logSize;
            glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &logSize);
            char* logMsg = new char[logSize];
            glGetShaderInfoLog(shader, logSize, NULL, logMsg);
            std::cerr << logMsg << std::endl;
            delete[] logMsg;

            exit (EXIT_FAILURE);
        }

        glAttachShader(program, shader);
    }
#ifndef OPENCV_V4D_USE_ES3
    /* Link output */
    glBindFragDataLocation(program, 0, outputAttributeName);
#else
    CV_UNUSED(outputAttributeName);
#endif
    /* link  and error check */
    glLinkProgram(program);

    GLint linked;
    glGetProgramiv(program, GL_LINK_STATUS, &linked);
    if (!linked) {
        std::cerr << "Shader program failed to link" << std::endl;
        GLint logSize;
        glGetProgramiv(program, GL_INFO_LOG_LENGTH, &logSize);
        char* logMsg = new char[logSize];
        glGetProgramInfoLog(program, logSize, NULL, logMsg);
        std::cerr << logMsg << std::endl;
        delete[] logMsg;

        exit (EXIT_FAILURE);
    }

    /* use program object */
    glUseProgram(program);

    return program;
}

std::string getGlInfo() {
    return reinterpret_cast<const char*>(glGetString(GL_VERSION));
}

std::string getClInfo() {
    std::stringstream ss;
#ifndef __EMSCRIPTEN__
    std::vector<cv::ocl::PlatformInfo> plt_info;
    cv::ocl::getPlatfomsInfo(plt_info);
    const cv::ocl::Device& defaultDevice = cv::ocl::Device::getDefault();
    cv::ocl::Device current;
    ss << endl;
    for (const auto& info : plt_info) {
        for (int i = 0; i < info.deviceNumber(); ++i) {
            ss << "\t";
            info.getDevice(current, i);
            if (defaultDevice.name() == current.name())
                ss << "* ";
            else
                ss << "  ";
            ss << info.version() << " = " << info.name() << endl;
            ss << "\t\t  GL sharing: "
                    << (current.isExtensionSupported("cl_khr_gl_sharing") ? "true" : "false")
                    << endl;
            ss << "\t\t  VAAPI media sharing: "
                    << (current.isExtensionSupported("cl_intel_va_api_media_sharing") ?
                            "true" : "false") << endl;
        }
    }
#endif
    return ss.str();
}

bool isIntelVaSupported() {
#ifndef __EMSCRIPTEN__
    try {
        std::vector<cv::ocl::PlatformInfo> plt_info;
        cv::ocl::getPlatfomsInfo(plt_info);
        cv::ocl::Device current;
        for (const auto& info : plt_info) {
            for (int i = 0; i < info.deviceNumber(); ++i) {
                info.getDevice(current, i);
                return current.isExtensionSupported("cl_intel_va_api_media_sharing");
            }
        }
    } catch (std::exception& ex) {
        cerr << "Intel VAAPI query failed: " << ex.what() << endl;
    } catch (...) {
        cerr << "Intel VAAPI query failed" << endl;
    }
#endif
    return false;
}

bool isClGlSharingSupported() {
#ifndef __EMSCRIPTEN__
    try {
        std::vector<cv::ocl::PlatformInfo> plt_info;
        cv::ocl::getPlatfomsInfo(plt_info);
        cv::ocl::Device current;
        for (const auto& info : plt_info) {
            for (int i = 0; i < info.deviceNumber(); ++i) {
                info.getDevice(current, i);
                return current.isExtensionSupported("cl_khr_gl_sharing");
            }
        }
    } catch (std::exception& ex) {
        cerr << "CL-GL sharing query failed: " << ex.what() << endl;
    } catch (...) {
        cerr << "CL-GL sharing query failed with unknown error." << endl;
    }
#endif
    return false;
}

/*!
 * Internal variable that signals that finishing all operation is requested
 */
static bool finish_requested = false;
/*!
 * Internal variable that tracks if signal handlers have already been installed
 */
static bool signal_handlers_installed = false;

/*!
 * Signal handler callback that signals the application to terminate.
 * @param ignore We ignore the signal number
 */
static void request_finish(int ignore) {
    CV_UNUSED(ignore);
    finish_requested = true;
}

/*!
 * Installs #request_finish() as signal handler for SIGINT and SIGTERM
 */
static void install_signal_handlers() {
    signal(SIGINT, request_finish);
    signal(SIGTERM, request_finish);
}

bool keepRunning() {
    if (!signal_handlers_installed) {
        install_signal_handlers();
    }
    return !finish_requested;
}

#ifndef __EMSCRIPTEN__
Sink makeVaSink(const string& outputFilename, const int fourcc, const float fps,
        const cv::Size& frameSize, const int vaDeviceIndex) {
    cv::Ptr<cv::VideoWriter> writer = new cv::VideoWriter(outputFilename, cv::CAP_FFMPEG,
            cv::VideoWriter::fourcc('V', 'P', '9', '0'), fps, frameSize, {
                    cv::VIDEOWRITER_PROP_HW_DEVICE, vaDeviceIndex,
                    cv::VIDEOWRITER_PROP_HW_ACCELERATION, cv::VIDEO_ACCELERATION_VAAPI,
                    cv::VIDEOWRITER_PROP_HW_ACCELERATION_USE_OPENCL, 1 });

    return Sink([=](const cv::UMat& frame) {
        (*writer) << frame;
        return writer->isOpened();
    });
}

Source makeVaSource(const string& inputFilename, const int vaDeviceIndex) {
    cv::Ptr<cv::VideoCapture> capture = new cv::VideoCapture(inputFilename, cv::CAP_FFMPEG, {
            cv::CAP_PROP_HW_DEVICE, vaDeviceIndex, cv::CAP_PROP_HW_ACCELERATION,
            cv::VIDEO_ACCELERATION_VAAPI, cv::CAP_PROP_HW_ACCELERATION_USE_OPENCL, 1 });
    float fps = capture->get(cv::CAP_PROP_FPS);

    return Source([=](cv::UMat& frame) {
        (*capture) >> frame;
        return !frame.empty();
    }, fps);
}
#endif

#ifndef __EMSCRIPTEN__
Sink makeWriterSink(const string& outputFilename, const int fourcc, const float fps,
        const cv::Size& frameSize) {
    if (isIntelVaSupported()) {
        return makeVaSink(outputFilename, fourcc, fps, frameSize, 0);
    }

    cv::Ptr<cv::VideoWriter> writer = new cv::VideoWriter(outputFilename, cv::CAP_FFMPEG,
            cv::VideoWriter::fourcc('V', 'P', '9', '0'), fps, frameSize);

    return Sink([=](const cv::UMat& frame) {
        cv::resize(frame, frame, frameSize);
        (*writer) << frame;
        return writer->isOpened();
    });
}

Source makeCaptureSource(const string& inputFilename) {
    if (isIntelVaSupported()) {
        return makeVaSource(inputFilename, 0);
    }

    cv::Ptr<cv::VideoCapture> capture = new cv::VideoCapture(inputFilename, cv::CAP_FFMPEG);
    float fps = capture->get(cv::CAP_PROP_FPS);

    return Source([=](cv::UMat& frame) {
        (*capture) >> frame;
        return !frame.empty();
    }, fps);
}

#else

using namespace emscripten;
uint8_t* current_frame = nullptr;

extern "C" {

EMSCRIPTEN_KEEPALIVE
void v4dSetVideoFramePointer(uint8_t* frame, int width, int height) {
    assert(current_frame == nullptr);
    current_frame = frame;
//    memset(current_frame, 127, width * height * 4);
}
}

GLuint framebuffer = 0;
GLuint texture = 0;

bool captureVideoFrameGPU(int width, int height) {
    int ret = EM_ASM_INT(
        if(typeof Module.ctx !== 'undefined' && Module.ctx !== null && Module.doCapture) {
            globalThis.gl = Module.ctx;
            globalThis.v4dMainFrameBuffer = globalThis.gl.getParameter(globalThis.gl.FRAMEBUFFER_BINDING);
            globalThis.v4dMainTexture = globalThis.gl.getFramebufferAttachmentParameter(globalThis.gl.FRAMEBUFFER, globalThis.gl.COLOR_ATTACHMENT0, globalThis.gl.FRAMEBUFFER_ATTACHMENT_OBJECT_NAME);
            return 1;
        } else {
            return 0;
        }
    );

    if(ret) {
        EM_ASM(
            if(typeof globalThis.v4dVideoElement === 'undefined' || globalThis.v4dVideoElement === null) {
              globalThis.v4dVideoElement = document.querySelector("#video");
            }
        );

        if(framebuffer == 0) {
            GL_CHECK(glGenFramebuffers(1, &framebuffer));
        }

        GL_CHECK(glBindFramebuffer(GL_READ_FRAMEBUFFER, framebuffer));

        if(texture == 0) {
            GL_CHECK(glGenTextures(1, &texture));
        }

        GL_CHECK(glBindTexture(GL_TEXTURE_2D, texture));

        EM_ASM(
            const level = 0;
            const internalFormat = globalThis.gl.RGBA;
            const border = 0;
            const srcFormat = globalThis.gl.RGBA;
            const srcType = globalThis.gl.UNSIGNED_BYTE;
            globalThis.gl.texImage2D(
            globalThis.gl.TEXTURE_2D,
            level,
            internalFormat,
            srcFormat,
            srcType,
            globalThis.v4dVideoElement
            );
        );

        GL_CHECK(glFramebufferTexture2D(GL_READ_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, texture, 0));
        EM_ASM(
            globalThis.gl.bindFramebuffer(globalThis.gl.DRAW_FRAMEBUFFER, globalThis.v4dMainFrameBuffer);
            globalThis.gl.bindTexture(globalThis.gl.TEXTURE_2D, globalThis.v4dMainTexture);
            globalThis.gl.pixelStorei(globalThis.gl.UNPACK_FLIP_Y_WEBGL, true);
            globalThis.gl.framebufferTexture2D(globalThis.gl.DRAW_FRAMEBUFFER, globalThis.gl.COLOR_ATTACHMENT0, globalThis.gl.TEXTURE_2D, globalThis.v4dMainTexture, 0);
        );
        return true;
    }
    return false;
}

EM_JS(void,copyVideoFrameCPU,(int p), {
        if(Module.doCapture) {
            if(typeof Module.cameraCtx === 'undefined' || Module.cameraCtx === null)
                Module.cameraCtx = document.querySelector("#cameraCanvas").getContext('2d', { willReadFrequently: true });
            if(typeof Module.videoElement === 'undefined' || Module.videoElement === null)
                Module.videoElement = document.querySelector("#video");

            Module.cameraCtx.drawImage(Module.videoElement, 0, 0, 1280, 720);
            var cameraArrayBuffer = Module.cameraCtx.getImageData(0, 0, 1280, 720);

            if(typeof cameraArrayBuffer !== 'undefined') {
                Module.HEAPU8.set(cameraArrayBuffer.data, p);
            }
        }
});

Source makeCaptureSource(int width, int height, cv::Ptr<V4D> window) {
    using namespace std;

    return Source([=](cv::UMat& frame) {
        //FIXME
        static cv::UMat tmp(cv::Size(width, height), CV_8UC4);
        try {
            if(frame.empty())
                frame.create(cv::Size(width, height), CV_8UC3);

            if (current_frame != nullptr) {
                run_sync_on_main<17>([&](){
                    FrameBufferContext::GLScope scope(window->fbCtx());
                    if(captureVideoFrameGPU(width, height)) {
                        FrameBufferContext::FrameBufferScope fbScope(window->fbCtx(), tmp);
                        cvtColor(tmp, frame, COLOR_BGRA2RGB);
                    }
                });

//                run_sync_on_main<16>([&](){
//                    copyVideoFrameCPU(reinterpret_cast<int>(current_frame));
//                    cv::Mat tmp(cv::Size(width, height), CV_8UC4, current_frame);
//                    cv::UMat utmp = tmp.getUMat(ACCESS_READ);
//                    cvtColor(utmp, frame, cv::COLOR_BGRA2RGB);
//                    utmp.release();
//                    tmp.release();
//                });
            } else {
                std::cerr << "Nothing captured" << endl;
            }
        } catch(std::exception& ex) {
            cerr << ex.what() << endl;
        }
        return true;
    }, 0);
}

#endif

void resizePreserveAspectRatio(const cv::UMat& src, cv::UMat& output, const cv::Size& dstSize, const cv::Scalar& bgcolor) {
    cv::UMat tmp;
    double hf = double(dstSize.height) / src.size().height;
    double wf = double(dstSize.width) / src.size().width;
    double f = std::min(hf, wf);
    if (f < 0)
        f = 1.0 / f;

    cv::resize(src, tmp, cv::Size(), f, f);

    int top = (dstSize.height - tmp.rows) / 2;
    int down = (dstSize.height - tmp.rows + 1) / 2;
    int left = (dstSize.width - tmp.cols) / 2;
    int right = (dstSize.width - tmp.cols + 1) / 2;

    cv::copyMakeBorder(tmp, output, top, down, left, right, cv::BORDER_CONSTANT, bgcolor);
}

}
}

