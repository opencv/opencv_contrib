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

void gl_check_error(const std::filesystem::path& file, unsigned int line, const char* expression) {
#ifndef NDEBUG
    int errorCode = glGetError();
//    cerr << "TRACE: " << file.filename() << " (" << line << ") : " << expression << " => code: " << errorCode << endl;
    if (errorCode != 0) {
        std::stringstream ss;
        ss << "GL failed in " << file.filename() << " (" << line << ") : " << "\nExpression:\n   "
                << expression << "\nError code:\n   " << errorCode;
        throw std::runtime_error(ss.str());
    }
#endif
}

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
#if !defined(__EMSCRIPTEN__) && !defined(OPENCV_V4D_USE_ES3)
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
    std::ostringstream oss;
    oss << "\n\t" << reinterpret_cast<const char*>(glGetString(GL_VERSION))
            << "\n\t" << reinterpret_cast<const char*>(glGetString(GL_RENDERER)) << endl;
    return oss.str();
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

class HTML5Capture {
private:
    cv::Ptr<V4D> window_;
    int width_;
    int height_;
    UMat fb_;
    GLuint framebuffer = 0;
    GLuint texture = 0;
public:
    HTML5Capture(cv::Ptr<V4D> window, int width, int height) :
        window_(window), width_(width), height_(height), fb_(cv::Size(width, height), CV_8UC4) {
        EM_ASM({
            globalThis.playing = false;
            globalThis.timeupdate = false;
            globalThis.v4dVideoElement = document.querySelector("#v4dVideoElement");
            globalThis.v4dCopyCanvasElement = document.createElement("canvas");
            globalThis.v4dCopyCanvasElement.id = "v4dCopyCanvasElement0";
            globalThis.v4dCopyCanvasElement.width = $0;
            globalThis.v4dCopyCanvasElement.height = $1;
            globalThis.v4dCopyCanvasElement.style.display = "none";
        }, width, height);
    }

    bool captureGPU(UMat& dst) {
        FrameBufferContext::GLScope scope(window_->fbCtx());

        int ret = EM_ASM_INT(
            if(typeof Module.ctx !== 'undefined' && Module.ctx != null && globalThis.doCapture) {
                globalThis.gl = Module.ctx;
                globalThis.v4dMainFrameBuffer = globalThis.gl.getParameter(globalThis.gl.FRAMEBUFFER_BINDING);
                globalThis.v4dMainTexture = globalThis.gl.getFramebufferAttachmentParameter(globalThis.gl.FRAMEBUFFER, globalThis.gl.COLOR_ATTACHMENT0, globalThis.gl.FRAMEBUFFER_ATTACHMENT_OBJECT_NAME);
                return 1;
            } else {
                return 0;
            }
        );
        if(ret) {
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
                globalThis.gl.framebufferTexture2D(globalThis.gl.DRAW_FRAMEBUFFER, globalThis.gl.COLOR_ATTACHMENT0, globalThis.gl.TEXTURE_2D, globalThis.v4dMainTexture, 0);
            );
            FrameBufferContext::FrameBufferScope fbScope(window_->fbCtx(), fb_);
            flip(fb_, fb_, 0);
            cvtColor(fb_, dst, COLOR_BGRA2RGB);

            return true;
        }
//        cerr << "not captured" << endl;
        return false;
    }

    void captureCPU() {
        EM_ASM(
            if(globalThis.doCapture) {
                if(typeof globalThis.v4dCopyCanvasContext === 'undefined' || globalThis.v4dCopyCanvasContext === null)
                    globalThis.v4dCopyCanvasContext = globalThis.v4dCopyCanvasElement.getContext('2d', { willReadFrequently: true });
                if(typeof globalThis.v4dFrameData === 'undefined' || globalThis.v4dFrameData === null)
                    globalThis.v4dFrameData = Module._malloc(width_ * height_ * 4);

                globalThis.v4dCopyCanvasElement.drawImage(globalThis.v4dVideoElement, 0, 0, 1280, 720);
                var cameraArrayBuffer = globalThis.v4dCopyCanvasContext.getImageData(0, 0, 1280, 720);
                Module.HEAPU8.set(cameraArrayBuffer.data, globalThis.v4dFrameData);
            }
        );
    }
};

cv::Ptr<HTML5Capture> capture = nullptr;
int capture_width = 0;
int capture_height = 0;

extern "C" {

EMSCRIPTEN_KEEPALIVE
void v4dInitCapture(int width, int height) {
    capture_width = width;
    capture_height = height;
}

}

Source makeCaptureSource(int width, int height, cv::Ptr<V4D> window) {
    using namespace std;

    return Source([=](cv::UMat& frame) {
        if(capture_width > 0 && capture_height > 0) {
            try {
                if(frame.empty())
                    frame.create(cv::Size(width, height), CV_8UC3);

                run_sync_on_main<17>([&](){
                    if(capture == nullptr)
                        capture = new HTML5Capture(window, capture_width, capture_height);
                    capture->captureGPU(frame);
                });
            } catch(std::exception& ex) {
                cerr << ex.what() << endl;
            }
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

