// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright Amir Hassan (kallaballa) <amir@viel-zu.org>

#include "opencv2/v4d/v4d.hpp"
#include <opencv2/imgcodecs.hpp>
#include "opencv2/v4d/util.hpp"
#include "opencv2/v4d/nvg.hpp"


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
void run_sync_on_main(std::function<void()> fn) {
#ifdef __EMSCRIPTEN__
    emscripten_sync_run_in_main_runtime_thread(EM_FUNC_SIG_V, cv::v4d::detail::get_fn_ptr<__COUNTER__>(fn));
#else
    fn();
#endif
}

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

//EM_JS(void,JScopyVideoFrameGPU,(int width, int height), {
//
//        function initFramebuffer(gl) {
//          console.log("init fb: " + width + "/" + height);
//            if(typeof globalThis.v4dVideoFrameBuffer === 'undefined' || globalThis.v4dVideoFrameBuffer === null) {
//                console.log("CREATE FRAMEBUFFER");
//                globalThis.v4dVideoFrameBuffer = gl.createFramebuffer();
//            }
//          if(typeof globalThis.v4dVideoFrameTexture === 'undefined' || globalThis.v4dVideoFrameTexture === null) {
//              console.log("CREATE TEXTURE");
//              globalThis.v4dVideoFrameTexture = gl.createTexture();
//          }
//
//          if(typeof globalThis.v4dVideoElement === 'undefined' || globalThis.v4dVideoElement === null) {
//              console.log("CREATE VIDEO ELEMENT");
//              globalThis.v4dVideoElement = document.querySelector("#video");
//          }
//
//          gl.bindFramebuffer(gl.READ_FRAMEBUFFER, globalThis.v4dVideoFrameBuffer);
//          gl.bindTexture(gl.TEXTURE_2D, globalThis.v4dVideoFrameTexture);
//          const level = 0;
//          const internalFormat = gl.RGBA;
//          const border = 0;
//          const srcFormat = gl.RGBA;
//          const srcType = gl.UNSIGNED_BYTE;
//          gl.texImage2D(
//            gl.TEXTURE_2D,
//            level,
//            internalFormat,
//            srcFormat,
//            srcType,
//            globalThis.v4dVideoElement
//          );
//
//          gl.framebufferTexture2D(gl.READ_FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, globalThis.v4dVideoFrameTexture, 0);
//          return gl.checkFramebufferStatus(gl.READ_FRAMEBUFFER) == gl.FRAMEBUFFER_COMPLETE;
//        }
//
//        function updateTexture(gl) {
//          if(initFramebuffer(gl)) {
//              console.log("video texture copy: " + globalThis.v4dVideoFrameTexture + " -> " + mainTexture);
//              gl.bindFramebuffer(gl.DRAW_FRAMEBUFFER, mainFrameBuffer);
//              gl.bindTexture(gl.TEXTURE_2D, mainTexture);
//              gl.framebufferTexture2D(gl.DRAW_FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, mainTexture, 0);
//
//              gl.blitFramebuffer( 0, 0, globalThis.v4dVideoElement.width, globalThis.v4dVideoElement.height,
//                      0, 0, width, height,
//                      gl.COLOR_BUFFER_BIT, gl.NEAREST);
////              gl.copyTexImage2D(gl.TEXTURE_2D, 0, gl["RGBA"], 0, 0, width, height, 0);
//          } else {
//              console.log("frambuffer incomplete");
//          }
//        }
//
//        var ctx;
//        if (typeof GL !== 'undefined' && typeof Module.ctx !== 'undefined') {
//            gl = Module.ctx;
//            mainFrameBuffer = gl.getParameter(gl.FRAMEBUFFER_BINDING);
//            mainTexture = gl.getFramebufferAttachmentParameter(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.FRAMEBUFFER_ATTACHMENT_OBJECT_NAME);
//            console.log(mainFrameBuffer + ":" + mainTexture);
//
//            if(Module.doCapture) {
//                gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, true);
//                updateTexture(gl, globalThis.v4dVideoFrameTexture, globalThis.v4dVideoElement);
//            } else {
//                console.log("video not playing");
//            }
//            console.log("6");
//        } else {
//            console.log("GL unavailable");
//        }
//
//        gl.flush();
//        gl.finish();
//});

EM_JS(void,copyVideoFrame,(int p), {
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
        } else {
            console.log("Camery not ready");
        }
});

long acc = 0;
Source makeCaptureSource(int width, int height) {
    using namespace std;

    return Source([=](cv::UMat& frame) {
        try {
            frame.create(cv::Size(width, height), CV_8UC3);

            if (current_frame != nullptr) {
                cv::Mat tmp(cv::Size(width, height), CV_8UC4, current_frame);
                //we have to read from the array to make sure the data "appears". this seems to be some kind of caching problem.
                //i tried volatile statements but that didn't work.
                cerr << tmp.data[0] << tmp.data[width * height - 1];
                cv::Mat v = frame.getMat(cv::ACCESS_WRITE);
                cvtColor(tmp, v, cv::COLOR_BGRA2RGB);
                v.release();

                run_sync_on_main([=](){
                    copyVideoFrame(reinterpret_cast<int>(current_frame));
                });
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

