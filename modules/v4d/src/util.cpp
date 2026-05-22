// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright Amir Hassan (kallaballa) <amir@viel-zu.org>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/core/ocl.hpp>

#include "../include/opencv2/v4d/v4d.hpp"
#include "../include/opencv2/v4d/util.hpp"

#include <csignal>
#include <unistd.h>
#include <chrono>
#include <mutex>
#include <functional>
#include <iostream>
#include <cmath>

using std::cerr;
using std::endl;

namespace cv {
namespace v4d {
namespace detail {

#ifdef __GNUG__
std::string demangle(const char* name) {
    int status = -4; // some arbitrary value to eliminate the compiler warning
    std::unique_ptr<char, void(*)(void*)> res {
        abi::__cxa_demangle(name, NULL, NULL, &status),
        std::free
    };

    return (status==0) ? res.get() : name ;
}

#else
// does nothing if not g++
std::string demangle(const char* name) {
    return name;
}
#endif

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

CV_EXPORTS void copy_shared(const cv::UMat& src, cv::UMat& dst) {
	if(dst.empty())
		dst.create(src.size(), src.type());
	src.copyTo(dst.getMat(cv::ACCESS_READ));
}

cv::Scalar colorConvert(const cv::Scalar& src, cv::ColorConversionCodes code) {
    cv::Mat tmpIn(1, 1, CV_8UC3);
    cv::Mat tmpOut(1, 1, CV_8UC3);

    tmpIn.at<cv::Vec3b>(0, 0) = cv::Vec3b(src[0], src[1], src[2]);
    cvtColor(tmpIn, tmpOut, code);
    const cv::Vec3b& vdst = tmpOut.at<cv::Vec3b>(0, 0);
    cv::Scalar dst(vdst[0], vdst[1], vdst[2], src[3]);
    return dst;
}

void gl_check_error(const std::filesystem::path& file, unsigned int line, const char* expression) {
    int errorCode = glGetError();
//    cerr << "TRACE: " << file.filename() << " (" << line << ") : " << expression << " => code: " << errorCode << endl;
    if (errorCode != 0) {
        std::stringstream ss;
        ss << "GL failed in " << file.filename() << " (" << line << ") : " << "\nExpression:\n   "
                << expression << "\nError code:\n   " << errorCode;
        CV_LOG_WARNING(nullptr, ss.str());
    }
}

void initShader(unsigned int handles[3], const char* vShader, const char* fShader, const char* outputAttributeName) {
    struct Shader {
        GLenum type;
        const char* source;
    } shaders[2] = { { GL_VERTEX_SHADER, vShader }, { GL_FRAGMENT_SHADER, fShader } };

    GLuint program = glCreateProgram();
    handles[0] = program;

    for (int i = 0; i < 2; ++i) {
        Shader& s = shaders[i];
        GLuint shader = glCreateShader(s.type);
        handles[i + 1] = shader;
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
#if !defined(OPENCV_V4D_USE_ES3)
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

}

std::string getGlVendor()  {
    std::ostringstream oss;
        oss << reinterpret_cast<const char*>(glGetString(GL_VENDOR));
        return oss.str();
}

std::string getGlInfo() {
    std::ostringstream oss;
    oss << "\n\t" << reinterpret_cast<const char*>(glGetString(GL_VERSION))
            << "\n\t" << reinterpret_cast<const char*>(glGetString(GL_RENDERER)) << endl;
    return oss.str();
}

std::string getClInfo() {
    std::stringstream ss;
#ifdef HAVE_OPENCL
    if(cv::ocl::useOpenCL()) {
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
    }
#endif
    return ss.str();
}

bool isIntelVaSupported() {
#ifdef HAVE_OPENCL
	if(cv::ocl::useOpenCL()) {
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
	}
#endif
    return false;
}

bool isClGlSharingSupported() {
#ifdef HAVE_OPENCL
	if(cv::ocl::useOpenCL()) {
		try {
			if(!cv::ocl::useOpenCL())
				return false;
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
	}
#endif
    return false;
}
static std::mutex finish_mtx;
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
	std::lock_guard guard(finish_mtx);
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
	std::lock_guard guard(finish_mtx);
    if (!signal_handlers_installed) {
        install_signal_handlers();
    }
    return !finish_requested;
}

void requestFinish() {
	request_finish(0);
}

cv::Ptr<Sink> makeVaSink(cv::Ptr<V4D> window, const string& outputFilename, const int fourcc, const float fps,
        const cv::Size& frameSize, const int vaDeviceIndex) {
    cv::Ptr<cv::VideoWriter> writer = new cv::VideoWriter(outputFilename, cv::CAP_FFMPEG,
            fourcc, fps, frameSize, {
                    cv::VIDEOWRITER_PROP_HW_DEVICE, vaDeviceIndex,
                    cv::VIDEOWRITER_PROP_HW_ACCELERATION, cv::VIDEO_ACCELERATION_VAAPI,
                    cv::VIDEOWRITER_PROP_HW_ACCELERATION_USE_OPENCL, 1 });
    if(isIntelVaSupported())
        window->sourceCtx()->copyContext();

    cerr << "Using a VA sink" << endl;
    if(writer->isOpened()) {
		return new Sink([=](const uint64_t& seq, const cv::UMat& frame) {
			CV_UNUSED(seq);
	        CLExecScope_t scope(window->sourceCtx()->getCLExecContext());
	        //FIXME cache it
            cv::UMat converted;
            cv::resize(frame, converted, frameSize);
            cvtColor(converted, converted, cv::COLOR_BGRA2RGB);
            (*writer) << converted;
			return writer->isOpened();
		});
    } else {
    	return new Sink();
    }
}

cv::Ptr<Source> makeVaSource(cv::Ptr<V4D> window, const string& inputFilename, const int vaDeviceIndex) {
    cv::Ptr<cv::VideoCapture> capture = new cv::VideoCapture(inputFilename, cv::CAP_FFMPEG, {
            cv::CAP_PROP_HW_DEVICE, vaDeviceIndex, cv::CAP_PROP_HW_ACCELERATION,
            cv::VIDEO_ACCELERATION_VAAPI, cv::CAP_PROP_HW_ACCELERATION_USE_OPENCL, 1 });
    float fps = capture->get(cv::CAP_PROP_FPS);
    cerr << "Using a VA source" << endl;
    if(isIntelVaSupported())
        window->sourceCtx()->copyContext();

    return new Source([=](cv::UMat& frame) {
        CLExecScope_t scope(window->sourceCtx()->getCLExecContext());
        (*capture) >> frame;
        return !frame.empty();
    }, fps);
}

static cv::Ptr<Sink> makeAnyHWSink(const string& outputFilename, const int fourcc, const float fps,
        const cv::Size& frameSize) {
    cv::Ptr<cv::VideoWriter> writer = new cv::VideoWriter(outputFilename, cv::CAP_FFMPEG,
            fourcc, fps, frameSize, { cv::VIDEOWRITER_PROP_HW_ACCELERATION, cv::VIDEO_ACCELERATION_ANY });

    if(writer->isOpened()) {
        return new Sink([=](const uint64_t& seq, const cv::UMat& frame) {
        	CV_UNUSED(seq);
            cv::UMat converted;
            cv::UMat context_corrected;
            frame.copyTo(context_corrected);
            cv::resize(context_corrected, converted, frameSize);
            cvtColor(converted, converted, cv::COLOR_BGRA2RGB);
            (*writer) << converted;
            return writer->isOpened();
        });
    } else {
        return new Sink();
    }
}

static cv::Ptr<Source> makeAnyHWSource(const string& inputFilename) {
    cv::Ptr<cv::VideoCapture> capture = new cv::VideoCapture(inputFilename, cv::CAP_FFMPEG, {
            cv::CAP_PROP_HW_ACCELERATION, cv::VIDEO_ACCELERATION_ANY });
    float fps = capture->get(cv::CAP_PROP_FPS);

    return new Source([=](cv::UMat& frame) {
        (*capture) >> frame;
        return !frame.empty();
    }, fps);
}

cv::Ptr<Sink> makeWriterSink(cv::Ptr<V4D> window, const string& outputFilename, const float fps, const cv::Size& frameSize) {
    int fourcc = 0;
    //FIXME find a cleverer way to guess a decent codec
    if(getGlVendor() == "NVIDIA Corporation") {
        fourcc = cv::VideoWriter::fourcc('H', '2', '6', '4');
    } else {
        fourcc = cv::VideoWriter::fourcc('V', 'P', '9', '0');
    }
    return makeWriterSink(window, outputFilename, fps, frameSize, fourcc);
}

cv::Ptr<Sink> makeWriterSink(cv::Ptr<V4D> window, const string& outputFilename, const float fps,
        const cv::Size& frameSize, int fourcc) {
    if (isIntelVaSupported()) {
        return makeVaSink(window, outputFilename, fourcc, fps, frameSize, 0);
    } else {
        try {
            return makeAnyHWSink(outputFilename, fourcc, fps, frameSize);
        } catch(...) {
            cerr << "Failed creating hardware source" << endl;
        }
    }

    cv::Ptr<cv::VideoWriter> writer = new cv::VideoWriter(outputFilename, cv::CAP_FFMPEG,
            fourcc, fps, frameSize);

    if(writer->isOpened()) {
		return new Sink([=](const uint64_t& seq, const cv::UMat& frame) {
			CV_UNUSED(seq);
            cv::UMat converted;
            cv::resize(frame, converted, frameSize);
            cvtColor(converted, converted, cv::COLOR_BGRA2RGB);
            (*writer) << converted;
			return writer->isOpened();
		});
    } else {
    	return new Sink();
    }
}

cv::Ptr<Source> makeCaptureSource(cv::Ptr<V4D> window, const string& inputFilename) {
    if (isIntelVaSupported()) {
        return makeVaSource(window, inputFilename, 0);
    } else {
        try {
            return makeAnyHWSource(inputFilename);
        } catch(...) {
            cerr << "Failed creating hardware source" << endl;
        }
    }

    cv::Ptr<cv::VideoCapture> capture = new cv::VideoCapture(inputFilename, cv::CAP_FFMPEG);
    float fps = capture->get(cv::CAP_PROP_FPS);

    return new Source([=](cv::UMat& frame) {
        (*capture) >> frame;
        return !frame.empty();
    }, fps);
}

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

