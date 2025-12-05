// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright Amir Hassan (kallaballa) <amir@viel-zu.org>

#ifndef SRC_OPENCV_V4D_UTIL_HPP_
#define SRC_OPENCV_V4D_UTIL_HPP_

#include "source.hpp"
#include "sink.hpp"
#include <filesystem>
#include <string>
#include <iostream>
#ifdef __GNUG__
#include <cstdlib>
#include <memory>
#include <cxxabi.h>
#endif
#include <opencv2/core/ocl.hpp>
#include <opencv2/imgproc.hpp>

#include <unistd.h>
#include <mutex>
#include <functional>
#include <iostream>
#include <cmath>
#include <thread>

namespace cv {
namespace v4d {
namespace detail {

using std::cout;
using std::endl;

inline uint64_t get_epoch_nanos() {
	return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
}

static thread_local std::mutex mtx_;

class CV_EXPORTS ThreadLocal {
public:
	CV_EXPORTS static std::mutex& mutex() {
    	return mtx_;
    }
};

class CV_EXPORTS Global {
	inline static std::mutex global_mtx_;

	inline static std::mutex frame_cnt_mtx_;
	inline static uint64_t frame_cnt_ = 0;

	inline static std::mutex start_time_mtx_;
	inline static uint64_t start_time_ = get_epoch_nanos();

	inline static std::mutex fps_mtx_;
	inline static double fps_ = 0;

	inline static std::mutex thread_id_mtx_;
	inline static const std::thread::id default_thread_id_;
	inline static std::thread::id main_thread_id_;
	inline static thread_local bool is_main_;

	inline static uint64_t run_cnt_ = 0;
	inline static bool first_run_ = true;

	inline static size_t workers_ready_ = 0;
    inline static size_t workers_started_ = 0;
    inline static size_t next_worker_idx_ = 0;
	inline static std::mutex sharedMtx_;
	inline static std::map<size_t, std::mutex*> shared_;
	typedef typename std::map<size_t, std::mutex*>::iterator Iterator;
public:
	template <typename T>
	class Scope {
	private:
		const T& t_;

//		ocl::OpenCLExecutionContext* pSavedExecCtx_ = nullptr;
//		ocl::OpenCLExecutionContext* pExecCtx_ = nullptr;
//
//		template<typename Tunused> void bind(const Tunused& t) {
//			//do nothing for all other types the UMat
//			CV_UNUSED(t);
//		}
//
//		void bind(const cv::UMat& t) {
//#ifdef HAVE_OPENCL
//			if(ocl::useOpenCL()) {
//				pExecCtx_ = (t.u && t.u->allocatorContext) ? static_cast<ocl::OpenCLExecutionContext*>(t.u->allocatorContext.get()) : nullptr;
//				if(pExecCtx_ && !pExecCtx_->empty()) {
//					pSavedExecCtx_ = &ocl::OpenCLExecutionContext::getCurrentRef();
//					pExecCtx_->bind();
//				} else {
//					pSavedExecCtx_ = nullptr;
//				}
//			}
//#endif
//		}
//
//		template<typename Tunused> void unbind(const Tunused& t) {
//			//do nothing for all other types the UMat
//			CV_UNUSED(t);
//		}
//
//		void unbind(const cv::UMat& t) {
//			CV_UNUSED(t);
//#ifdef HAVE_OPENCL
//	        if(ocl::useOpenCL() && pSavedExecCtx_ && !pSavedExecCtx_->empty()) {
//	        	pSavedExecCtx_->bind();
//	        }
//#endif
//		}

public:

		Scope(const T& t) : t_(t) {
			lock(t_);
//			bind(t_);
		}

		~Scope() {
			unlock(t_);
//			unbind(t_);
		}
	};

	CV_EXPORTS static std::mutex& mutex() {
    	return global_mtx_;
    }

	CV_EXPORTS static uint64_t next_frame_cnt() {
	    std::unique_lock<std::mutex> lock(frame_cnt_mtx_);
    	return frame_cnt_++;
    }

	CV_EXPORTS static uint64_t frame_cnt() {
	    std::unique_lock<std::mutex> lock(frame_cnt_mtx_);
    	return frame_cnt_;
    }

	CV_EXPORTS static void mul_frame_cnt(const double& factor) {
	    std::unique_lock<std::mutex> lock(frame_cnt_mtx_);
    	frame_cnt_ *= factor;
    }

	CV_EXPORTS static void add_to_start_time(const size_t& st) {
		std::unique_lock<std::mutex> lock(start_time_mtx_);
		start_time_ += st;
    }

	CV_EXPORTS static uint64_t start_time() {
		std::unique_lock<std::mutex> lock(start_time_mtx_);
        return start_time_;
    }

	CV_EXPORTS static double fps() {
		std::unique_lock<std::mutex> lock(fps_mtx_);
    	return fps_;
    }

	CV_EXPORTS static void set_fps(const double& f) {
		std::unique_lock<std::mutex> lock(fps_mtx_);
    	fps_ = f;
    }

	CV_EXPORTS static void set_main_id(const std::thread::id& id) {
		std::unique_lock<std::mutex> lock(thread_id_mtx_);
		main_thread_id_ = id;
    }

	CV_EXPORTS static const bool is_main() {
		std::unique_lock<std::mutex> lock(start_time_mtx_);
		return (main_thread_id_ == default_thread_id_ || main_thread_id_ == std::this_thread::get_id());
	}

	CV_EXPORTS static bool is_first_run() {
		static std::mutex mtx;
		std::unique_lock<std::mutex> lock(mtx);
    	bool f = first_run_;
    	first_run_ = false;
		return f;
    }

	CV_EXPORTS static uint64_t next_run_cnt() {
	    static std::mutex mtx;
	    std::unique_lock<std::mutex> lock(mtx);
    	return run_cnt_++;
    }

	CV_EXPORTS static void set_workers_started(const size_t& ws) {
	    static std::mutex mtx;
	    std::unique_lock<std::mutex> lock(mtx);
		workers_started_ = ws;
	}

	CV_EXPORTS static size_t workers_started() {
	    static std::mutex mtx;
	    std::unique_lock<std::mutex> lock(mtx);
		return workers_started_;
	}

	CV_EXPORTS static size_t next_worker_ready() {
	    static std::mutex mtx;
	    std::unique_lock<std::mutex> lock(mtx);
		return ++workers_ready_;
	}

	CV_EXPORTS static size_t next_worker_idx() {
	    static std::mutex mtx;
	    std::unique_lock<std::mutex> lock(mtx);
		return next_worker_idx_++;
	}

	template<typename T>
	static bool isShared(const T& shared) {
		std::lock_guard<std::mutex> guard(sharedMtx_);
		std::cerr << "shared:" << reinterpret_cast<size_t>(&shared) << std::endl;
		return shared_.find(reinterpret_cast<size_t>(&shared)) != shared_.end();
	}

	template<typename T>
	static void registerShared(const T& shared) {
		std::lock_guard<std::mutex> guard(sharedMtx_);
		std::cerr << "register:" << reinterpret_cast<size_t>(&shared) << std::endl;
		shared_.insert(std::make_pair(reinterpret_cast<size_t>(&shared), new std::mutex()));
	}

	template<typename T>
	static void lock(const T& shared) {
		Iterator it, end;
		std::mutex* mtx = nullptr;
		{
			std::lock_guard<std::mutex> guard(sharedMtx_);
			it = shared_.find(reinterpret_cast<size_t>(&shared));
			end = shared_.end();
			if(it != end) {
				mtx = (*it).second;
			}
		}

		if(mtx != nullptr) {
			mtx->lock();
			return;
		}
		CV_Assert(!"You are trying to lock a non-shared variable");
	}

	template<typename T>
	static void unlock(const T& shared) {
		Iterator it, end;
		std::mutex* mtx = nullptr;
		{
			std::lock_guard<std::mutex> guard(sharedMtx_);
			it = shared_.find(reinterpret_cast<size_t>(&shared));
			end = shared_.end();
			if(it != end) {
				mtx = (*it).second;
			}
		}

		if(mtx != nullptr) {
			mtx->unlock();
			return;
		}

		CV_Assert(!"You are trying to unlock a non-shared variable");
	}

	template<typename T>
	static T safe_copy(const T& shared) {
		std::lock_guard<std::mutex> guard(sharedMtx_);
		auto it = shared_.find(reinterpret_cast<size_t>(&shared));

		if(it != shared_.end()) {
			std::lock_guard<std::mutex> guard(*(*it).second);
			return shared;
		} else {
			CV_Assert(!"You are unnecessarily safe copying a variable");
			//unreachable
			return shared;
		}
	}

	static cv::UMat safe_copy(const cv::UMat& shared) {
		std::lock_guard<std::mutex> guard(sharedMtx_);
		cv::UMat copy;
		auto it = shared_.find(reinterpret_cast<size_t>(&shared));
		if(it != shared_.end()) {
			std::lock_guard<std::mutex> guard(*(*it).second);
			//workaround for context conflicts
			shared.getMat(cv::ACCESS_READ).copyTo(copy);
			return copy;
		} else {
			CV_Assert(!"You are unnecessarily safe copying a variable");
			//unreachable
			shared.getMat(cv::ACCESS_READ).copyTo(copy);
			return copy;
		}
	}
};

//https://stackoverflow.com/a/27885283/1884837
template<class T>
struct function_traits : function_traits<decltype(&T::operator())> {
};

// partial specialization for function type
template<class R, class... Args>
struct function_traits<R(Args...)> {
    using result_type = R;
    using argument_types = std::tuple<std::remove_reference_t<Args>...>;
};

// partial specialization for function pointer
template<class R, class... Args>
struct function_traits<R (*)(Args...)> {
    using result_type = R;
    using argument_types = std::tuple<std::remove_reference_t<Args>...>;
};

// partial specialization for std::function
template<class R, class... Args>
struct function_traits<std::function<R(Args...)>> {
    using result_type = R;
    using argument_types = std::tuple<std::remove_reference_t<Args>...>;
};

// partial specialization for pointer-to-member-function (i.e., operator()'s)
template<class T, class R, class... Args>
struct function_traits<R (T::*)(Args...)> {
    using result_type = R;
    using argument_types = std::tuple<std::remove_reference_t<Args>...>;
};

template<class T, class R, class... Args>
struct function_traits<R (T::*)(Args...) const> {
    using result_type = R;
    using argument_types = std::tuple<std::remove_reference_t<Args>...>;
};


//https://stackoverflow.com/questions/281818/unmangling-the-result-of-stdtype-infoname
CV_EXPORTS std::string demangle(const char* name);

template <const size_t _UniqueId, typename _Res, typename... _ArgTypes>
struct fun_ptr_helper
{
public:
    typedef std::function<_Res(_ArgTypes...)> function_type;

    static void bind(function_type&& f)
    { instance().fn_.swap(f); }

    static void bind(const function_type& f)
    { instance().fn_=f; }

    static _Res invoke(_ArgTypes... args)
    { return instance().fn_(args...); }

    typedef decltype(&fun_ptr_helper::invoke) pointer_type;
    static pointer_type ptr()
    { return &invoke; }

private:
    static fun_ptr_helper& instance()
    {
        static fun_ptr_helper inst_;
        return inst_;
    }

    fun_ptr_helper() {}

    function_type fn_;
};

template <const size_t _UniqueId, typename _Res, typename... _ArgTypes>
typename fun_ptr_helper<_UniqueId, _Res, _ArgTypes...>::pointer_type
get_fn_ptr(const std::function<_Res(_ArgTypes...)>& f)
{
    fun_ptr_helper<_UniqueId, _Res, _ArgTypes...>::bind(f);
    return fun_ptr_helper<_UniqueId, _Res, _ArgTypes...>::ptr();
}

template<typename T>
std::function<typename std::enable_if<std::is_function<T>::value, T>::type>
make_function(T *t)
{
    return {t};
}

//https://stackoverflow.com/a/33047781/1884837
struct Lambda {
    template<typename Tret, typename T>
    static Tret lambda_ptr_exec() {
        return (Tret) (*(T*)fn<T>());
    }

    template<typename Tret = void, typename Tfp = Tret(*)(), typename T>
    static Tfp ptr(T& t) {
        fn<T>(&t);
        return (Tfp) lambda_ptr_exec<Tret, T>;
    }

    template<typename T>
    static const void* fn(const void* new_fn = nullptr) {
        CV_Assert(new_fn);
    	return new_fn;
    }
};

CV_EXPORTS size_t cnz(const cv::UMat& m);
}
using std::string;
class V4D;



CV_EXPORTS void copy_shared(const cv::UMat& src, cv::UMat& dst);

/*!
 * Convenience function to color convert from Scalar to Scalar
 * @param src The scalar to color convert
 * @param code The color converions code
 * @return The color converted scalar
 */
CV_EXPORTS cv::Scalar colorConvert(const cv::Scalar& src, cv::ColorConversionCodes code);

/*!
 * Convenience function to check for OpenGL errors. Should only be used via the macro #GL_CHECK.
 * @param file The file path of the error.
 * @param line The file line of the error.
 * @param expression The expression that failed.
 */
CV_EXPORTS void gl_check_error(const std::filesystem::path& file, unsigned int line, const char* expression);
/*!
 * Convenience macro to check for OpenGL errors.
 */
#ifndef NDEBUG
#define GL_CHECK(expr)                            \
    expr;                                        \
    cv::v4d::gl_check_error(__FILE__, __LINE__, #expr);
#else
#define GL_CHECK(expr)                            \
    expr;
#endif
CV_EXPORTS void initShader(unsigned int handles[3], const char* vShader, const char* fShader, const char* outputAttributeName);

/*!
 * Returns the OpenGL vendor string
 * @return a string object with the OpenGL vendor information
 */
CV_EXPORTS std::string getGlVendor();
/*!
 * Returns the OpenGL Version information.
 * @return a string object with the OpenGL version information
 */
CV_EXPORTS std::string getGlInfo();
/*!
 * Returns the OpenCL Version information.
 * @return a string object with the OpenCL version information
 */
CV_EXPORTS std::string getClInfo();
/*!
 * Determines if Intel VAAPI is supported
 * @return true if it is supported
 */
CV_EXPORTS bool isIntelVaSupported();
/*!
 * Determines if cl_khr_gl_sharing is supported
 * @return true if it is supported
 */
CV_EXPORTS bool isClGlSharingSupported();
/*!
 * Tells the application if it's alright to keep on running.
 * Note: If you use this mechanism signal handlers are installed
 * @return true if the program should keep on running
 */
CV_EXPORTS bool keepRunning();

CV_EXPORTS void requestFinish();

/*!
 * Creates an Intel VAAPI enabled VideoWriter sink object to use in conjunction with #V4D::setSink().
 * Usually you would call #makeWriterSink() and let it automatically decide if VAAPI is available.
 * @param outputFilename The filename to write the video to.
 * @param fourcc    The fourcc code of the codec to use.
 * @param fps       The fps of the target video.
 * @param frameSize The frame size of the target video.
 * @param vaDeviceIndex The VAAPI device index to use.
 * @return A VAAPI enabled sink object.
 */
CV_EXPORTS cv::Ptr<Sink> makeVaSink(cv::Ptr<V4D> window, const string& outputFilename, const int fourcc, const float fps,
        const cv::Size& frameSize, const int vaDeviceIndex);
/*!
 * Creates an Intel VAAPI enabled VideoCapture source object to use in conjunction with #V4D::setSource().
 * Usually you would call #makeCaptureSource() and let it automatically decide if VAAPI is available.
 * @param inputFilename The file to read from.
 * @param vaDeviceIndex The VAAPI device index to use.
 * @return A VAAPI enabled source object.
 */
CV_EXPORTS cv::Ptr<Source> makeVaSource(cv::Ptr<V4D> window, const string& inputFilename, const int vaDeviceIndex);
/*!
 * Creates a VideoWriter sink object to use in conjunction with #V4D::setSink().
 * This function automatically determines if Intel VAAPI is available and enables it if so.
 * @param outputFilename The filename to write the video to.
 * @param fourcc    The fourcc code of the codec to use.
 * @param fps       The fps of the target video.
 * @param frameSize The frame size of the target video.
  * @return A (optionally VAAPI enabled) VideoWriter sink object.
 */
CV_EXPORTS cv::Ptr<Sink> makeWriterSink(cv::Ptr<V4D> window, const string& outputFilename, const float fps,
        const cv::Size& frameSize);
CV_EXPORTS cv::Ptr<Sink> makeWriterSink(cv::Ptr<V4D> window, const string& outputFilename, const float fps,
        const cv::Size& frameSize, const int fourcc);
/*!
 * Creates a VideoCapture source object to use in conjunction with #V4D::setSource().
 * This function automatically determines if Intel VAAPI is available and enables it if so.
 * @param inputFilename The file to read from.
 * @return A (optionally VAAPI enabled) VideoCapture enabled source object.
 */
CV_EXPORTS cv::Ptr<Source> makeCaptureSource(cv::Ptr<V4D> window, const string& inputFilename);

void resizePreserveAspectRatio(const cv::UMat& src, cv::UMat& output, const cv::Size& dstSize, const cv::Scalar& bgcolor = {0,0,0,255});

}
}

#endif /* SRC_OPENCV_V4D_UTIL_HPP_ */
