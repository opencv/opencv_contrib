// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright Amir Hassan (kallaballa) <amir@viel-zu.org>

#ifndef SRC_OPENCV_V4D_V4D_HPP_
#define SRC_OPENCV_V4D_V4D_HPP_

#ifdef __EMSCRIPTEN__
#  include <emscripten.h>
#  include <emscripten/threading.h>
#endif

#include "source.hpp"
#include "sink.hpp"
#include "util.hpp"
#include "nvg.hpp"
#include "detail/threadpool.hpp"
#include "opencv2/v4d/detail/gl.hpp"
#include "opencv2/v4d/detail/framebuffercontext.hpp"
#include "opencv2/v4d/detail/nanovgcontext.hpp"
#include "opencv2/v4d/detail/imguicontext.hpp"
#include "opencv2/v4d/detail/timetracker.hpp"
#include "opencv2/v4d/detail/glcontext.hpp"


#include <iostream>
#include <future>
#include <set>
#include <map>
#include <string>
#include <memory>
#include <vector>
#include <type_traits>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

using std::cout;
using std::cerr;
using std::endl;
using std::string;

struct GLFWwindow;


/*!
 * OpenCV namespace
 */
namespace cv {
/*!
 * V4D namespace
 */
namespace v4d {
class FormHelper;
enum AllocateFlags {
    NONE = 0,
    NANOVG = 1,
    IMGUI = 2,
    ALL = NANOVG | IMGUI
};
/*!
 * Private namespace
 */
namespace detail {

//https://stackoverflow.com/questions/19961873/test-if-a-lambda-is-stateless#:~:text=As%20per%20the%20Standard%2C%20if,lambda%20is%20stateless%20or%20not.
template <typename T, typename U>
struct helper : helper<T, decltype(&U::operator())>
{};

template <typename T, typename C, typename R, typename... A>
struct helper<T, R(C::*)(A...) const>
{
    static const bool value = std::is_convertible<T, R(*)(A...)>::value;
};

template<typename T>
struct is_stateless
{
    static const bool value = helper<T,T>::value;
};

class FrameBufferContext;
class CLVAContext;
class NanoVGContext;
class GLContext;
class ImGuiContextImpl;
template<typename T> std::string int_to_hex( T i )
{
  std::stringstream stream;
  stream << "0x"
         << std::setfill ('0') << std::setw(sizeof(T) * 2)
         << std::hex << i;
  return stream.str();
}

template<typename Tfn> std::string func_id(Tfn& fn) {
    return int_to_hex((size_t) &fn);
}
}

using namespace cv::v4d::detail;

class CV_EXPORTS V4D {
    friend class detail::FrameBufferContext;
    friend class HTML5Capture;
    static const std::thread::id default_thread_id_;
    static std::thread::id main_thread_id_;
    static concurrent::threadpool thread_pool_;
    std::map<std::string, cv::Ptr<cv::UMat>> umat_pool_;
    std::map<std::string, std::shared_ptr<void>> data_pool_;
    cv::Ptr<V4D> self_;
    cv::Size initialSize_;
    bool debug_;
    cv::Rect viewport_;
    bool stretching_;
    bool focused_ = false;
    FrameBufferContext* mainFbContext_ = nullptr;
    CLVAContext* clvaContext_ = nullptr;
    NanoVGContext* nvgContext_ = nullptr;
    ImGuiContextImpl* imguiContext_ = nullptr;
    std::mutex glCtxMtx_;
    std::map<int32_t,GLContext*> glContexts_;
    bool closed_ = false;
    cv::Ptr<Source> source_;
    cv::Ptr<Sink> sink_;
    cv::UMat currentReaderFrame_;
    cv::UMat nextReaderFrame_;
    cv::UMat currentWriterFrame_;
    std::future<bool> futureReader_;
    std::future<void> futureWriter_;
    std::function<bool(int key, int scancode, int action, int modifiers)> keyEventCb_;
    std::function<void(int button, int action, int modifiers)> mouseEventCb_;
    cv::Point2f mousePos_;
    uint64_t frameCnt_ = 0;
    bool showFPS_ = true;
    bool printFPS_ = true;
    bool showTracking_ = true;
    uint64_t currentSeqNr_ = 0;
    size_t numWorkers_ = 0;
public:
    /*!
     * Creates a V4D object which is the central object to perform visualizations with.
     * @param initialSize The initial size of the heavy-weight window.
     * @param frameBufferSize The initial size of the framebuffer backing the window (needs to be equal or greate then initial size).
     * @param offscreen Don't create a window and rather render offscreen.
     * @param title The window title.
     * @param major The OpenGL major version to request.
     * @param minor The OpenGL minor version to request.
     * @param compat Request a compatibility context.
     * @param samples MSAA samples.
     * @param debug Create a debug OpenGL context.
     */
    CV_EXPORTS static cv::Ptr<V4D> make(int w, int h, const string& title, AllocateFlags flags = ALL, bool offscreen = false, bool debug = false, int samples = 0);
    CV_EXPORTS static cv::Ptr<V4D> make(const cv::Size& size, const cv::Size& fbsize, const string& title, AllocateFlags flags = ALL, bool offscreen = false, bool debug = false, int samples = 0);
    /*!
     * Default destructor
     */
    CV_EXPORTS virtual ~V4D();

    template<typename T>
    T once(std::function<T()> fn) {
        static thread_local std::once_flag onceFlag;
        std::call_once(onceFlag, fn);
    }


    void once(std::function<void()> fn) {
        static thread_local std::once_flag onceFlag;
        std::call_once(onceFlag, fn);
    }
    template<typename T>
    T& get(const string& name) {
        auto it = data_pool_.find(name);
        std::shared_ptr<void> p = std::make_shared<T>();
        if(it == data_pool_.end()) {
            data_pool_.insert({name, p });
        }else
            p = (*it).second;

        return *(std::static_pointer_cast<T, void>(p).get());
    }

    template<typename T>
    T& init(const string& name, std::function<std::shared_ptr<T>()> creatorFunc) {
        auto it = data_pool_.find(name);
        std::shared_ptr<void> p;
        if(it == data_pool_.end())
            data_pool_.insert({name, p = std::static_pointer_cast<void, T>(creatorFunc())});
        else
            p = (*it).second;

        return *static_cast<T*>(p.get());
    }

    CV_EXPORTS cv::Ptr<cv::UMat> get(const string& name);
    CV_EXPORTS cv::Ptr<cv::UMat> get(const string& name, cv::Size sz, int type);

    CV_EXPORTS size_t workers();
    CV_EXPORTS bool isMain() const;
    /*!
     * The internal framebuffer exposed as OpenGL Texture2D.
     * @return The texture object.
     */
    CV_EXPORTS cv::ogl::Texture2D& texture();
    CV_EXPORTS std::string title();
    template <typename Tfn, typename ... Args>
    void gl(Tfn fn, Args&& ... args) {
        TimeTracker::getInstance()->execute("gl(" + detail::func_id(fn) + ")/" + std::to_string(-1), [this, fn, &args...](){
            glCtx(-1).render([=]() {
                fn(args...);
            });
        });
    }

    template <typename Tfn, typename ... Args>
    void gl(const size_t& idx, Tfn fn, Args&& ... args) {
        TimeTracker::getInstance()->execute("gl(" + detail::func_id(fn) + ")/" + std::to_string(-1), [this, fn, idx, &args...](){
            glCtx(idx).render([=]() {
                fn(args...);
            });
        });
    }

    /*!
     * Execute function object fn inside a framebuffer context.
     * The context acquires the framebuffer from OpenGL (either by up-/download or by cl-gl sharing)
     * and provides it to the functon object. This is a good place to use OpenCL
     * directly on the framebuffer.
     * @param fn A function object that is passed the framebuffer to be read/manipulated.
     */
    template <typename Tfn, typename ... Args>
    void fb(Tfn fn, Args& ... args) {
        CV_Assert(detail::is_stateless<decltype(fn)>::value);
        TimeTracker::getInstance()->execute("fb(" + detail::func_id(fn) + ")", [this, fn, &args...]{
            fbCtx().execute(fn, args...);
        });
    }
    /*!
     * Execute function object fn inside a nanovg context.
     * The context takes care of setting up opengl and nanovg states.
     * A function object passed like that can use the functions in cv::viz::nvg.
     * @param fn A function that is passed the size of the framebuffer
     * and performs drawing using cv::viz::nvg
     */
    template <typename Tfn, typename ... Args>
    void nvg(Tfn fn, Args&& ... args) {
        TimeTracker::getInstance()->execute("nvg(" + detail::func_id(fn) + ")", [this, fn, &args...](){
            nvgCtx().render([this, fn, &args...]() {
                fn(args...);
            });
        });
    }

    CV_EXPORTS void imgui(std::function<void(ImGuiContext* ctx)> fn);

    /*!
     * Copy the framebuffer contents to an OutputArray.
     * @param arr The array to copy to.
     */
    CV_EXPORTS void copyTo(cv::UMat& arr);
    /*!
     * Copy the InputArray contents to the framebuffer.
     * @param arr The array to copy.
     */
    CV_EXPORTS void copyFrom(const cv::UMat& arr);
    /*!
     * Execute function object fn in a loop.
     * This function main purpose is to abstract the run loop for portability reasons.
     * @param fn A functor that will be called repeatetly until the application terminates or the functor returns false
     */
    CV_EXPORTS void run(std::function<bool(cv::Ptr<V4D>)> fn
#ifndef __EMSCRIPTEN__
            , size_t workers = 0
#endif
    );
    /*!
     * Called to feed an image directly to the framebuffer
     */
    CV_EXPORTS void feed(cv::InputArray in);
    /*!
     * Fetches a copy of frambuffer
     * @return a copy of the framebuffer
     */
    CV_EXPORTS cv::_InputArray fetch();

    /*!
     * Called to capture to the framebuffer from a #cv::viz::Source object provided via #V4D::setSource().
     * @return true if successful.
     */
    CV_EXPORTS bool capture();

    /*!
     * Called to capture from a function object.
     * The functor fn is passed a UMat which it writes to which in turn is captured to the framebuffer.
     * @param fn The functor that provides the data.
     * @return true if successful-
     */
    CV_EXPORTS bool capture(std::function<void(cv::UMat&)> fn);

    /*!
     * Called to write the framebuffer to a #cv::viz::Sink object provided via #V4D::setSink()
     */
    CV_EXPORTS void write();
    /*!
     * Called to pass the frambuffer to a functor which consumes it (e.g. writes to a video file).
     * @param fn The functor that consumes the data,
     */
    CV_EXPORTS void write(std::function<void(const cv::UMat&)> fn);
    /*!
     * Set the current #cv::viz::Source object. Usually created using #makeCaptureSource().
     * @param src A #cv::viz::Source object.
     */
    CV_EXPORTS void setSource(Source& src);
    CV_EXPORTS Source& getSource();
    CV_EXPORTS bool hasSource();

    /*!
     * Checks if the current #cv::viz::Source is ready.
     * @return true if it is ready.
     */
    CV_EXPORTS bool isSourceReady();
    /*!
     * Set the current #cv::viz::Sink object. Usually created using #makeWriterSink().
     * @param sink A #cv::viz::Sink object.
     */
    CV_EXPORTS void setSink(Sink& sink);
    CV_EXPORTS Sink& getSink();
    CV_EXPORTS bool hasSink();
    /*!
     * Checks if the current #cv::viz::Sink is ready.
     * @return true if it is ready.
     */
    CV_EXPORTS bool isSinkReady();
    /*!
     * Get the window position.
     * @return The window position.
     */
    CV_EXPORTS cv::Vec2f position();
    /*!
     * Get the current viewport reference.
     * @return The current viewport reference.
     */
    CV_EXPORTS cv::Rect& viewport();
    /*!
     * Get the pixel ratio of the display x-axis.
     * @return The pixel ratio of the display x-axis.
     */
    CV_EXPORTS float pixelRatioX();
    /*!
     * Get the pixel ratio of the display y-axis.
     * @return The pixel ratio of the display y-axis.
     */
    CV_EXPORTS float pixelRatioY();
    /*!
     * Set the window size.
     * @param sz The new window size.
     */
    CV_EXPORTS cv::Size initialSize();
    /*!
     * Get the current size of the window.
     * @return The window size.
     */
    CV_EXPORTS cv::Size fbSize();
    /*!
     * Set the window size
     * @param sz The future size of the window.
     */
    CV_EXPORTS void setSize(const cv::Size& sz);
    /*!
     * Get the window size.
     * @return The window size.
     */
    CV_EXPORTS cv::Size size();
    /*!
     * Get the frambuffer size.
     * @return The framebuffer size.
     */

    CV_EXPORTS bool getShowFPS();
    CV_EXPORTS void setShowFPS(bool s);
    CV_EXPORTS bool getPrintFPS();
    CV_EXPORTS void setPrintFPS(bool p);
    CV_EXPORTS bool getShowTracking();
    CV_EXPORTS void setShowTracking(bool st);

    CV_EXPORTS bool isFullscreen();
    /*!
     * Enable or disable fullscreen mode.
     * @param f if true enable fullscreen mode else disable.
     */
    CV_EXPORTS void setFullscreen(bool f);
    /*!
     * Determines if the window is resizeable.
     * @return true if the window is resizeable.
     */
    CV_EXPORTS bool isResizable();
    /*!
     * Set the window resizable.
     * @param r if r is true set the window resizable.
     */
    CV_EXPORTS void setResizable(bool r);
    /*!
     * Determine if the window is visible.
     * @return true if the window is visible.
     */
    CV_EXPORTS bool isVisible();
    /*!
     * Set the window visible or invisible.
     * @param v if v is true set the window visible.
     */
    CV_EXPORTS void setVisible(bool v);
    /*!
     * Enable/Disable scaling the framebuffer during blitting.
     * @param s if true enable scaling.
     */
    CV_EXPORTS void setStretching(bool s);
    /*!
     * Determine if framebuffer is scaled during blitting.
     * @return true if framebuffer is scaled during blitting.
     */
    CV_EXPORTS bool isStretching();
    /*!
     * Determine if th V4D object is marked as focused.
     * @return true if the V4D object is marked as focused.
     */
    CV_EXPORTS bool isFocused();
    /*!
     * Mark the V4D object as focused.
     * @param s if true mark as focused.
     */
    CV_EXPORTS void setFocused(bool f);
    /*!
     * Everytime a frame is displayed this count is incremented-
     * @return the current frame count-
     */
    CV_EXPORTS uint64_t frameCount();
    /*!
     * Determine if the window is closed.
     * @return true if the window is closed.
     */
    CV_EXPORTS bool isClosed();
    /*!
     * Close the window.
     */
    CV_EXPORTS void close();
    /*!
     * Display the framebuffer in the native window by blitting.
     * @return false if the window is closed.
     */
    CV_EXPORTS bool display();
    /*!
     * Print basic system information to stderr.
     */
    CV_EXPORTS void printSystemInfo();

    CV_EXPORTS void makeCurrent();

    CV_EXPORTS GLFWwindow* getGLFWWindow();

    CV_EXPORTS FrameBufferContext& fbCtx();
    CV_EXPORTS CLVAContext& clvaCtx();
    CV_EXPORTS NanoVGContext& nvgCtx();
    CV_EXPORTS ImGuiContextImpl& imguiCtx();
    CV_EXPORTS GLContext& glCtx(int32_t idx = 0);

    CV_EXPORTS bool hasFbCtx();
    CV_EXPORTS bool hasClvaCtx();
    CV_EXPORTS bool hasNvgCtx();
    CV_EXPORTS bool hasImguiCtx();
    CV_EXPORTS bool hasGlCtx(uint32_t idx = 0);
    CV_EXPORTS size_t numGlCtx();
private:
    V4D(const cv::Size& size, const cv::Size& fbsize,
            const string& title, AllocateFlags flags, bool offscreen, bool debug, int samples);

    cv::Point2f getMousePosition();
    void setMousePosition(const cv::Point2f& pt);

    void swapContextBuffers();
protected:
    cv::Ptr<V4D> self();
    void fence();
    bool wait(uint64_t timeout = 0);
};
}
} /* namespace kb */

#ifdef __EMSCRIPTEN__
#  define thread_local
#endif

#endif /* SRC_OPENCV_V4D_V4D_HPP_ */
