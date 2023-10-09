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
#include "detail/backend.hpp"
#include "detail/threadpool.hpp"
#include "detail/gl.hpp"
#include "detail/framebuffercontext.hpp"
#include "detail/nanovgcontext.hpp"
#include "detail/imguicontext.hpp"
#include "detail/timetracker.hpp"
#include "detail/glcontext.hpp"
#include "detail/sourcecontext.hpp"
#include "detail/sinkcontext.hpp"


#include <iostream>
#include <future>
#include <set>
#include <map>
#include <string>
#include <memory>
#include <vector>
#include <type_traits>
#define UNW_LOCAL_ONLY
#include <libunwind.h>
#include <stdio.h>

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

enum AllocateFlags {
    NONE = 0,
    NANOVG = 1,
    IMGUI = 2,
    ALL = NANOVG | IMGUI
};

class Plan {
public:
	virtual ~Plan() {};
	virtual void infere(cv::Ptr<V4D> window) {};
};
/*!
 * Private namespace
 */
namespace detail {

template <typename T> using static_not = std::integral_constant<bool, !T::value>;

//https://stackoverflow.com/questions/19961873/test-if-a-lambda-is-stateless#:~:text=As%20per%20the%20Standard%2C%20if,lambda%20is%20stateless%20or%20not.
template <typename T, typename U>
struct helper : helper<T, decltype(&U::operator())>
{};

template <typename T, typename C, typename R, typename... A>
struct helper<T, R(C::*)(A...) const>
{
	static const bool value = std::is_convertible<T, std::function<R(A...)>>::value || std::is_convertible<T, R(*)(A...)>::value;
};

template<typename T>
struct is_stateless
{
    static const bool value = helper<T,T>::value;
};

template<typename T> std::string int_to_hex( T i )
{
  std::stringstream stream;
  stream << "0x"
         << std::setfill ('0') << std::setw(sizeof(T) * 2)
         << std::hex << i;
  return stream.str();
}

//template<typename Tfn> std::string func_hex(Tfn& fn) {
//    return int_to_hex((size_t) &fn);
//}

template<typename Tlamba> std::string lambda_ptr_hex(Tlamba&& l) {
    return int_to_hex((size_t)Lambda::ptr(l));
}
}

using namespace cv::v4d::detail;

class CV_EXPORTS V4D {
	friend class detail::FrameBufferContext;
    friend class HTML5Capture;
    static const std::thread::id default_thread_id_;
    static std::thread::id main_thread_id_;
	static bool first_run_;
    std::map<std::string, cv::Ptr<cv::UMat>> umat_pool_;
    std::map<std::string, std::shared_ptr<void>> data_pool_;
    cv::Ptr<V4D> self_;
    cv::Size initialSize_;
    bool debug_;
    cv::Rect viewport_;
    bool stretching_;
    bool focused_ = false;
    cv::Ptr<FrameBufferContext> mainFbContext_ = nullptr;
    cv::Ptr<SourceContext> sourceContext_ = nullptr;
    cv::Ptr<SinkContext> sinkContext_ = nullptr;
    cv::Ptr<NanoVGContext> nvgContext_ = nullptr;
    cv::Ptr<ImGuiContextImpl> imguiContext_ = nullptr;
    cv::Ptr<SingleContext> singleContext_ = nullptr;
    cv::Ptr<ParallelContext> parallelContext_ = nullptr;
    std::mutex glCtxMtx_;
    std::map<int32_t,cv::Ptr<GLContext>> glContexts_;
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
    size_t numWorkers_ = 0;
    std::vector<std::tuple<std::string,bool,long>> accesses_;
    std::map<std::string, cv::Ptr<Transaction>> transactions_;

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

    struct Node {
    	string name_;
    	std::set<long> read_deps_;
    	std::set<long> write_deps_;
    	cv::Ptr<Transaction> tx_  = nullptr;
    	bool initialized() {
    		return tx_;
    	}
    };

    std::vector<cv::Ptr<Node>> nodes_;

    void findNode(const string& name, cv::Ptr<Node>& found) {
    	CV_Assert(!name.empty());
    	if(nodes_.empty())
    		return;

    	if(nodes_.back()->name_ == name)
    		found = nodes_.back();

    }

    void makePlan() {
    	cout << std::this_thread::get_id() << " ### MAKE PLAN ### " << endl;
    	for(const auto& t : accesses_) {
    		const string& name = std::get<0>(t);
    		const bool& read = std::get<1>(t);
    		const long& dep = std::get<2>(t);
    		cv::Ptr<Node> n;
    		findNode(name, n);

    		if(!n) {
    			n = new Node();
    			n->name_ = name;
    			n->tx_ = transactions_[name];
    			CV_Assert(!n->name_.empty());
    			CV_Assert(n->tx_);
    			nodes_.push_back(n);
        		cout << "make: " << std::this_thread::get_id() << " " << n->name_ << endl;
    		}


    		if(read) {
    			n->read_deps_.insert(dep);
    		} else {
    			n->write_deps_.insert(dep);
    		}
    	}
    }

	void runPlan() {
		cout << std::this_thread::get_id() << " ### RUN PLAN ### " << endl;
		bool isEnabled = true;

		for (auto& n : nodes_) {
			if (n->tx_->hasCondition()) {
				isEnabled = n->tx_->enabled();
				cout << "cond: " << std::this_thread::get_id() << " " << n->name_ << ": " << isEnabled << endl;
			}

			if (!(n->tx_->hasCondition()) && isEnabled) {
				n->tx_->getContext()->execute([=]() {
					cout << "run: " << std::this_thread::get_id() << " " << n->name_ << ": " << isEnabled << endl;
					n->tx_->perform();
				});
			}
		}
	}

    template<typename Tenabled, typename T, typename ...Args>
    typename std::enable_if<std::is_same<Tenabled, std::false_type>::value, void>::type
	emit_access(const string& context, bool read, const T* tp) {
    	//disabled
    }

    template<typename Tenabled, typename T, typename ...Args>
    typename std::enable_if<std::is_same<Tenabled, std::true_type>::value, void>::type
	emit_access(const string& context, bool read, const T* tp) {
    	cout << "access: " << std::this_thread::get_id() << " " << context << string(read ? " <- " : " -> ") << demangle(typeid(std::remove_const_t<T>).name()) << "(" << (long)tp << ") " << endl;
    	accesses_.push_back(std::make_tuple(context, read, (long)tp));
    }

    template<typename Tfn, typename ...Args>
    void add_transaction(cv::Ptr<V4DContext> ctx, const string& invocation, Tfn fn, Args&& ...args) {
    	auto it = transactions_.find(invocation);
    	if(it == transactions_.end()) {
    		auto tx = make_transaction(fn, std::forward<Args>(args)...);
    		tx->setContext(ctx);
    		transactions_.insert({invocation, tx});
    	}
    }

//    template<typename Tfn, typename Tfb = cv::UMat&, typename ...Args>
//    void add_transaction(const string& context, Tfn&& fn, cv::UMat&& fb, Args&& ...args) {
//    	auto it = transactions_.find(context);
//    	if(it == transactions_.end()) {
//    		transactions_.insert({context, make_transaction<Tfn&&, Tfb&&>(std::forward<Tfn>(fn), std::forward<Tfb>(fb), std::forward<Args>(args)...)});
//    	}
//    }
//
//    template<typename Tfn, typename Tfb = const cv::UMat&&, typename ...Args>
//    void add_transaction(const string& context, Tfn&& fn, const cv::UMat&& fb, Args&& ...args) {
//    	auto it = transactions_.find(context);
//    	if(it == transactions_.end()) {
//    		transactions_.insert({context, make_transaction<Tfn&&, Tfb&&>(std::forward<Tfn>(fn), std::forward<Tfb>(fb), std::forward<Args>(args)...)});
//    	}
//    }

    std::size_t index(const std::thread::id id)
    {
        static std::size_t nextindex = 0;
        static std::mutex my_mutex;
        static std::unordered_map<std::thread::id, std::size_t> ids;
        std::lock_guard<std::mutex> lock(my_mutex);
        auto iter = ids.find(id);
        if(iter == ids.end())
            return ids[id] = nextindex++;
        return iter->second;
    }


    template<typename Tfn, typename Textra>
    const string make_id(const string& name, Tfn&& fn, const Textra& extra) {
    	stringstream ss;
    	stringstream ssExtra;
    	ssExtra << extra;
    	if(ssExtra.str().empty()) {
    		ss << name << "(" << index(std::this_thread::get_id()) << "-" << detail::lambda_ptr_hex(std::forward<Tfn>(fn)) << ")";
    	}
    	else {
    		ss << name << "(" << index(std::this_thread::get_id()) << "-" << detail::lambda_ptr_hex(std::forward<Tfn>(fn)) << ")/" << ssExtra.str();
    	}

    	return ss.str();
    }

    template<typename Tfn>
    const string make_id(const string& name, Tfn&& fn, const string& extra = "") {
    	return make_id<Tfn, std::string>(name, fn, extra);
    }


    template<typename Tfn, typename Textra>
    void print_id(const string& name, Tfn&& fn, const Textra& extra) {
   		cerr << make_id<Tfn, Textra>(name, fn, extra) << endl;
    }

    template<typename Tfn>
    void print_id(const string& name, Tfn&& fn, const string& extra = "") {
   		cerr << make_id<Tfn, std::string>(name, fn, extra) << endl;
    }

    template <typename Tfn, typename ... Args>
    void gl(Tfn fn, Args&& ... args) {
        CV_Assert(detail::is_stateless<std::remove_cv_t<std::remove_reference_t<decltype(fn)>>>::value);
        const string id = make_id("gl", fn, -1);
        TimeTracker::getInstance()->execute(id, [this, fn, id, &args...](){
            emit_access<std::true_type, cv::UMat, Args...>(id, true, &fbCtx()->fb());
            (emit_access<std::true_type, std::remove_reference_t<Args>, Args...>(id, std::is_const_v<std::remove_reference_t<Args>>, &args),...);
            emit_access<std::true_type, cv::UMat, Args...>(id, false, &fbCtx()->fb());
        	std::function<void((Args...))> functor(fn);
            add_transaction(glCtx(-1), id, std::forward<decltype(functor)>(fn), std::forward<Args>(args)...);
        });
    }

    template <typename Tfn, typename ... Args>
    void gl(const size_t& idx, Tfn fn, Args&& ... args) {
        CV_Assert(detail::is_stateless<std::remove_cv_t<std::remove_reference_t<decltype(fn)>>>::value);
        const string id = make_id("gl", fn, idx);
        TimeTracker::getInstance()->execute(id, [this, fn, idx, id, &args...](){
            emit_access<std::true_type, cv::UMat, Args...>(id, true, &fbCtx()->fb());
            (emit_access<std::true_type, std::remove_reference_t<Args>, Args...>(id, std::is_const_v<std::remove_reference_t<Args>>, &args),...);
            emit_access<std::true_type, cv::UMat, Args...>(id, false, &fbCtx()->fb());
        	std::function<void((Args...))> functor(fn);
            add_transaction(glCtx(idx), id, std::forward<decltype(functor)>(fn), std::forward<Args>(args)...);
        });
    }

    template <typename Tfn>
    void graph(Tfn fn) {
        CV_Assert(detail::is_stateless<std::remove_cv_t<std::remove_reference_t<decltype(fn)>>>::value);
        const string id = make_id("graph", fn);

        TimeTracker::getInstance()->execute(id, [this, fn, id](){
            std::function functor = fn;
            emit_access<std::true_type, decltype(fn)>(id, true, &fn);
            add_transaction(singleCtx(), id, functor);
        });
    }

    template <typename Tfn, typename ... Args>
    void graph(Tfn fn, Args&& ... args) {
        CV_Assert(detail::is_stateless<std::remove_cv_t<std::remove_reference_t<decltype(fn)>>>::value);
        const string id = make_id("graph", fn);

        TimeTracker::getInstance()->execute(id, [this, fn, id, &args...](){
            (emit_access<std::true_type, std::remove_reference_t<Args>, Args...>(id, std::is_const_v<std::remove_reference_t<Args>>, &args),...);
            std::function functor = fn;
            add_transaction(singleCtx(), id, functor, std::forward<Args>(args)...);
        });
    }

    template <typename Tfn>
    void endgraph(Tfn fn) {
        CV_Assert(detail::is_stateless<std::remove_cv_t<std::remove_reference_t<decltype(fn)>>>::value);
        const string id = make_id("endgraph", fn);

        TimeTracker::getInstance()->execute(id, [this, fn, id] {
            std::function functor = fn;
            emit_access<std::true_type, decltype(fn)>(id, true, &fn);
            add_transaction(singleCtx(), id, functor);
        });
    }

    template <typename Tfn, typename ... Args>
    void endgraph(Tfn fn, Args&& ... args) {
        CV_Assert(detail::is_stateless<std::remove_cv_t<std::remove_reference_t<decltype(fn)>>>::value);
        const string id = make_id("endgraph", fn);

        TimeTracker::getInstance()->execute(id, [this, fn, id, &args...](){
            (emit_access<std::true_type, std::remove_reference_t<Args>, Args...>(id, std::is_const_v<std::remove_reference_t<Args>>, &args),...);
            std::function functor = fn;
            add_transaction(singleCtx(), id, functor, std::forward<Args>(args)...);
        });
    }

    template <typename Tfn, typename ... Args>
    void fb(Tfn fn, Args&& ... args) {
        CV_Assert(detail::is_stateless<std::remove_cv_t<std::remove_reference_t<decltype(fn)>>>::value);
        const string id = make_id("fb", fn);
        TimeTracker::getInstance()->execute(id, [this, fn, id, &args...]{
            using Tfb = typename std::tuple_element<0, typename function_traits<Tfn>::argument_types>::type;
            using Tfbbase = typename std::remove_cv<Tfb>::type;
            using Tfbconst = std::add_const_t<Tfbbase>;

            static_assert((std::is_same<Tfb, cv::UMat&>::value || std::is_same<Tfb, const cv::UMat&>::value) || !"The first argument must be eiter of type 'cv::UMat&' or 'const cv::UMat&'");
            emit_access<std::true_type, cv::UMat, Tfb, Args...>(id, true, &fbCtx()->fb());
            (emit_access<std::true_type, std::remove_reference_t<Args>, Tfb, Args...>(id, std::is_const_v<std::remove_reference_t<Args>>, &args),...);
            emit_access<static_not<typename std::is_const<Tfbbase>::type>, cv::UMat, Tfb, Args...>(id, false, &fbCtx()->fb());
        	std::function<void((Tfb,Args...))> functor(fn);
            add_transaction<decltype(functor),Tfb>(fbCtx(),id, std::forward<decltype(functor)>(functor), fbCtx()->fb(), std::forward<Args>(args)...);
        });
    }

    template <typename Tfn, typename ... Args>
    void capture(Tfn fn, Args&& ... args) {
        CV_Assert(detail::is_stateless<std::remove_cv_t<std::remove_reference_t<decltype(fn)>>>::value);
        const string id = make_id("capture", fn);
        TimeTracker::getInstance()->execute(id, [this, fn, id, &args...]{
            using Tfb = typename std::tuple_element<0, typename function_traits<Tfn>::argument_types>::type;

            static_assert((std::is_same<Tfb,const cv::UMat&>::value) || !"The first argument must be of type 'const cv::UMat&'");
            emit_access<std::true_type, cv::UMat, Tfb, Args...>(id, true, &sourceCtx()->sourceBuffer());
            (emit_access<std::true_type, std::remove_reference_t<Args>, Tfb, Args...>(id, std::is_const_v<std::remove_reference_t<Args>>, &args),...);
        	std::function<void((Tfb,Args...))> functor(fn);
            add_transaction<decltype(functor),Tfb>(std::dynamic_pointer_cast<V4DContext>(sourceCtx()),id, std::forward<decltype(functor)>(functor), sourceCtx()->sourceBuffer(), std::forward<Args>(args)...);
        });
    }

    template <typename Tfn, typename ... Args>
    void write(Tfn fn, Args&& ... args) {
        CV_Assert(detail::is_stateless<std::remove_cv_t<std::remove_reference_t<decltype(fn)>>>::value);
        const string id = make_id("write", fn);
        TimeTracker::getInstance()->execute(id, [this, fn, id, &args...]{
            using Tfb = typename std::tuple_element<0, typename function_traits<Tfn>::argument_types>::type;

            static_assert((std::is_same<Tfb,cv::UMat&>::value) || !"The first argument must be of type 'cv::UMat&'");
            emit_access<std::true_type, cv::UMat, Tfb, Args...>(id, true, &sinkCtx()->sinkBuffer());
            (emit_access<std::true_type, std::remove_reference_t<Args>, Tfb, Args...>(id, std::is_const_v<std::remove_reference_t<Args>>, &args),...);
            emit_access<std::true_type, cv::UMat, Tfb, Args...>(id, false, &sinkCtx()->sinkBuffer());
        	std::function<void((Tfb,Args...))> functor(fn);
            add_transaction<decltype(functor),Tfb>(std::dynamic_pointer_cast<V4DContext>(sinkCtx()),id, std::forward<decltype(functor)>(functor), sinkCtx()->sinkBuffer(), std::forward<Args>(args)...);
        });
    }

    /*!
     * Execute function object fn inside a nanovg context.
     * The context takes care of setting up opengl and nanovg states.
     * A function object passed like that can use the functions in cv::viz::nvg.
     * @param fn A function that is passed the size of the framebuffer
     * and performs drawing using cv::v4d::nvg
     */
    template <typename Tfn, typename ... Args>
    void nvg(Tfn fn, Args&&... args) {
        CV_Assert(detail::is_stateless<std::remove_cv_t<std::remove_reference_t<decltype(fn)>>>::value);
        const string id = make_id("nvg", fn);
        TimeTracker::getInstance()->execute(id, [this, fn, id, &args...](){
            emit_access<std::true_type, cv::UMat, Args...>(id, true, &fbCtx()->fb());
            (emit_access<std::true_type, std::remove_reference_t<Args>, Args...>(id, std::is_const_v<std::remove_reference_t<Args>>, &args),...);
            emit_access<std::true_type, cv::UMat, Args...>(id, false, &fbCtx()->fb());
        	std::function<void((Args...))> functor(fn);
            add_transaction<decltype(functor)>(nvgCtx(), id, std::forward<decltype(functor)>(fn), std::forward<Args>(args)...);
        });
    }

    template <typename Tfn, typename ... Args>
    void single(Tfn fn, Args&&... args) {
        CV_Assert(detail::is_stateless<std::remove_cv_t<std::remove_reference_t<decltype(fn)>>>::value);
        const string id = make_id("single", fn);
        TimeTracker::getInstance()->execute(id, [this, fn, id, &args...](){
            (emit_access<std::true_type, std::remove_reference_t<Args>, Args...>(id, std::is_const_v<std::remove_reference_t<Args>>, &args),...);
        	std::function<void((Args...))> functor(fn);
            add_transaction<decltype(functor)>(singleCtx(), id, std::forward<decltype(functor)>(fn), std::forward<Args>(args)...);
        });
    }

    template <typename Tfn, typename ... Args>
     void parallel(Tfn fn, Args&&... args) {
         CV_Assert(detail::is_stateless<std::remove_cv_t<std::remove_reference_t<decltype(fn)>>>::value);
         const string id = make_id("parallel", fn);
         TimeTracker::getInstance()->execute(id, [this, fn, id, &args...](){
             (emit_access<std::true_type, std::remove_reference_t<Args>, Args...>(id, std::is_const_v<std::remove_reference_t<Args>>, &args),...);
             std::function<void((Args...))> functor(fn);
             add_transaction<decltype(functor)>(parallelCtx(), id, std::forward<decltype(functor)>(fn), std::forward<Args>(args)...);
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


	#ifdef __EMSCRIPTEN__
	bool first = true;
	static void do_frame(void* void_fn_ptr) {
		 if(first) {
			 glfwSwapInterval(0);
			 first = false;
		 }
		 auto* fn_ptr = reinterpret_cast<std::function<bool()>*>(void_fn_ptr);
		 if (fn_ptr) {
			 auto& fn = *fn_ptr;
				 fn();
		 }
	 }
	#endif

	template<typename Tplan>
	void run(size_t workers) {
		cv::Ptr<Tplan> plan = new Tplan();
		std::vector<std::thread*> threads;
		{
			static std::mutex runMtx;
			std::unique_lock<std::mutex> lock(runMtx);

			numWorkers_ = workers;
			if (this->hasSource()) {
				this->getSource()->setThreadSafe(true);
			}

			if(first_run_) {
				main_thread_id_ = std::this_thread::get_id();
				first_run_ = false;
				cerr << "Starting with " << workers << " extra workers" << endl;
			}

			if(workers > 0  || !this->isMain()) {
				cv::setNumThreads(0);
				cerr << "Setting threads to 0" << endl;
			}

			if(this->isMain()) {
				int w = this->initialSize().width;
				int h = this->initialSize().height;
				const string title = this->title();
				bool debug = this->debug_;
				auto src = this->getSource();

				for (size_t i = 0; i < workers; ++i) {
					threads.push_back(
							new std::thread(
									[w,h,i,title,debug,src] {
										cv::Ptr<cv::v4d::V4D> worker = V4D::make(
												w,
												h,
												title + "-worker-" + std::to_string(i),
												NANOVG,
												!debug,
												debug,
												0);
										if (src) {
											src->setThreadSafe(true);
											worker->setSource(src);
										}
//										if (this->hasSink()) {
	//                                        Sink sink = this->getSink();
	//                                        sink.setThreadSafe(true);
	//                                        worker->setSink(sink);
//										}
										worker->run<Tplan>(0);
								}
							)
					);
				}
			}
		}

	//	if(this->isMain())
	//		this->makeCurrent();

	#ifndef __EMSCRIPTEN__
			bool success = true;
			CLExecScope_t scope(this->fbCtx()->getCLExecContext());
			plan->infere(self());
			this->makePlan();
			do {
				this->runPlan();
			} while(this->display());
	#else
		if(this->isMain()) {
			std::function<bool()> fnFrame([=,this](){
				return fn(self());
			});

			emscripten_set_main_loop_arg(do_frame, &fnFrame, -1, true);
		} else {
			while (true) {
				fn(self());
			}
		}
	#endif

		if(this->isMain()) {
			for(auto& t : threads)
				t->join();
		}
	}
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
     * Set the current #cv::viz::Source object. Usually created using #makeCaptureSource().
     * @param src A #cv::viz::Source object.
     */
    CV_EXPORTS void setSource(cv::Ptr<Source> src);
    CV_EXPORTS cv::Ptr<Source> getSource();
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

    CV_EXPORTS cv::Ptr<FrameBufferContext> fbCtx();
    CV_EXPORTS cv::Ptr<SourceContext> sourceCtx();
    CV_EXPORTS cv::Ptr<SinkContext> sinkCtx();
    CV_EXPORTS cv::Ptr<NanoVGContext> nvgCtx();
    CV_EXPORTS cv::Ptr<SingleContext> singleCtx();
    CV_EXPORTS cv::Ptr<ParallelContext> parallelCtx();
    CV_EXPORTS cv::Ptr<ImGuiContextImpl> imguiCtx();
    CV_EXPORTS cv::Ptr<GLContext> glCtx(int32_t idx = 0);

    CV_EXPORTS bool hasFbCtx();
    CV_EXPORTS bool hasSourceCtx();
    CV_EXPORTS bool hasSinkCtx();
    CV_EXPORTS bool hasNvgCtx();
    CV_EXPORTS bool hasSingleCtx();
    CV_EXPORTS bool hasParallelCtx();
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

