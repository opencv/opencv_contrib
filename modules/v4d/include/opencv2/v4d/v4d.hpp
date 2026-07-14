// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright Amir Hassan (kallaballa) <amir@viel-zu.org>

#ifndef SRC_OPENCV_V4D_V4D_HPP_
#define SRC_OPENCV_V4D_V4D_HPP_

#include "source.hpp"
#include "sink.hpp"
#include "util.hpp"
#include "nvg.hpp"
#include "threadsafemap.hpp"
#include "detail/transaction.hpp"
#include "detail/framebuffercontext.hpp"
#include "detail/nanovgcontext.hpp"
#include "detail/imguicontext.hpp"
#include "detail/timetracker.hpp"
#include "detail/glcontext.hpp"
#include "detail/sourcecontext.hpp"
#include "detail/sinkcontext.hpp"
#include "detail/resequence.hpp"
#include "events.hpp"

#include <type_traits>
#include <shared_mutex>
#include <iostream>
#include <future>
#include <set>
#include <map>
#include <string>
#include <memory>
#include <vector>
#include <type_traits>
#include <stdio.h>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/core/utils/logger.hpp>



using std::cout;
using std::cerr;
using std::endl;
using std::string;
using namespace std::chrono_literals;


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
	const cv::Size sz_;
	const cv::Rect vp_;
public:

	//predefined branch predicates
	constexpr static auto always_ = []() { return true; };
	constexpr static auto isTrue_ = [](const bool& b) { return b; };
	constexpr static auto isFalse_ = [](const bool& b) { return !b; };
	constexpr static auto and_ = [](const bool& a, const bool& b) { return a && b; };
	constexpr static auto or_ = [](const bool& a, const bool& b) { return a || b; };

	explicit Plan(const cv::Rect& vp) : sz_(cv::Size(vp.width, vp.height)), vp_(vp){};
	explicit Plan(const cv::Size& sz) : sz_(sz), vp_(0, 0, sz.width, sz.height){};
	virtual ~Plan() {};

	virtual void gui(cv::Ptr<V4D> window) { CV_UNUSED(window); };
	virtual void setup(cv::Ptr<V4D> window) { CV_UNUSED(window); };
	virtual void infer(cv::Ptr<V4D> window) = 0;
	virtual void teardown(cv::Ptr<V4D> window) { CV_UNUSED(window); };

	const cv::Size& size() {
		return sz_;
	}
	const cv::Rect& viewport() {
		return vp_;
	}
};
/*!
 * Private namespace
 */
namespace detail {

template <typename T> using static_not = std::integral_constant<bool, !T::value>;

template<typename T, typename ... Args>
struct is_function
{
    static const bool value = std::is_constructible<T,std::function<void(Args...)>>::value;
};

//https://stackoverflow.com/a/34873353/1884837
template<class T>
struct is_stateless_lambda : std::integral_constant<bool, sizeof(T) == sizeof(std::true_type)>{};

template<typename T> std::string int_to_hex( T i )
{
  std::stringstream stream;
  stream << "0x"
         << std::setfill ('0') << std::setw(sizeof(T) * 2)
         << std::hex << i;
  return stream.str();
}

template<typename Tlamba> std::string lambda_ptr_hex(Tlamba&& l) {
    return int_to_hex((size_t)Lambda::ptr(l));
}

static std::size_t index(const std::thread::id id)
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

template<typename Tfn, typename ... Args>
const string make_id(const string& name, Tfn&& fn, Args&& ... args) {
	stringstream ss;
	ss << name << "(" << index(std::this_thread::get_id()) << "-" << detail::lambda_ptr_hex(std::forward<Tfn>(fn)) << ")";
	((ss << ',' << int_to_hex((long)&args)), ...);
	return ss.str();
}

}


using namespace cv::v4d::detail;

class CV_EXPORTS V4D {
	friend class detail::FrameBufferContext;
    friend class HTML5Capture;
    int32_t workerIdx_ = -1;
    cv::Ptr<V4D> self_;
    cv::Ptr<Plan> plan_;
    const cv::Size initialSize_;
    AllocateFlags flags_;
    bool debug_;
    cv::Rect viewport_;
    bool stretching_;
    int samples_;
    bool focused_ = false;
    cv::Ptr<FrameBufferContext> mainFbContext_ = nullptr;
    cv::Ptr<SourceContext> sourceContext_ = nullptr;
    cv::Ptr<SinkContext> sinkContext_ = nullptr;
    cv::Ptr<NanoVGContext> nvgContext_ = nullptr;
    cv::Ptr<ImGuiContextImpl> imguiContext_ = nullptr;
    cv::Ptr<OnceContext> onceContext_ = nullptr;
    cv::Ptr<PlainContext> plainContext_ = nullptr;
    std::mutex glCtxMtx_;
    std::map<int32_t,cv::Ptr<GLContext>> glContexts_;
    bool closed_ = false;
    cv::Ptr<Source> source_;
    cv::Ptr<Sink> sink_;
    cv::UMat captureFrame_;
    cv::UMat writerFrame_;
    std::function<bool(int key, int scancode, int action, int modifiers)> keyEventCb_;
    std::function<void(int button, int action, int modifiers)> mouseEventCb_;
    cv::Point2f mousePos_;
    uint64_t frameCnt_ = 0;
    bool showFPS_ = true;
    bool printFPS_ = false;
    bool showTracking_ = true;
    std::vector<std::tuple<std::string,bool,long>> accesses_;
    std::map<std::string, cv::Ptr<Transaction>> transactions_;
    bool disableIO_ = false;
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
    CV_EXPORTS static cv::Ptr<V4D> make(const cv::Size& size, const string& title, AllocateFlags flags = ALL, bool offscreen = false, bool debug = false, int samples = 0);
    CV_EXPORTS static cv::Ptr<V4D> make(const cv::Size& size, const cv::Size& fbsize, const string& title, AllocateFlags flags = ALL, bool offscreen = false, bool debug = false, int samples = 0);
    CV_EXPORTS static cv::Ptr<V4D> make(const V4D& v4d, const string& title);
    /*!
     * Default destructor
     */
    CV_EXPORTS virtual ~V4D();

    CV_EXPORTS const int32_t& workerIndex() const;
    CV_EXPORTS size_t workers_running();
    /*!
     * The internal framebuffer exposed as OpenGL Texture2D.
     * @return The texture object.
     */
    CV_EXPORTS cv::ogl::Texture2D& texture();
    CV_EXPORTS std::string title() const;

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

    void makeGraph() {
//    	cout << std::this_thread::get_id() << " ### MAKE PLAN ### " << endl;
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
//        		cout << "make: " << std::this_thread::get_id() << " " << n->name_ << endl;
    		}


    		if(read) {
    			n->read_deps_.insert(dep);
    		} else {
    			n->write_deps_.insert(dep);
    		}
    	}
    }

    void runGraph() {
		bool isEnabled = true;

		for (auto& n : nodes_) {
			if (n->tx_->isPredicate()) {
				isEnabled = n->tx_->enabled();
			} else if (isEnabled) {
				if(n->tx_->lock()) {
					std::lock_guard<std::mutex> guard(Global::mutex());
					n->tx_->getContext()->execute([n]() {
						TimeTracker::getInstance()->execute(n->name_, [n](){
							n->tx_->perform();
						});
					});
				} else {
					n->tx_->getContext()->execute([n]() {
						TimeTracker::getInstance()->execute(n->name_, [n](){
							n->tx_->perform();
						});
					});
				}
			}
		}
	}

	void clearGraph() {
		nodes_.clear();
		accesses_.clear();
	}

    template<typename Tenabled, typename T, typename ...Args>
    typename std::enable_if<std::is_same<Tenabled, std::false_type>::value, void>::type
	emit_access(const string& context, bool read, const T* tp) {
    	//disabled
    }

    template<typename Tenabled, typename T, typename ...Args>
    typename std::enable_if<std::is_same<Tenabled, std::true_type>::value, void>::type
	emit_access(const string& context, bool read, const T* tp) {
//    	cout << "access: " << std::this_thread::get_id() << " " << context << string(read ? " <- " : " -> ") << demangle(typeid(std::remove_const_t<T>).name()) << "(" << (long)tp << ") " << endl;
    	accesses_.push_back(std::make_tuple(context, read, (long)tp));
    }

    template<typename Tfn, typename ...Args>
    void add_transaction(bool lock, cv::Ptr<V4DContext> ctx, const string& invocation, Tfn fn, Args&& ...args) {
    	auto it = transactions_.find(invocation);
    	if(it == transactions_.end()) {
    		auto tx = make_transaction(lock, fn, std::forward<Args>(args)...);
    		tx->setContext(ctx);
    		transactions_.insert({invocation, tx});
    	}
    }

    template <typename Tfn, typename ... Args>
    void init_context_call(Tfn fn, Args&& ... args) {
    	static_assert(detail::is_stateless_lambda<std::remove_cv_t<std::remove_reference_t<decltype(fn)>>>::value, "All passed functors must be stateless lambdas");
    	static_assert(std::conjunction<std::is_lvalue_reference<Args>...>::value, "All arguments must be l-value references");
    	cv::v4d::event::set_current_glfw_window(getGLFWWindow());
    }


    template <typename Tfn, typename ... Args>
    typename std::enable_if<std::is_invocable_v<Tfn, Args...>, void>::type
    gl(Tfn fn, Args&& ... args) {
    	init_context_call(fn, args...);
        const string id = make_id("gl-1", fn, args...);
		emit_access<std::true_type, cv::UMat, Args...>(id, true, &fbCtx()->fb());
		(emit_access<std::true_type, std::remove_reference_t<Args>, Args...>(id, std::is_const_v<std::remove_reference_t<Args>>, &args),...);
		emit_access<std::true_type, cv::UMat, Args...>(id, false, &fbCtx()->fb());
		std::function functor(fn);
		add_transaction(false, glCtx(-1), id, std::forward<decltype(functor)>(fn), std::forward<Args>(args)...);
    }

    template <typename Tfn, typename ... Args>
    void gl(int32_t idx, Tfn fn, Args&& ... args) {
        init_context_call(fn, args...);

        const string id = make_id("gl" + std::to_string(idx), fn, args...);
		emit_access<std::true_type, cv::UMat, Args...>(id, true, &fbCtx()->fb());
		(emit_access<std::true_type, std::remove_reference_t<Args>, Args...>(id, std::is_const_v<std::remove_reference_t<Args>>, &args),...);
		emit_access<std::true_type, cv::UMat, Args...>(id, false, &fbCtx()->fb());
		std::function<void((const int32_t&,Args...))> functor(fn);
		add_transaction<decltype(functor),const int32_t&>(false, glCtx(idx),id, std::forward<decltype(functor)>(functor), glCtx(idx)->getIndex(), std::forward<Args>(args)...);
    }

    template <typename Tfn>
    void branch(Tfn fn) {
        init_context_call(fn);
        const string id = make_id("branch", fn);
		std::function functor = fn;
		emit_access<std::true_type, decltype(fn)>(id, true, &fn);
		add_transaction(true, plainCtx(), id, functor);
    }

    template <typename Tfn, typename ... Args>
    void branch(Tfn fn, Args&& ... args) {
        init_context_call(fn, args...);

        const string id = make_id("branch", fn, args...);

		(emit_access<std::true_type, std::remove_reference_t<Args>, Args...>(id, std::is_const_v<std::remove_reference_t<Args>>, &args),...);
		std::function functor = fn;
		add_transaction(true, plainCtx(), id, functor, std::forward<Args>(args)...);
    }

    template <typename Tfn, typename ... Args>
    void branch(int workerIdx, Tfn fn, Args&& ... args) {
        init_context_call(fn, args...);

        const string id = make_id("branch-pin" + std::to_string(workerIdx), fn, args...);

		(emit_access<std::true_type, std::remove_reference_t<Args>, Args...>(id, std::is_const_v<std::remove_reference_t<Args>>, &args),...);
		std::function<bool(Args...)> functor = fn;
		std::function<bool(Args...)> wrap = [this, workerIdx, functor](Args&& ... args){
			return this->workerIndex() == workerIdx && functor(args...);
		};
		add_transaction(true, plainCtx(), id, wrap, std::forward<Args>(args)...);
    }

    template <typename Tfn>
    void endbranch(Tfn fn) {
        init_context_call(fn);
        const string id = make_id("endbranch", fn);

		std::function functor = fn;
		emit_access<std::true_type, decltype(fn)>(id, true, &fn);
		add_transaction(true, plainCtx(), id, functor);
    }

    template <typename Tfn, typename ... Args>
    void endbranch(Tfn fn, Args&& ... args) {
        init_context_call(fn, args...);

        const string id = make_id("endbranch", fn, args...);

		(emit_access<std::true_type, std::remove_reference_t<Args>, Args...>(id, std::is_const_v<std::remove_reference_t<Args>>, &args),...);
		std::function<bool(Args...)> functor = [this](Args&& ... args){
			return true;
		};
		add_transaction(true, plainCtx(), id, functor, std::forward<Args>(args)...);
    }

    template <typename Tfn, typename ... Args>
    void endbranch(int workerIdx, Tfn fn, Args&& ... args) {
        init_context_call(fn, args...);

        const string id = make_id("endbranch-pin" + std::to_string(workerIdx), fn, args...);

		(emit_access<std::true_type, std::remove_reference_t<Args>, Args...>(id, std::is_const_v<std::remove_reference_t<Args>>, &args),...);
		std::function<bool(Args...)> functor = [this, workerIdx](Args&& ... args){
			return this->workerIndex() == workerIdx;
		};
		add_transaction(true, plainCtx(), id, functor, std::forward<Args>(args)...);
    }

    template <typename Tfn, typename ... Args>
    void fb(Tfn fn, Args&& ... args) {
        init_context_call(fn, args...);

        const string id = make_id("fb", fn, args...);
		using Tfb = std::add_lvalue_reference_t<typename std::tuple_element<0, typename function_traits<Tfn>::argument_types>::type>;
		using Tfbbase = typename std::remove_cv<Tfb>::type;

		static_assert((std::is_same<Tfb, cv::UMat&>::value || std::is_same<Tfb, const cv::UMat&>::value) || !"The first argument must be eiter of type 'cv::UMat&' or 'const cv::UMat&'");
		emit_access<std::true_type, cv::UMat, Tfb, Args...>(id, true, &fbCtx()->fb());
		(emit_access<std::true_type, std::remove_reference_t<Args>, Tfb, Args...>(id, std::is_const_v<std::remove_reference_t<Args>>, &args),...);
		emit_access<static_not<typename std::is_const<Tfbbase>::type>, cv::UMat, Tfb, Args...>(id, false, &fbCtx()->fb());
		std::function<void((Tfb,Args...))> functor(fn);
		add_transaction<decltype(functor),Tfb>(false, fbCtx(),id, std::forward<decltype(functor)>(functor), fbCtx()->fb(), std::forward<Args>(args)...);
    }

    void capture() {
    	if(disableIO_)
    		return;
    	capture([](const cv::UMat& inputFrame, cv::UMat& f){
    		if(!inputFrame.empty())
    			inputFrame.copyTo(f);
    	}, captureFrame_);

        fb([](cv::UMat& frameBuffer, const cv::UMat& f) {
        	if(!f.empty())
        		f.copyTo(frameBuffer);
        }, captureFrame_);
    }

    template <typename Tfn, typename ... Args>
    void capture(Tfn fn, Args&& ... args) {
        init_context_call(fn, args...);


    	if(disableIO_)
    		return;
        const string id = make_id("capture", fn, args...);
		using Tfb = std::add_lvalue_reference_t<typename std::tuple_element<0, typename function_traits<Tfn>::argument_types>::type>;

		static_assert((std::is_same<Tfb,const cv::UMat&>::value) || !"The first argument must be of type 'const cv::UMat&'");
		emit_access<std::true_type, cv::UMat, Tfb, Args...>(id, true, &sourceCtx()->sourceBuffer());
		(emit_access<std::true_type, std::remove_reference_t<Args>, Tfb, Args...>(id, std::is_const_v<std::remove_reference_t<Args>>, &args),...);
		std::function<void((Tfb,Args...))> functor(fn);
		add_transaction<decltype(functor),Tfb>(false, std::dynamic_pointer_cast<V4DContext>(sourceCtx()),id, std::forward<decltype(functor)>(functor), sourceCtx()->sourceBuffer(), std::forward<Args>(args)...);
    }

    void write() {
    	if(disableIO_)
    		return;

        fb([](const cv::UMat& frameBuffer, cv::UMat& f) {
            frameBuffer.copyTo(f);
        }, writerFrame_);

    	write([](cv::UMat& outputFrame, const cv::UMat& f){
    		f.copyTo(outputFrame);
    	}, writerFrame_);
    }

    template <typename Tfn, typename ... Args>
    void write(Tfn fn, Args&& ... args) {
        init_context_call(fn, args...);


    	if(disableIO_)
    		return;
        const string id = make_id("write", fn, args...);
		using Tfb = std::add_lvalue_reference_t<typename std::tuple_element<0, typename function_traits<Tfn>::argument_types>::type>;

		static_assert((std::is_same<Tfb,cv::UMat&>::value) || !"The first argument must be of type 'cv::UMat&'");
		emit_access<std::true_type, cv::UMat, Tfb, Args...>(id, true, &sinkCtx()->sinkBuffer());
		(emit_access<std::true_type, std::remove_reference_t<Args>, Tfb, Args...>(id, std::is_const_v<std::remove_reference_t<Args>>, &args),...);
		emit_access<std::true_type, cv::UMat, Tfb, Args...>(id, false, &sinkCtx()->sinkBuffer());
		std::function<void((Tfb,Args...))> functor(fn);
		add_transaction<decltype(functor),Tfb>(false, std::dynamic_pointer_cast<V4DContext>(sinkCtx()),id, std::forward<decltype(functor)>(functor), sinkCtx()->sinkBuffer(), std::forward<Args>(args)...);
    }

    template <typename Tfn, typename ... Args>
    void nvg(Tfn fn, Args&&... args) {
        init_context_call(fn, args...);

        const string id = make_id("nvg", fn, args...);
		emit_access<std::true_type, cv::UMat, Args...>(id, true, &fbCtx()->fb());
		(emit_access<std::true_type, std::remove_reference_t<Args>, Args...>(id, std::is_const_v<std::remove_reference_t<Args>>, &args),...);
		emit_access<std::true_type, cv::UMat, Args...>(id, false, &fbCtx()->fb());
		std::function functor(fn);
		add_transaction<decltype(functor)>(false, nvgCtx(), id, std::forward<decltype(functor)>(fn), std::forward<Args>(args)...);
    }

    template <typename Tfn, typename ... Args>
    void once(Tfn fn, Args&&... args) {
        CV_Assert(detail::is_stateless_lambda<std::remove_cv_t<std::remove_reference_t<decltype(fn)>>>::value);
        const string id = make_id("once", fn, args...);
		(emit_access<std::true_type, std::remove_reference_t<Args>, Args...>(id, std::is_const_v<std::remove_reference_t<Args>>, &args),...);
		std::function functor(fn);
		add_transaction<decltype(functor)>(false, onceCtx(), id, std::forward<decltype(functor)>(fn), std::forward<Args>(args)...);
    }

    template <typename Tfn, typename ... Args>
    void plain(Tfn fn, Args&&... args) {
        init_context_call(fn, args...);

        const string id = make_id("plain", fn, args...);
		(emit_access<std::true_type, std::remove_reference_t<Args>, Args...>(id, std::is_const_v<std::remove_reference_t<Args>>, &args),...);
		std::function functor(fn);
		add_transaction<decltype(functor)>(false, fbCtx(), id, std::forward<decltype(functor)>(fn), std::forward<Args>(args)...);
    }

    template<typename Tfn, typename ... Args>
    void imgui(Tfn fn, Args&& ... args) {
    	init_context_call(fn, args...);

        if(!hasImguiCtx())
        	return;

        auto s = self();

        imguiCtx()->build([s, fn, &args...](ImGuiContext* ctx) {
			fn(s, ctx, args...);
		});
    }
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

	template<typename Tplan>
	void run(cv::Ptr<Tplan> plan, int32_t workers = -1) {
		plan_ = std::static_pointer_cast<Plan>(plan);

		static Resequence reseq;
		//for now, if automatic determination of the number of workers is requested,
		//set workers always to 2
		CV_Assert(workers > -2);
		if(workers == -1) {
			workers = 2;
		} else {
			++workers;
		}

		std::vector<std::thread*> threads;
		{
			static std::mutex runMtx;
			std::unique_lock<std::mutex> lock(runMtx);

			cerr << "run plan: " << std::this_thread::get_id() << " workers: " << workers << endl;

			if(Global::is_first_run()) {
				Global::set_main_id(std::this_thread::get_id());
				cerr << "Starting with " << workers - 1<< " extra workers" << endl;
				cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_SILENT);
			}

			if(workers > 1) {
				cv::setNumThreads(0);
			}

			if(Global::is_main()) {
				cv::Size sz = this->initialSize();
				const string title = this->title();
				bool debug = this->debug_;
				auto src = this->getSource();
				auto sink = this->getSink();
				Global::set_workers_started(workers);
				std::vector<cv::Ptr<Tplan>> plans;
				//make sure all Plans are constructed before starting the workers
				for (size_t i = 0; i < workers; ++i) {
					plans.push_back(new Tplan(plan->size()));
				}
				for (size_t i = 0; i < workers; ++i) {
					threads.push_back(
						new std::thread(
							[this, i, src, sink, plans] {
								cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_SILENT);
								cv::Ptr<cv::v4d::V4D> worker = V4D::make(*this, this->title() + "-worker-" + std::to_string(i));
								if (src) {
									worker->setSource(src);
								}
								if (sink) {
									worker->setSink(sink);
								}
								cv::Ptr<Tplan> newPlan = plans[i];
								worker->run(newPlan, 0);
							}
						)
					);
				}
			}
		}

		CLExecScope_t scope(this->fbCtx()->getCLExecContext());
		this->fbCtx()->makeCurrent();

		if(Global::is_main()) {
			this->printSystemInfo();
		} else {
			try {
				plan->setup(self());
				this->makeGraph();
				this->runGraph();
				this->clearGraph();
				if(!Global::is_main() && Global::workers_started() == Global::next_worker_ready()) {
					cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_INFO);
				}
			} catch(std::exception& ex) {
				CV_Error_(cv::Error::StsError, ("pipeline setup failed: %s", ex.what()));
			}
		}
		if(Global::is_main()) {
			try {
				plan->gui(self());
			} catch(std::exception& ex) {
				CV_Error_(cv::Error::StsError, ("GUI setup failed: %s", ex.what()));
			}
		} else {
			plan->infer(self());
			this->makeGraph();
		}

		try {
			if(Global::is_main()) {
				do {
					//refresh-rate depends on swap interval (1) for sync
				} while(keepRunning() && this->display());
				requestFinish();
				reseq.finish();
			} else {
				cerr << "Starting pipeling with " << this->nodes_.size() << " nodes." << endl;

				static std::mutex seqMtx;
				do {
					reseq.notify();
					uint64_t seq;
					{
						std::unique_lock<std::mutex> lock(seqMtx);
						seq = Global::next_run_cnt();
					}

					this->runGraph();
					reseq.waitFor(seq);
				} while(keepRunning() && this->display());
			}
		} catch(std::exception& ex) {
			requestFinish();
			reseq.finish();
			CV_LOG_WARNING(nullptr, "-> pipeline terminated: " << ex.what());
		}

		if(!Global::is_main()) {
			this->clearGraph();

			try {
				plan->teardown(self());
				this->makeGraph();
				this->runGraph();
				this->clearGraph();
			} catch(std::exception& ex) {
				CV_Error_(cv::Error::StsError, ("pipeline tear-down failed: %s", ex.what()));
			}
		} else {
			for(auto& t : threads)
				t->join();
		}
	}
/*!
     * Called to feed an image directly to the framebuffer
     */
	void feed(cv::UMat& in);
    /*!
     * Fetches a copy of frambuffer
     * @return a copy of the framebuffer
     */
    CV_EXPORTS cv::UMat fetch();

    /*!
     * Set the current #cv::viz::Source object. Usually created using #makeCaptureSource().
     * @param src A #cv::viz::Source object.
     */
    CV_EXPORTS void setSource(cv::Ptr<Source> src);
    CV_EXPORTS cv::Ptr<Source> getSource();
    CV_EXPORTS bool hasSource();

    /*!
     * Set the current #cv::viz::Sink object. Usually created using #makeWriterSink().
     * @param sink A #cv::viz::Sink object.
     */
    CV_EXPORTS void setSink(cv::Ptr<Sink> sink);
    CV_EXPORTS cv::Ptr<Sink> getSink();
    CV_EXPORTS bool hasSink();
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
    CV_EXPORTS const cv::Size& initialSize() const;
    CV_EXPORTS const cv::Size& fbSize() const;
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
    CV_EXPORTS void setDisableIO(bool d);

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
    CV_EXPORTS const uint64_t& frameCount() const;
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

    CV_EXPORTS GLFWwindow* getGLFWWindow() const;

    CV_EXPORTS cv::Ptr<FrameBufferContext> fbCtx() const;
    CV_EXPORTS cv::Ptr<SourceContext> sourceCtx();
    CV_EXPORTS cv::Ptr<SinkContext> sinkCtx();
    CV_EXPORTS cv::Ptr<NanoVGContext> nvgCtx();
    CV_EXPORTS cv::Ptr<OnceContext> onceCtx();
    CV_EXPORTS cv::Ptr<PlainContext> plainCtx();
    CV_EXPORTS cv::Ptr<ImGuiContextImpl> imguiCtx();
    CV_EXPORTS cv::Ptr<GLContext> glCtx(int32_t idx = 0);

    CV_EXPORTS bool hasFbCtx();
    CV_EXPORTS bool hasSourceCtx();
    CV_EXPORTS bool hasSinkCtx();
    CV_EXPORTS bool hasNvgCtx();
    CV_EXPORTS bool hasOnceCtx();
    CV_EXPORTS bool hasParallelCtx();
    CV_EXPORTS bool hasImguiCtx();
    CV_EXPORTS bool hasGlCtx(uint32_t idx = 0);
    CV_EXPORTS size_t numGlCtx();
private:
    V4D(const V4D& v4d, const string& title);
    V4D(const cv::Size& size, const cv::Size& fbsize,
            const string& title, AllocateFlags flags, bool offscreen, bool debug, int samples);

    cv::Point2f getMousePosition();
    void setMousePosition(const cv::Point2f& pt);

    void swapContextBuffers();
protected:
    AllocateFlags flags();
    cv::Ptr<V4D> self();
    void fence();
    bool wait(uint64_t timeout = 0);
};
}
} /* namespace cv */

#endif /* SRC_OPENCV_V4D_V4D_HPP_ */

