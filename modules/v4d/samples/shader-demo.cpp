// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright Amir Hassan (kallaballa) <amir@viel-zu.org>

#include <opencv2/v4d/v4d.hpp>

using std::cerr;
using std::endl;

/* Demo parameters */
#ifndef __EMSCRIPTEN__
constexpr long unsigned int WIDTH = 1280;
constexpr long unsigned int HEIGHT = 720;
#else
constexpr long unsigned int WIDTH = 960;
constexpr long unsigned int HEIGHT = 960;
#endif
constexpr bool OFFSCREEN = false;
const unsigned long DIAG = hypot(double(WIDTH), double(HEIGHT));
#ifndef __EMSCRIPTEN__
constexpr const char* OUTPUT_FILENAME = "shader-demo.mkv";
#endif

// vertex position, color
static const float vertices[] = {
//    x      y      z
        -1.0f, -1.0f, -0.0f, 1.0f, 1.0f, -0.0f, -1.0f, 1.0f, -0.0f, 1.0f, -1.0f, -0.0f };

static const unsigned int indices[] = {
//  2---,1
//  | .' |
//  0'---3
        0, 1, 2, 0, 3, 1 };

//easing function for the bungee zoom
static float easeInOutQuint(float x) {
    return x < 0.5f ? 16.0f * x * x * x * x * x : 1.0f - std::pow(-2.0f * x + 2.0f, 5.0f) / 2.0f;
}

using namespace cv::v4d;

class ShaderDemoPlan : public Plan {
	struct Params {
		/* Mandelbrot control parameters */
		int glowKernelSize_ = std::max(int(DIAG / 200 % 2 == 0 ? DIAG / 200 + 1 : DIAG / 200), 1);
		// Red, green, blue and alpha. All from 0.0f to 1.0f
		float baseColorVal_[4] = {0.2, 0.6, 1.0, 1.0};
		//contrast boost
		int contrastBoost_ = 50; //0.0-255
		//max fractal iterations
		int maxIterations_ = 1000;
		//center x coordinate
		float centerX_ = -0.119609;
		//center y coordinate
		float centerY_ = 0.13262;
		float zoomFactor_ = 1.0;
		float currentZoom_ = 1.0;
		float zoomIncr_ = 0.99;
		bool manualNavigation_ = false;
	} params_;

	struct Handles {
		/* GL uniform handles */
		GLint baseColorHdl_;
		GLint contrastBoostHdl_;
		GLint maxIterationsHdl_;
		GLint centerXHdl_;
		GLint centerYHdl_;
		GLint currentZoomHdl_;
		GLint resolutionHdl_;

		/* Shader program handle */
		GLuint shaderHdl_;

		/* Object handles */
		GLuint vao_;
		GLuint vbo_, ebo_;
	} handles_;

	struct Cache {
		cv::UMat down;
		cv::UMat up;
		cv::UMat blur;
		cv::UMat dst16;
	} cache_;

	cv::Size sz_;
#ifndef __EMSCRIPTEN__
	static void glow_effect(const cv::UMat& src, cv::UMat& dst, const int ksize, Cache& cache) {
		cv::bitwise_not(src, dst);

		cv::resize(dst, cache.down, cv::Size(), 0.5, 0.5);
		cv::boxFilter(cache.down, cache.blur, -1, cv::Size(ksize, ksize), cv::Point(-1, -1), true,
				cv::BORDER_REPLICATE);
		cv::resize(cache.blur, cache.up, src.size());

		cv::multiply(dst, cache.up, cache.dst16, 1, CV_16U);
		cv::divide(cache.dst16, cv::Scalar::all(255.0), dst, 1, CV_8U);

		cv::bitwise_not(dst, dst);
	}
#endif

	//Load objects and buffers
	static void load_buffers(Handles& handles) {
	    GL_CHECK(glGenVertexArrays(1, &handles.vao_));
	    GL_CHECK(glBindVertexArray(handles.vao_));

	    GL_CHECK(glGenBuffers(1, &handles.vbo_));
	    GL_CHECK(glGenBuffers(1, &handles.ebo_));

	    GL_CHECK(glBindBuffer(GL_ARRAY_BUFFER, handles.vbo_));
	    GL_CHECK(glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW));

	    GL_CHECK(glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, handles.ebo_));
	    GL_CHECK(glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW));

	    GL_CHECK(glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*) 0));
	    GL_CHECK(glEnableVertexAttribArray(0));

	    GL_CHECK(glBindBuffer(GL_ARRAY_BUFFER, 0));
	    GL_CHECK(glBindVertexArray(0));
	}

	//mandelbrot shader code adapted from my own project: https://github.com/kallaballa/FractalDive#after
	static GLuint load_shader() {
	#if !defined(__EMSCRIPTEN__) && !defined(OPENCV_V4D_USE_ES3)
	    const string shaderVersion = "330";
	#else
	    const string shaderVersion = "300 es";
	#endif

	    const string vert =
	            "    #version " + shaderVersion
	                    + R"(
	    in vec4 position;
	    
	    void main()
	    {
	        gl_Position = vec4(position.xyz, 1.0);
	    })";

	    const string frag =
	            "    #version " + shaderVersion
	                    + R"(
	    precision lowp float;

	    out vec4 outColor;
	    
	    uniform vec4 base_color;
	    uniform int contrast_boost;
	    uniform int max_iterations;
	    uniform float current_zoom;
	    uniform float center_y;
	    uniform float center_x;
		uniform vec2 resolution;

	    int get_iterations()
	    {
	        float pointr = (((gl_FragCoord.x / resolution[1]) - 0.5f) * current_zoom + center_x) * 5.0f;
	        float pointi = (((gl_FragCoord.y / resolution[1]) - 0.5f) * current_zoom + center_y) * 5.0f;
	        const float four = 4.0f;

	        int iterations = 0;
	        float zi = 0.0f;
	        float zr = 0.0f;
	        float zrsqr = 0.0f;
	        float zisqr = 0.0f;

	        while (iterations < max_iterations && zrsqr + zisqr < four) {
	           //equals following line as a consequence of binomial expansion: zi = (((zr + zi)*(zr + zi)) - zrsqr) - zisqr
	            zi = (zr + zr) * zi;

	            zi += pointi;
	            zr = (zrsqr - zisqr) + pointr;
	    
	            zrsqr = zr * zr;
	            zisqr = zi * zi;
	            ++iterations;
	        }
	        return iterations;
	    }
	     
	    void determine_color()
	    {
	        int iter = get_iterations();
	        if (iter < max_iterations) {   
	            float iterations = float(iter) / float(max_iterations);
	            float cb = float(contrast_boost);
	    
	            outColor = vec4(base_color[0] * iterations * cb, base_color[1] * iterations * cb, base_color[2] * iterations * cb, base_color[3]);
	        } else {
	            outColor = vec4(0,0,0,0);
	        }
	    }

	    void main()
	    {
	        determine_color();
	    })";

	    return cv::v4d::initShader(vert.c_str(), frag.c_str(), "fragColor");
	}

	//Initialize shaders, objects, buffers and uniforms
	static void init_scene(const cv::Size& sz, Handles& handles) {
	    GL_CHECK(glEnable(GL_BLEND));
	    GL_CHECK(glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA));
	    handles.shaderHdl_ = load_shader();
	    load_buffers(handles);

	    handles.baseColorHdl_ = glGetUniformLocation(handles.shaderHdl_, "base_color");
	    handles.contrastBoostHdl_ = glGetUniformLocation(handles.shaderHdl_, "contrast_boost");
	    handles.maxIterationsHdl_ = glGetUniformLocation(handles.shaderHdl_, "max_iterations");
	    handles.currentZoomHdl_ = glGetUniformLocation(handles.shaderHdl_, "current_zoom");
	    handles.centerXHdl_ = glGetUniformLocation(handles.shaderHdl_, "center_x");
	    handles.centerYHdl_ = glGetUniformLocation(handles.shaderHdl_, "center_y");
	    handles.resolutionHdl_ = glGetUniformLocation(handles.shaderHdl_, "resolution");
	    GL_CHECK(glViewport(0, 0, sz.width, sz.height));
	}

	//Free OpenGL resources
	static void destroy_scene(Handles& handles) {
		glDeleteShader(handles.shaderHdl_);
		glDeleteBuffers(1, &handles.vbo_);
		glDeleteBuffers(1, &handles.ebo_);
		glDeleteVertexArrays(1, &handles.vao_);
	}

	//Render the mandelbrot fractal on top of a video
	static void render_scene(const cv::Size& sz, Params& params, Handles& handles) {
		//bungee zoom
	    if (params.currentZoom_ >= 1) {
	    	params.zoomIncr_ = -0.01;
	    } else if (params.currentZoom_ < 2.5e-06) {
	    	params.zoomIncr_ = +0.01;
	    }

	    GL_CHECK(glUseProgram(handles.shaderHdl_));
	    GL_CHECK(glUniform4f(handles.baseColorHdl_, params.baseColorVal_[0], params.baseColorVal_[1], params.baseColorVal_[2], params.baseColorVal_[3]));
	    GL_CHECK(glUniform1i(handles.contrastBoostHdl_, params.contrastBoost_));
	    GL_CHECK(glUniform1i(handles.maxIterationsHdl_, params.maxIterations_));
	    GL_CHECK(glUniform1f(handles.centerYHdl_, params.centerY_));
	    GL_CHECK(glUniform1f(handles.centerXHdl_, params.centerX_));
	    if (!params.manualNavigation_) {
	    	params.currentZoom_ += params.zoomIncr_;
	        GL_CHECK(glUniform1f(handles.currentZoomHdl_, easeInOutQuint(params.currentZoom_)));
	    } else {
	    	params.currentZoom_ = 1.0 / pow(params.zoomFactor_, 5.0f);
	        GL_CHECK(glUniform1f(handles.currentZoomHdl_, params.currentZoom_));
	    }
	    float res[2] = {float(sz.width), float(sz.height)};
	    GL_CHECK(glUniform2fv(handles.resolutionHdl_, 1, res));

	    GL_CHECK(glBindVertexArray(handles.vao_));
	    GL_CHECK(glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0));
	}
public:
	void gui(cv::Ptr<V4D> window) override {
		window->imgui([](cv::Ptr<V4D> win, ImGuiContext* ctx, Params& params) {
			CV_UNUSED(win);
			using namespace ImGui;
			SetCurrentContext(ctx);
			Begin("Fractal");
			Text("Navigation");
			SliderInt("Iterations", &params.maxIterations_, 3, 50000);
			if(SliderFloat("X", &params.centerX_, -1.0f, 1.0f))
				params.manualNavigation_ = true;

			if(SliderFloat("Y", &params.centerY_, -1.0f, 1.0f))
				params.manualNavigation_ = true;

			if(SliderFloat("Zoom", &params.zoomFactor_, 1.0f, 100.0f))
				params.manualNavigation_ = true;
	#ifndef __EMSCRIPTEN__
			Text("Glow");
			SliderInt("Kernel Size", &params.glowKernelSize_, 1, 127);
	#endif
			Text("Color");
			ColorPicker4("Color", params.baseColorVal_);
			SliderInt("Contrast boost", &params.contrastBoost_, 1, 255);
			End();
		}, params_);
	}

	void setup(cv::Ptr<V4D> window) override {
		sz_ = window->fbSize();
		window->gl([](const cv::Size &sz, Handles& handles) {
			init_scene(sz, handles);
		}, sz_, handles_);
	}

	void infer(cv::Ptr<V4D> window) override {
		window->capture();

		window->gl([](const cv::Size &sz, Params& params, Handles& handles) {
			render_scene(sz, params, handles);
		}, sz_, params_, handles_);

	#ifndef __EMSCRIPTEN__
		window->fb([](cv::UMat& framebuffer, const Params& params, Cache& cache) {
			glow_effect(framebuffer, framebuffer, params.glowKernelSize_, cache);
		}, params_, cache_);
	#endif

		window->write();
	}

	void teardown(cv::Ptr<V4D> window) override {
		window->gl([](Handles& handles) {
			destroy_scene(handles);
		}, handles_);
	}
};

int main(int argc, char** argv) {
#ifndef __EMSCRIPTEN__
	if (argc != 2) {
        cerr << "Usage: shader-demo <video-file>" << endl;
        exit(1);
    }
#else
	CV_UNUSED(args);
	CV_UNUSED(argv);
#endif
    try {
        cv::Ptr<V4D> window = V4D::make(WIDTH, HEIGHT, "Mandelbrot Shader Demo", IMGUI, OFFSCREEN);

#ifndef __EMSCRIPTEN__
        auto src = makeCaptureSource(window, argv[1]);
        window->setSource(src);
        auto sink = makeWriterSink(window, OUTPUT_FILENAME, src->fps(), cv::Size(WIDTH, HEIGHT));
        window->setSink(sink);
#else
        auto src = makeCaptureSource(window);
        window->setSource(src);
#endif

        window->run<ShaderDemoPlan>(0);
    } catch (std::exception& ex) {
        cerr << "Exception: " << ex.what() << endl;
    }
    return 0;
}
