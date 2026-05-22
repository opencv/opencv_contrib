// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright Amir Hassan (kallaballa) <amir@viel-zu.org>

#include <opencv2/v4d/v4d.hpp>

using namespace cv::v4d;

class ShaderDemoPlan : public Plan {
public:
	using Plan::Plan;

	//A value greater 1 will enable experimental tiling with one context per tile.
	constexpr static size_t TILING_ = 1;
    constexpr static size_t NUM_CONTEXTS_ = TILING_ * TILING_;
private:
    // vertex position, color
    constexpr static float vertices[12] = {
        //    x      y      z
        -1.0f, -1.0f, -0.0f, 1.0f, 1.0f, -0.0f, -1.0f, 1.0f, -0.0f, 1.0f, -1.0f, -0.0f };

    constexpr static unsigned int indices[6] = {
        //  2---,1
        //  | .' |
        //  0'---3
        0, 1, 2, 0, 3, 1 };

    static struct Params {
        /* Mandelbrot control parameters */
        // Red, green, blue and alpha. All from 0.0f to 1.0f
        float baseColorVal_[4] = {0.2, 0.6, 1.0, 0.8};
        //contrast boost
        int contrastBoost_ = 255; //0.0-255
        //max fractal iterations
        int maxIterations_ = 50000;
        //center x coordinate
        float centerX_ = -0.466;
        //center y coordinate
        float centerY_ = 0.57052;
        float zoomFactor_ = 1.0;
        float currentZoom_ = 4.0;
        bool zoomIn = true;
        float zoomIncr_ = -currentZoom_ / 1000;
        bool manualNavigation_ = false;
    } params_;

    struct Handles {
        /* GL uniform handles */
        GLint baseColorHdl_;
        GLint contrastBoostHdl_;
        GLint maxIterationsHdl_;
        GLint centerXHdl_;
        GLint centerYHdl_;
        GLint offsetXHdl_;
        GLint offsetYHdl_;
        GLint currentZoomHdl_;
        GLint resolutionHdl_;

        /* Shader program handle */
        GLuint shaderHdl_;

        /* Object handles */
        GLuint vao_;
        GLuint vbo_, ebo_;
    } handles_[NUM_CONTEXTS_];

    cv::Rect viewports_[NUM_CONTEXTS_];

    struct Cache {
        cv::UMat down;
        cv::UMat up;
        cv::UMat blur;
        cv::UMat dst16;
    } cache_;

    //easing function for the bungee zoom
    static float easeInOutQuint(float x) {
        return x < 0.5f ? 16.0f * x * x * x * x * x : 1.0f - std::pow(-2.0f * x + 2.0f, 5.0f) / 2.0f;
    }

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
        #if !defined(OPENCV_V4D_USE_ES3)
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
        precision highp float;

        out vec4 outColor;

        uniform vec4 base_color;
        uniform int contrast_boost;
        uniform int max_iterations;
        uniform float current_zoom;
        uniform float center_y;
        uniform float center_x;
        uniform float offset_y;
        uniform float offset_x;

        uniform vec2 resolution;

        int get_iterations()
        {
            float pointr = (((gl_FragCoord.x / resolution[0]) - 0.5f) * current_zoom + center_x);
            float pointi = (((gl_FragCoord.y / resolution[1]) - 0.5f) * current_zoom + center_y);
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

        void mandelbrot()
        {
            int iter = get_iterations();
            if (iter < max_iterations) {
                float iterations = float(iter) / float(max_iterations);
                float cb = float(contrast_boost);
                float logBase;
                if(iter % 2 == 0)
					logBase = 25.0f;
				else
					logBase = 50.0f;
                
				float logDiv = log2(logBase);
				float colorBoost = iterations * cb;
				outColor = vec4(log2((logBase - 1.0f) * base_color[0] * colorBoost + 1.0f)/logDiv, 
								log2((logBase - 1.0f) * base_color[1] * colorBoost + 1.0f)/logDiv, 
								log2((logBase - 1.0f) * base_color[2] * colorBoost + 1.0f)/logDiv, 
								base_color[3]);
            } else {
                outColor = vec4(0,0,0,0);
            }
        }

        void main()
        {
            mandelbrot();
        })";
        unsigned int handles[3];
        cv::v4d::initShader(handles, vert.c_str(), frag.c_str(), "fragColor");
        return handles[0];
    }

    //Initialize shaders, objects, buffers and uniforms
    static void init_scene(const cv::Rect& viewport, Handles& handles) {
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
        handles.offsetXHdl_ = glGetUniformLocation(handles.shaderHdl_, "offset_x");
        handles.offsetYHdl_ = glGetUniformLocation(handles.shaderHdl_, "offset_y");
        handles.resolutionHdl_ = glGetUniformLocation(handles.shaderHdl_, "resolution");
        GL_CHECK(glViewport(viewport.x, viewport.y, viewport.width, viewport.height));
    }

    //Free OpenGL resources
    static void destroy_scene(Handles& handles) {
        glDeleteShader(handles.shaderHdl_);
        glDeleteBuffers(1, &handles.vbo_);
        glDeleteBuffers(1, &handles.ebo_);
        glDeleteVertexArrays(1, &handles.vao_);
    }

    //Render the mandelbrot fractal on top of a video
    static void render_scene(const cv::Size& sz, const cv::Rect& viewport, Params& params, Handles& handles) {
        GL_CHECK(glViewport(viewport.x, viewport.y, viewport.width, viewport.height));

        //bungee zoom
        if (params.currentZoom_ >= 3) {
            params.zoomIn = true;
        } else if (params.currentZoom_ < 0.05) {
        	params.zoomIn = false;
        }

        params.zoomIncr_ = (params.currentZoom_ / 100);
    	if(params.zoomIn)
    		params.zoomIncr_ = -params.zoomIncr_;

        GL_CHECK(glUseProgram(handles.shaderHdl_));
        GL_CHECK(glUniform4f(handles.baseColorHdl_, params.baseColorVal_[0], params.baseColorVal_[1], params.baseColorVal_[2], params.baseColorVal_[3]));
        GL_CHECK(glUniform1i(handles.contrastBoostHdl_, params.contrastBoost_));
        GL_CHECK(glUniform1i(handles.maxIterationsHdl_, params.maxIterations_));
        GL_CHECK(glUniform1f(handles.centerYHdl_, params.centerY_));
        GL_CHECK(glUniform1f(handles.centerXHdl_, params.centerX_));
        GL_CHECK(glUniform1f(handles.offsetYHdl_, viewport.x));
        GL_CHECK(glUniform1f(handles.offsetXHdl_, viewport.y));

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
    ShaderDemoPlan(const cv::Rect& viewport) : Plan(viewport) {
		Global::registerShared(params_);
	}

	ShaderDemoPlan(const cv::Size& sz) : ShaderDemoPlan(cv::Rect(0,0,sz.width, sz.height)) {
	}

    void gui(cv::Ptr<V4D> window) override {
        window->imgui([](cv::Ptr<V4D> win, ImGuiContext* ctx, Params& params) {
            CV_UNUSED(win);
            using namespace ImGui;
            SetCurrentContext(ctx);
            Begin("Fractal");
            Text("Navigation");
            SliderInt("Iterations", &params.maxIterations_, 3, 100000);
            DragFloat("X", &params.centerX_, 0.000001, -1.0f, 1.0f);
            DragFloat("Y", &params.centerY_, 0.000001, -1.0f, 1.0f);
            if(SliderFloat("Zoom", &params.zoomFactor_, 0.0001f, 10.0f))
                params.manualNavigation_ = true;
            Text("Color");
            ColorPicker4("Color", params.baseColorVal_);
            SliderInt("Contrast boost", &params.contrastBoost_, 1, 255);
            End();
        }, params_);
    }

    void setup(cv::Ptr<V4D> window) override {
        float w = size().width;
        float h = size().height;
        float tw = w / TILING_;
        float th = h / TILING_;

        for(size_t i = 0; i < TILING_; ++i) {
            for(size_t j = 0; j < TILING_; ++j) {
            	viewports_[i * TILING_ + j] = cv::Rect(tw * i, th * j, tw - 1, th - 1);
            }
        }

        for(size_t i = 0; i < NUM_CONTEXTS_; ++i) {
            window->gl(i, [](const int32_t& ctxID, const cv::Rect& viewport, Handles& handles) {
                init_scene(viewport, handles);
            }, viewports_[i], handles_[i]);
        }
    }

    void infer(cv::Ptr<V4D> window) override {
        window->capture();

        for(size_t i = 0; i < NUM_CONTEXTS_; ++i) {
            window->gl(i,[](const int32_t& ctxID, const cv::Size& sz, const cv::Rect& viewport, Params& params, Handles& handles) {
            	Params p = Global::safe_copy(params);
                render_scene(sz, viewport, p, handles);
            }, size(), viewports_[i], params_, handles_[i]);
        }

        window->write();
    }

    void teardown(cv::Ptr<V4D> window) override {
        for(size_t i = 0; i < NUM_CONTEXTS_; ++i) {
            window->gl(i, [](const int32_t& ctxID, Handles& handles) {
                destroy_scene(handles);
            }, handles_[i]);
        }
    }
};

ShaderDemoPlan::Params ShaderDemoPlan::params_;

int main(int argc, char** argv) {
    if (argc != 2) {
        cerr << "Usage: shader-demo <video-file>" << endl;
        exit(1);
    }

    cv::Ptr<ShaderDemoPlan> plan = new ShaderDemoPlan(cv::Size(1280, 720));
	cv::Ptr<V4D> window = V4D::make(plan->size(), "Mandelbrot Shader Demo", IMGUI);

	auto src = makeCaptureSource(window, argv[1]);
	auto sink = makeWriterSink(window, "shader-demo.mkv", src->fps(), plan->size());
	window->setSource(src);
	window->setSink(sink);

	window->run(plan);

	return 0;
}
