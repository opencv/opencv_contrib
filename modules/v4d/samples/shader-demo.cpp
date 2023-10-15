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

/* Mandelbrot control parameters */
static int glow_kernel_size = std::max(int(DIAG / 200 % 2 == 0 ? DIAG / 200 + 1 : DIAG / 200), 1);
// Red, green, blue and alpha. All from 0.0f to 1.0f
static float base_color_val[4] = {0.2, 0.6, 1.0, 1.0};
//contrast boost
static int contrast_boost = 50; //0.0-255
//max fractal iterations
static int max_iterations = 1000;
//center x coordinate
static float center_x = -0.119609;
//center y coordinate
static float center_y = 0.13262;
static float zoom_factor = 1.0;
static float current_zoom = 1.0;
static float zoom_incr = 0.99;
static bool manual_navigation = false;

/* GL uniform handles */
static thread_local GLint base_color_hdl;
static thread_local GLint contrast_boost_hdl;
static thread_local GLint max_iterations_hdl;
static thread_local GLint center_x_hdl;
static thread_local GLint center_y_hdl;
static thread_local GLint current_zoom_hdl;
static thread_local GLint resolution_hdl;

/* Shader program handle */
static thread_local GLuint shader_program_hdl;

/* Object handles */
static thread_local GLuint VAO;
static thread_local GLuint VBO, EBO;

// vertex position, color
static const float vertices[] = {
//    x      y      z
        -1.0f, -1.0f, -0.0f, 1.0f, 1.0f, -0.0f, -1.0f, 1.0f, -0.0f, 1.0f, -1.0f, -0.0f };

static const unsigned int indices[] = {
//  2---,1
//  | .' |
//  0'---3
        0, 1, 2, 0, 3, 1 };

//Load objects and buffers
static void load_buffer_data() {
    glGenVertexArrays(1, &VAO);
    glBindVertexArray(VAO);

    glGenBuffers(1, &VBO);
    glGenBuffers(1, &EBO);

    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);

    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*) 0);
    glEnableVertexAttribArray(0);

    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);
}

//mandelbrot shader code adapted from my own project: https://github.com/kallaballa/FractalDive#after
static void load_shader() {
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
    uniform float center_x;
    uniform float center_y;
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

    shader_program_hdl = cv::v4d::initShader(vert.c_str(), frag.c_str(), "fragColor");
}

//easing function for the bungee zoom
static float easeInOutQuint(float x) {
    return x < 0.5f ? 16.0f * x * x * x * x * x : 1.0f - std::pow(-2.0f * x + 2.0f, 5.0f) / 2.0f;
}

//Initialize shaders, objects, buffers and uniforms
static void init_scene(const cv::Size& sz) {
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    load_shader();
    load_buffer_data();

    base_color_hdl = glGetUniformLocation(shader_program_hdl, "base_color");
    contrast_boost_hdl = glGetUniformLocation(shader_program_hdl, "contrast_boost");
    max_iterations_hdl = glGetUniformLocation(shader_program_hdl, "max_iterations");
    current_zoom_hdl = glGetUniformLocation(shader_program_hdl, "current_zoom");
    center_x_hdl = glGetUniformLocation(shader_program_hdl, "center_x");
    center_y_hdl = glGetUniformLocation(shader_program_hdl, "center_y");
    resolution_hdl = glGetUniformLocation(shader_program_hdl, "resolution");
    glViewport(0, 0, sz.width, sz.height);
}

//Render the mandelbrot fractal on top of a video
static void render_scene(const cv::Size& sz) {
	//bungee zoom
    if (current_zoom >= 1) {
        zoom_incr = -0.01;
    } else if (current_zoom < 2.5e-06) {
        zoom_incr = +0.01;
    }

    glUseProgram(shader_program_hdl);
    glUniform4f(base_color_hdl, base_color_val[0], base_color_val[1], base_color_val[2], base_color_val[3]);
    glUniform1i(contrast_boost_hdl, contrast_boost);
    glUniform1i(max_iterations_hdl, max_iterations);
    glUniform1f(center_y_hdl, center_y);
    glUniform1f(center_x_hdl, center_x);
    if (!manual_navigation) {
        current_zoom += zoom_incr;
        glUniform1f(current_zoom_hdl, easeInOutQuint(current_zoom));
    } else {
        current_zoom = 1.0 / pow(zoom_factor, 5.0f);
        glUniform1f(current_zoom_hdl, current_zoom);
    }
    float res[2] = {float(sz.width), float(sz.height)};
    glUniform2fv(resolution_hdl, 1, res);

    glBindVertexArray(VAO);
    glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);
}

#ifndef __EMSCRIPTEN__
static void glow_effect(const cv::UMat& src, cv::UMat& dst, const int ksize) {
    static thread_local cv::UMat resize;
    static thread_local cv::UMat blur;
    static thread_local cv::UMat dst16;

    cv::bitwise_not(src, dst);

    cv::resize(dst, resize, cv::Size(), 0.5, 0.5);
    cv::boxFilter(resize, resize, -1, cv::Size(ksize, ksize), cv::Point(-1, -1), true,
            cv::BORDER_REPLICATE);
    cv::resize(resize, blur, src.size());

    cv::multiply(dst, blur, dst16, 1, CV_16U);
    cv::divide(dst16, cv::Scalar::all(255.0), dst, 1, CV_8U);

    cv::bitwise_not(dst, dst);
}
#endif

using namespace cv::v4d;

//Setup the GUI using ImGUI.
static void setup_gui(cv::Ptr<V4D> window) {
    window->imgui([](ImGuiContext* ctx) {
        using namespace ImGui;
        SetCurrentContext(ctx);
        Begin("Fractal");
        Text("Navigation");
        SliderInt("Iterations", &max_iterations, 3, 50000);
        if(SliderFloat("X", &center_x, -1.0f, 1.0f))
            manual_navigation = true;

        if(SliderFloat("Y", &center_y, -1.0f, 1.0f))
            manual_navigation = true;

        if(SliderFloat("Zoom", &zoom_factor, 1.0f, 100.0f))
            manual_navigation = true;
#ifndef __EMSCRIPTEN__
        Text("Glow");
        SliderInt("Kernel Size", &glow_kernel_size, 1, 127);
#endif
        Text("Color");
        ColorPicker4("Color", base_color_val);
        SliderInt("Contrast boost", &contrast_boost, 1, 255);
        End();
    });
}

class ShaderDemoPlan : public Plan {
public:
void setup(cv::Ptr<V4D> window) override {
	window->gl([](const cv::Size &sz) {
		init_scene(sz);
	}, window->fbSize());
}

void infer(cv::Ptr<V4D> window) override {
	window->capture();

	window->gl([](const cv::Size &sz) {
		render_scene(sz);
	}, window->fbSize());

#ifndef __EMSCRIPTEN__
    window->fb([](cv::UMat& framebuffer) {
        glow_effect(framebuffer, framebuffer, glow_kernel_size);
    });
#endif

	window->write();
}
};

#ifndef __EMSCRIPTEN__
int main(int argc, char** argv) {
    if (argc != 2) {
        cerr << "Usage: shader-demo <video-file>" << endl;
        exit(1);
    }
#else
int main() {
#endif
    try {
        cv::Ptr<V4D> window = V4D::make(WIDTH, HEIGHT, "Mandelbrot Shader Demo", IMGUI, OFFSCREEN, false, 0);
        if (!OFFSCREEN) {
            setup_gui(window);
        }


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
