#define CL_TARGET_OPENCL_VERSION 120

#include "../common/viz2d.hpp"
#include "../common/util.hpp"

using std::cerr;
using std::endl;

constexpr long unsigned int WIDTH = 1920;
constexpr long unsigned int HEIGHT = 1080;
constexpr double FPS = 60;
constexpr bool OFFSCREEN = false;
constexpr const char* OUTPUT_FILENAME = "shader-demo.mkv";
constexpr const int VA_HW_DEVICE_INDEX = 0;
const unsigned long DIAG = hypot(double(WIDTH), double(HEIGHT));

int glow_kernel_size = std::max(int(DIAG / 100 % 2 == 0 ? DIAG / 100 + 1 : DIAG / 100), 1);

/** mandelbrot control parameters **/
// Red, green, blue and alpha. All from 0.0f to 1.0f
nanogui::Color base_color_val(0.2f, 0.6f, 1.0f, 1.0f);
// Keep alpha separate for the GUI
float alpha = 1.0f; //0.0-1.0
//contrast boost
int contrast_boost = 15; //0.0-255
int max_iterations = 500;
float center_x = -0.32487;
float center_y = 0.000001;
float zoom = 1.0;
float zoom_multiplier = 0.99;
long iterations = 0;

/** GL uniform handles **/
GLint base_color_hdl;
GLint contrast_boost_hdl;
GLint max_iterations_hdl;
GLint center_x_hdl;
GLint center_y_hdl;
GLint zoom_hdl;

/** shader and program handle **/
GLuint shader_program_hdl;

#ifndef __EMSCRIPTEN__
//vertex array
GLuint VAO;
#endif
GLuint VBO, EBO;

// vertex position, color
float vertices[] =
{
//    x      y      z
    -1.0f, -1.0f, -0.0f,
     1.0f,  1.0f, -0.0f,
    -1.0f,  1.0f, -0.0f,
     1.0f, -1.0f, -0.0f
};

unsigned int indices[] =
{
//  2---,1
//  | .' |
//  0'---3
    0, 1, 2,
    0, 3, 1
};

void load_buffer_data(){

#ifndef __EMSCRIPTEN__
    glGenVertexArrays(1, &VAO);
    glBindVertexArray(VAO);
#endif

    glGenBuffers(1, &VBO);
    glGenBuffers(1, &EBO);

    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);

    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);

    glBindBuffer(GL_ARRAY_BUFFER, 0);
#ifndef __EMSCRIPTEN__
    glBindVertexArray(0);
#endif
}

//workaround: required with emscripten + nanogui on every iteration before renderin
void rebind_buffers() {
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);

    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);

    glBindBuffer(GL_ARRAY_BUFFER, 0);
}

GLuint init_shader(const char* vShader, const char* fShader, const char* outputAttributeName) {
    struct Shader {
        GLenum       type;
        const char*      source;
    }  shaders[2] = {
        { GL_VERTEX_SHADER, vShader },
        { GL_FRAGMENT_SHADER, fShader }
    };

    GLuint program = glCreateProgram();

    for ( int i = 0; i < 2; ++i ) {
        Shader& s = shaders[i];
        GLuint shader = glCreateShader( s.type );
        glShaderSource( shader, 1, (const GLchar**) &s.source, NULL );
        glCompileShader( shader );

        GLint  compiled;
        glGetShaderiv( shader, GL_COMPILE_STATUS, &compiled );
        if ( !compiled ) {
            std::cerr << " failed to compile:" << std::endl;
            GLint  logSize;
            glGetShaderiv( shader, GL_INFO_LOG_LENGTH, &logSize );
            char* logMsg = new char[logSize];
            glGetShaderInfoLog( shader, logSize, NULL, logMsg );
            std::cerr << logMsg << std::endl;
            delete [] logMsg;

            exit( EXIT_FAILURE );
        }

        glAttachShader( program, shader );
    }
#ifndef __EMSCRIPTEN__
    /* Link output */
    glBindFragDataLocation(program, 0, outputAttributeName);
#endif
    /* link  and error check */
    glLinkProgram(program);

    GLint  linked;
    glGetProgramiv( program, GL_LINK_STATUS, &linked );
    if ( !linked ) {
        std::cerr << "Shader program failed to link" << std::endl;
        GLint  logSize;
        glGetProgramiv( program, GL_INFO_LOG_LENGTH, &logSize);
        char* logMsg = new char[logSize];
        glGetProgramInfoLog( program, logSize, NULL, logMsg );
        std::cerr << logMsg << std::endl;
        delete [] logMsg;

        exit( EXIT_FAILURE );
    }

    /* use program object */
    glUseProgram(program);

    return program;
}

//mandelbrot shader code adapted from my own project: https://github.com/kallaballa/FractalDive#after
void load_shader(){
#ifndef __EMSCRIPTEN__
    const string shaderVersion = "330";
#else
    const string shaderVersion = "300 es";
#endif

    const string vert = "    #version " + shaderVersion + R"(
    in vec4 position;

    void main()
    {
        gl_Position = vec4(position.xyz, 1.0);
    })";

    const string frag = "#version " + shaderVersion + R"(
    precision highp float;

    out vec4 outColor;
    
    uniform vec4 base_color;
    uniform int contrast_boost;
    uniform int max_iterations;
    uniform float zoom;
    uniform float center_x;
    uniform float center_y;
    
    int get_iterations()
    {
        float pointr = ((gl_FragCoord.x / 1080.0f - 0.5f) * zoom + center_x) * 5.0f;
        float pointi = ((gl_FragCoord.y / 1080.0f - 0.5f) * zoom + center_y) * 5.0f;
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
     
    vec4 return_color()
    {
        int iter = get_iterations();
        if (iter == max_iterations) {   
            return vec4(0.0f, 0.0f, 0.0f, 1.0f);
        }

        float iterations = float(iter) / float(max_iterations);
        //convert to float
        float cb = float(contrast_boost);

        return vec4(base_color[0] * iterations * cb, base_color[1] * iterations * cb, base_color[2] * iterations * cb, base_color[3]);
    }
     
    void main()
    {
        outColor = return_color();
    })";

    cerr << "##### Vertex Shader #####" << endl;
    cerr << vert << endl;

    cerr << "##### Fragment Shader #####" << endl;
    cerr << frag << endl;

    shader_program_hdl = init_shader(vert.c_str(),  frag.c_str(), "fragColor");
}

void init_scene(const cv::Size& sz) {
    load_shader();
    load_buffer_data();

    base_color_hdl = glGetUniformLocation(shader_program_hdl, "base_color");
    contrast_boost_hdl = glGetUniformLocation(shader_program_hdl, "contrast_boost");
    max_iterations_hdl = glGetUniformLocation(shader_program_hdl, "max_iterations");
    zoom_hdl = glGetUniformLocation(shader_program_hdl, "zoom");
    center_x_hdl = glGetUniformLocation(shader_program_hdl, "center_x");
    center_y_hdl = glGetUniformLocation(shader_program_hdl, "center_y");

    glViewport(0, 0, WIDTH, HEIGHT);
    glEnable(GL_DEPTH_TEST);
}

void render_scene(const cv::Size& sz) {
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    if(zoom > 1) {
        zoom_multiplier = 0.99;
        iterations = 0;
    } else if(zoom < 2.5e-06) {
        zoom_multiplier = 1.01;
        iterations = 0;
    }

    glUseProgram(shader_program_hdl);
    glUniform4f(base_color_hdl, base_color_val[0], base_color_val[1], base_color_val[2], alpha);
    glUniform1i(contrast_boost_hdl, contrast_boost);
    glUniform1i(max_iterations_hdl, max_iterations);
    glUniform1f(center_y_hdl, center_y);
    glUniform1f(center_x_hdl, center_x);
    glUniform1f(zoom_hdl, zoom*=zoom_multiplier);

#ifndef __EMSCRIPTEN__
    glBindVertexArray(VAO);
#endif
    glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);
}

void glow_effect(const cv::UMat &src, cv::UMat &dst, const int ksize) {
    static cv::UMat resize;
    static cv::UMat blur;
    static cv::UMat dst16;

    cv::bitwise_not(src, dst);

    //Resize for some extra performance
    cv::resize(dst, resize, cv::Size(), 0.5, 0.5);
    //Cheap blur
    cv::boxFilter(resize, resize, -1, cv::Size(ksize, ksize), cv::Point(-1,-1), true, cv::BORDER_REPLICATE);
    //Back to original size
    cv::resize(resize, blur, src.size());

    //Multiply the src image with a blurred version of itself
    cv::multiply(dst, blur, dst16, 1, CV_16U);
    //Normalize and convert back to CV_8U
    cv::divide(dst16, cv::Scalar::all(255.0), dst, 1, CV_8U);

    cv::bitwise_not(dst, dst);
}

cv::Ptr<kb::viz2d::Viz2D> v2d = new kb::viz2d::Viz2D(cv::Size(WIDTH, HEIGHT), cv::Size(WIDTH, HEIGHT), OFFSCREEN, "Shader Demo");

void setup_gui(cv::Ptr<kb::viz2d::Viz2D> v2d) {
    v2d->makeWindow(5, 30, "Effect");
#ifndef __EMSCRIPTEN__
    v2d->makeGroup("Glow");
    auto* kernelSize = v2d->makeFormVariable("Kernel Size", glow_kernel_size, 1, 127, true, "", "Intensity of glow defined by kernel size");
    kernelSize->set_callback([=](const int& k) {
        static int lastKernelSize = glow_kernel_size;

        if(k == lastKernelSize)
            return;

        if(k <= lastKernelSize) {
            glow_kernel_size = std::max(int(k % 2 == 0 ? k - 1 : k), 1);
        } else if(k > lastKernelSize)
            glow_kernel_size = std::max(int(k % 2 == 0 ? k + 1 : k), 1);

        lastKernelSize = k;
        kernelSize->set_value(glow_kernel_size);
    });
#endif
    v2d->makeGroup("Color");
    v2d->makeColorPicker("Color", base_color_val, "The base color of the fractal visualization",[&](const nanogui::Color &c) {
        base_color_val[0] = c[0];
        base_color_val[1] = c[1];
        base_color_val[2] = c[2];
    });
    v2d->makeFormVariable("Alpha", alpha, 0.0f, 1.0f, true, "", "The opacity of the fractal visualization");
    v2d->makeFormVariable("Contrast boost", contrast_boost, 1, 255, true, "", "Boost contrast by this factor");

}

void iteration() {
#ifdef __EMSCRIPTEN__
    //required in conjunction with emscripten + nanovg
    rebind_buffers();
#endif
    //Render using OpenGL
    v2d->gl(render_scene);

//To slow for WASM but works
#ifndef __EMSCRIPTEN__
    //Aquire the frame buffer for use by OpenCL
    v2d->clgl([](cv::UMat &frameBuffer) {
        //Glow effect (OpenCL)
        glow_effect(frameBuffer, frameBuffer, glow_kernel_size);
    });
#endif

    update_fps(v2d, true);

#ifndef __EMSCRIPTEN__
    v2d->write();
#endif
    //If onscreen rendering is enabled it displays the framebuffer in the native window. Returns false if the window was closed.
    if (!v2d->display())
        exit(0);

    ++iterations;
}

int main(int argc, char **argv) {
    using namespace kb::viz2d;
    try {
        print_system_info();
        if(!v2d->isOffscreen()) {
            setup_gui(v2d);
            v2d->setVisible(true);
        }
#ifndef __EMSCRIPTEN__
        v2d->makeVAWriter(OUTPUT_FILENAME, cv::VideoWriter::fourcc('V', 'P', '9', '0'), FPS, v2d->getFrameBufferSize(), 0);
#endif
        v2d->gl(init_scene);

#ifndef __EMSCRIPTEN__
        while(true)
            iteration();
#else
        emscripten_set_main_loop(iteration, -1, false);
#endif
    } catch(std::exception& ex) {
        cerr << "Exception: " << ex.what() << endl;
    }
    return 0;
}
