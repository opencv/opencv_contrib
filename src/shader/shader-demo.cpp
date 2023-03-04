#define CL_TARGET_OPENCL_VERSION 120

#include "../common/viz2d.hpp"
#include "../common/util.hpp"

constexpr long unsigned int WIDTH = 1920;
constexpr long unsigned int HEIGHT = 1080;
constexpr double FPS = 60;
constexpr bool OFFSCREEN = false;
constexpr const char* OUTPUT_FILENAME = "shader-demo.mkv";
constexpr const int VA_HW_DEVICE_INDEX = 0;
const unsigned long DIAG = hypot(double(WIDTH), double(HEIGHT));

const int kernel_size = std::max(int(DIAG / 138 % 2 == 0 ? DIAG / 138 + 1 : DIAG / 138), 1);

//mandelbrot control parameters
GLint centerX;
GLint centerY;
GLint zoom;
float centerXVal = -0.5;
float centerYVal = 0;
float zoomLevel = 1.0;

using std::cerr;
using std::endl;

GLuint vertexBuffer, VAO, shaderProgram;
GLint positionAttribute, colorAttribute;

void loadBufferData(){
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

    unsigned int VBO, EBO;

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

GLuint initShader(const char* vShader, const char* fShader, const char* outputAttributeName) {
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

//mandelbrot shader code adapted from https://physicspython.wordpress.com/2020/02/16/visualizing-the-mandelbrot-set-using-opengl-part-1/
void loadShader(){
#ifndef __EMSCRIPTEN__
    const char *  vert = R"(#version 150
in vec4 position;

void main()
{
    gl_Position = vec4(position.xyz, 1.0);
})";
#else
    const char *  vert = R"(#version 300 es
    in vec4 position;

    void main (void)
    {
        gl_Position = vec4(position.xyz, 1.0);
    })";
#endif
#ifndef __EMSCRIPTEN__
    const char *  frag = R"(#version 150
in vec4 gl_FragCoord;

out vec4 outColor;

uniform float zoom;
uniform float center_x;
uniform float center_y;

#define MAX_ITERATIONS 500
 
int get_iterations()
{
    float real = ((gl_FragCoord.x / 1080.0 - 0.5) * zoom + center_x) * 5.0;
    float imag = ((gl_FragCoord.y / 1080.0 - 0.5) * zoom + center_y) * 5.0;
 
    int iterations = 0;
    float const_real = real;
    float const_imag = imag;
 
    while (iterations < MAX_ITERATIONS)
    {
        float tmp_real = real;
        real = (real * real - imag * imag) + const_real;
        imag = (2.0 * tmp_real * imag) + const_imag;
         
        float dist = real * real + imag * imag;
         
        if (dist > 4.0)
        break;
 
        ++iterations;
    }
    return iterations;
}
 
vec4 return_color()
{
    int iter = get_iterations();
    if (iter == MAX_ITERATIONS)
    {   
        gl_FragDepth = 0.0f;
        return vec4(0.0f, 0.0f, 0.0f, 1.0f);
    }
    float r = 0.225;
    float g = 0.45;
    float b = 0.9;
    float lightness = 4.0;
    float iterations = float(iter) / MAX_ITERATIONS;
    return vec4(r * iterations * lightness, g * iterations * lightness, b * iterations * lightness, 1.0f);
}
 
void main()
{
    outColor = return_color();
})";
#else
    const char *  frag = R"(#version 300 es
precision mediump float;

out vec4 outColor;

uniform float zoom;
uniform float center_x;
uniform float center_y;

#define MAX_ITERATIONS 500
 
int get_iterations()
{
    float real = ((gl_FragCoord.x / 1080.0 - 0.5) * zoom + center_x) * 5.0;
    float imag = ((gl_FragCoord.y / 1080.0 - 0.5) * zoom + center_y) * 5.0;
 
    int iterations = 0;
    float const_real = real;
    float const_imag = imag;
 
    while (iterations < MAX_ITERATIONS)
    {
        float tmp_real = real;
        real = (real * real - imag * imag) + const_real;
        imag = (2.0 * tmp_real * imag) + const_imag;
         
        float dist = real * real + imag * imag;
         
        if (dist > 4.0)
        break;
 
        ++iterations;
    }
    return iterations;
}
 
vec4 return_color()
{
    int iter = get_iterations();
    if (iter == MAX_ITERATIONS)
    {   
        gl_FragDepth = 0.0f;
        return vec4(0.0f, 0.0f, 0.0f, 1.0f);
    }
    float r = 0.225;
    float g = 0.45;
    float b = 0.9;
    float lightness = 4.0;
    float iterations = float(iter) / float(MAX_ITERATIONS);
    return vec4(r * iterations * lightness, g * iterations * lightness, b * iterations * lightness, 1.0f);
}
 
void main()
{
    outColor = return_color();
})";
#endif
    shaderProgram = initShader(vert,  frag, "fragColor");
}

long iterations = 0;
void init_scene(const cv::Size& sz) {
    loadShader();
    loadBufferData();
    zoom = glGetUniformLocation(shaderProgram, "zoom");
    centerX = glGetUniformLocation(shaderProgram, "center_x");
    centerY = glGetUniformLocation(shaderProgram, "center_y");

    glViewport(0, 0, WIDTH, HEIGHT);
    glEnable(GL_DEPTH_TEST);
}


void render_scene(const cv::Size& sz) {
    glClearColor(0.2f, 0.0f, 0.2f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    if(iterations > 700) {
        centerXVal = -0.5;
        zoomLevel = 1.0;
        iterations = 0;
    }
    glUseProgram(shaderProgram);
    glUniform1f(centerY, centerYVal);
    glUniform1f(centerX, centerXVal*=0.9992);
    glUniform1f(zoom, zoomLevel*=0.99);

#ifndef __EMSCRIPTEN__
    glBindVertexArray(VAO);
#endif
    glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);
    ++iterations;
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

cv::Ptr<kb::viz2d::Viz2D> v2d = new kb::viz2d::Viz2D(cv::Size(WIDTH, HEIGHT), cv::Size(WIDTH, HEIGHT), OFFSCREEN, "Tetra Demo");

void iteration() {
    //Render using OpenGL
    v2d->gl(render_scene);

//To slow for wasm
#ifndef __EMSCRIPTEN__
    //Aquire the frame buffer for use by OpenCL
    v2d->clgl([](cv::UMat &frameBuffer) {
        //Glow effect (OpenCL)
        glow_effect(frameBuffer, frameBuffer, kernel_size);
    });

    update_fps(v2d, true);
#endif

#ifndef __EMSCRIPTEN__
    v2d->write();
#endif
    //If onscreen rendering is enabled it displays the framebuffer in the native window. Returns false if the window was closed.
    if (!v2d->display())
        exit(0);
}

int main(int argc, char **argv) {
    using namespace kb::viz2d;
    try {
        print_system_info();
        if(!v2d->isOffscreen())
            v2d->setVisible(true);
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
