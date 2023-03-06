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

const int kernel_size = std::max(int(DIAG / 300 % 2 == 0 ? DIAG / 300 + 1 : DIAG / 300), 1);

//mandelbrot control parameters
int max_iterations_val = 500;
float center_x_val = -0.32485;
float center_y_val = 0.00001;
float zoom_val = 1.0;
long iterations = 0;

//WebGL uniforms
GLint max_iterations;
GLint center_x;
GLint center_y;
GLint zoom;

//shader handle
GLuint shader_program;

#ifndef __EMSCRIPTEN__
//vertex array
GLuint VAO;
#endif

void load_buffer_data(){
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

//mandelbrot shader code adapted from https://physicspython.wordpress.com/2020/02/16/visualizing-the-mandelbrot-set-using-opengl-part-1/
void load_shader(){
#ifndef __EMSCRIPTEN__
    const string vertHeader = R"(
    #version 150
)";
#else
    const string vertHeader = R"(
    #version 300 es
)";
#endif

#ifndef __EMSCRIPTEN__
    const string fragHeader = R"(
    #version 150
    in vec4 gl_FragCoord;
)";

#else
    const string fragHeader = R"(
    #version 300 es
)";
#endif

    const string  vert = vertHeader + R"(

    in vec4 position;

    void main()
    {
        gl_Position = vec4(position.xyz, 1.0);
    })";

    const string frag = fragHeader + R"(
    precision highp float;

    out vec4 outColor;
    
    uniform int max_iterations;
    uniform float zoom;
    uniform float center_x;
    uniform float center_y;
    
    int get_iterations()
    {
        float pointr = ((gl_FragCoord.x / 1080.0f - 0.5f) * zoom + center_x) * 5.0f;
        float pointi = ((gl_FragCoord.y / 1080.0f - 0.5f) * zoom + center_y) * 5.0f;
     
        int iterations = 0;
        float four = 4.0f;
        float zi = 0;
        float zr = 0;
        float zrsqr = 0;
        float zisqr = 0;

        while (iterations < max_iterations && zrsqr + zisqr <= four)
        {
            //equals line below as a consequence of binomial expansion: zi = (square(zr + zi) - zrsqr) - zisqr
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
        if (iter == max_iterations)
        {   
            gl_FragDepth = 0.0f;
            return vec4(0.0f, 0.0f, 0.0f, 1.0f);
        }
        float r = 0.2;
        float g = 0.6;
        float b = 1.0;
        float contrast = 15.0;
        float iterations = float(iter) / float(max_iterations);
        return vec4(r * iterations * contrast, g * iterations * contrast, b * iterations * contrast, 1.0f);
    }
     
    void main()
    {
        outColor = return_color();
    })";

    shader_program = init_shader(vert.c_str(),  frag.c_str(), "fragColor");
}

void init_scene(const cv::Size& sz) {
    load_shader();
    load_buffer_data();
    max_iterations = glGetUniformLocation(shader_program, "max_iterations");
    zoom = glGetUniformLocation(shader_program, "zoom");
    center_x = glGetUniformLocation(shader_program, "center_x");
    center_y = glGetUniformLocation(shader_program, "center_y");

    glViewport(0, 0, WIDTH, HEIGHT);
    glEnable(GL_DEPTH_TEST);
}

void render_scene(const cv::Size& sz) {
    glClearColor(0.2f, 0.0f, 0.2f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    if(zoom_val < 1.0e-07) {
        center_x_val = -0.32485;
        center_y_val = 0.00001;
        zoom_val = 1.0;
        iterations = 0;
    }

    glUseProgram(shader_program);
    glUniform1i(max_iterations, max_iterations_val);
    glUniform1f(center_y, center_y_val);
    glUniform1f(center_x, center_x_val);
    glUniform1f(zoom, zoom_val*=0.95);

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

cv::Ptr<kb::viz2d::Viz2D> v2d = new kb::viz2d::Viz2D(cv::Size(WIDTH, HEIGHT), cv::Size(WIDTH, HEIGHT), OFFSCREEN, "Shader Demo");

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
#else
    //in WASM the shaders and nanovg don't work together
    update_fps(v2d, false);
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
