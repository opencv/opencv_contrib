#define CL_TARGET_OPENCL_VERSION 120

#include "../common/viz2d.hpp"
#include "../common/util.hpp"

constexpr long unsigned int WIDTH = 1920;
constexpr long unsigned int HEIGHT = 1080;
constexpr double FPS = 60;
constexpr bool OFFSCREEN = false;
constexpr const char* OUTPUT_FILENAME = "tetra-demo.mkv";
constexpr const int VA_HW_DEVICE_INDEX = 0;
const unsigned long DIAG = hypot(double(WIDTH), double(HEIGHT));

const int kernel_size = std::max(int(DIAG / 138 % 2 == 0 ? DIAG / 138 + 1 : DIAG / 138), 1);

using std::cerr;
using std::endl;

GLuint vertexBuffer, vertexArrayObject, shaderProgram;
GLint positionAttribute, colorAttribute;

void loadBufferData(){
    // vertex position, color
    float vertexData[32] = {
         -0.5, -0.5, 0.0, 1.0 ,  1.0, 0.0, 0.0, 1.0  ,
         -0.5,  0.5, 0.0, 1.0 ,  0.0, 1.0, 0.0, 1.0  ,
          0.5,  0.5, 0.0, 1.0 ,  0.0, 0.0, 1.0, 1.0  ,
          0.5, -0.5, 0.0, 1.0 ,  1.0, 1.0, 1.0, 1.0
    };

#ifndef __EMSCRIPTEN__
    glGenVertexArrays(1, &vertexArrayObject);
    glBindVertexArray(vertexArrayObject);
#endif

    glGenBuffers(1, &vertexBuffer);
    glBindBuffer(GL_ARRAY_BUFFER, vertexBuffer);
    glBufferData(GL_ARRAY_BUFFER, 32 * sizeof(float), vertexData, GL_STATIC_DRAW);

    glEnableVertexAttribArray(positionAttribute);
    glEnableVertexAttribArray(colorAttribute);
    int vertexSize =sizeof(float)*8;
    glVertexAttribPointer(positionAttribute, 4, GL_FLOAT, GL_FALSE,vertexSize , (const GLvoid *)0);
    glVertexAttribPointer(colorAttribute  , 4, GL_FLOAT, GL_FALSE, vertexSize, (const GLvoid *)(sizeof(float)*4));
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

void loadShader(){
#ifndef __EMSCRIPTEN__
    const char *  vert = R"(#version 150
    in vec4 position;
    in vec4 color;
    
    out vec4 colorV;
    
    void main (void)
    {
        colorV = color;
        gl_Position = position;
    })";
#else
    const char *  vert = R"(#version 100
    attribute vec4 position;
    attribute vec4 color;
    
    varying vec4 colorV;
    
    void main (void)
    {
        colorV = color;
        gl_Position = position / vec4(2.0, 2.0, 1.0, 1.0) - vec4(0.4, 0.4, 0.0, 0.0);
    })";
#endif
#ifndef __EMSCRIPTEN__
    const char *  frag = R"(#version 150
    
    in vec4 colorV;
    out vec4 fragColor;
    
    void main(void)
    {
        fragColor = colorV;
    })";
#else
    const char *  frag = R"(#version 100
    precision mediump float;

    varying vec4 colorV;

    void main(void)
    {
        gl_FragColor = colorV;
    })";
#endif
    shaderProgram = initShader(vert,  frag, "fragColor");

    colorAttribute = glGetAttribLocation(shaderProgram, "color");
    if (colorAttribute < 0) {
        std::cerr << "Shader did not contain the 'color' attribute." << std::endl;
    }
    positionAttribute = glGetAttribLocation(shaderProgram, "position");
    if (positionAttribute < 0) {
        std::cerr << "Shader did not contain the 'position' attribute." << std::endl;
    }
}

void init_scene(const cv::Size& sz) {
    loadShader();
    loadBufferData();

    glEnable(GL_DEPTH_TEST);
}

void render_scene(const cv::Size& sz) {
    glClearColor( 0.0f, 0.0f, 1.0f, 1.0f );
    glClear( GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT );

    glUseProgram(shaderProgram);

    glDrawArrays(GL_TRIANGLE_FAN, 0, 4);
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
    //Reinitialize the scene on every frame because nvg interfers
    v2d->gl(init_scene);

    //Render using OpenGL
    v2d->gl(render_scene);

//To slow for wasm
#ifndef __EMSCRIPTEN__
    //Aquire the frame buffer for use by OpenCL
    v2d->clgl([](cv::UMat &frameBuffer) {
        //Glow effect (OpenCL)
        glow_effect(frameBuffer, frameBuffer, kernel_size);
    });
#endif
    update_fps(v2d, true);
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
