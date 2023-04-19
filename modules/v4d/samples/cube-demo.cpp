// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright Amir Hassan (kallaballa) <amir@viel-zu.org>

#include <opencv2/v4d/v4d.hpp>
//adapted from https://gitlab.com/wikibooks-opengl/modern-tutorials/-/blob/master/tut05_cube/cube.cpp
#include <cstdio>
#include <cstdlib>
#include <cmath>

constexpr long unsigned int WIDTH = 1920;
constexpr long unsigned int HEIGHT = 1080;
constexpr double FPS = 60;
constexpr bool OFFSCREEN = false;
constexpr const char* OUTPUT_FILENAME = "cube-demo.mkv";
const unsigned long DIAG = hypot(double(WIDTH), double(HEIGHT));

const int GLOW_KERNEL_SIZE = std::max(int(DIAG / 138 % 2 == 0 ? DIAG / 138 + 1 : DIAG / 138),
        1);

using std::cerr;
using std::endl;

cv::Ptr<cv::viz::V4D> v4d = cv::viz::V4D::make(cv::Size(WIDTH, HEIGHT),
        cv::Size(WIDTH, HEIGHT), OFFSCREEN, "Cube Demo");

GLuint vbo_cube_vertices, vbo_cube_colors;
GLuint ibo_cube_elements;
GLuint program;
GLint attribute_coord3d, attribute_v_color;
GLint uniform_mvp;

GLuint init_shader(const char* vShader, const char* fShader, const char* outputAttributeName) {
    struct Shader {
        GLenum type;
        const char* source;
    } shaders[2] = { { GL_VERTEX_SHADER, vShader }, { GL_FRAGMENT_SHADER, fShader } };

    GLuint program = glCreateProgram();

    for (int i = 0; i < 2; ++i) {
        Shader& s = shaders[i];
        GLuint shader = glCreateShader(s.type);
        glShaderSource(shader, 1, (const GLchar**) &s.source, NULL);
        glCompileShader(shader);

        GLint compiled;
        glGetShaderiv(shader, GL_COMPILE_STATUS, &compiled);
        if (!compiled) {
            std::cerr << " failed to compile:" << std::endl;
            GLint logSize;
            glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &logSize);
            char* logMsg = new char[logSize];
            glGetShaderInfoLog(shader, logSize, NULL, logMsg);
            std::cerr << logMsg << std::endl;
            delete[] logMsg;

            exit (EXIT_FAILURE);
        }

        glAttachShader(program, shader);
    }
#ifndef OPENCV_V4D_ES_VERSION
    /* Link output */
    glBindFragDataLocation(program, 0, outputAttributeName);
#endif
    /* link  and error check */
    glLinkProgram(program);

    GLint linked;
    glGetProgramiv(program, GL_LINK_STATUS, &linked);
    if (!linked) {
        std::cerr << "Shader program failed to link" << std::endl;
        GLint logSize;
        glGetProgramiv(program, GL_INFO_LOG_LENGTH, &logSize);
        char* logMsg = new char[logSize];
        glGetProgramInfoLog(program, logSize, NULL, logMsg);
        std::cerr << logMsg << std::endl;
        delete[] logMsg;

        exit (EXIT_FAILURE);
    }

    /* use program object */
    glUseProgram(program);

    return program;
}

//mandelbrot shader code adapted from my own project: https://github.com/kallaballa/FractalDive#after
void load_shader() {
#ifndef OPENCV_V4D_ES_VERSION
    const string shaderVersion = "330";
#else
    const string shaderVersion = "300 es";
#endif

    const string vert =
            "    #version " + shaderVersion
                    + R"(
    precision lowp float;
    layout(location = 0) in vec3 coord3d;
    layout(location = 1) in vec3 v_color;
    
    uniform mat4 mvp;
    out vec3 f_color;
    
    void main(void) {
      gl_Position = mvp * vec4(coord3d, 1.0);
      f_color = v_color;
    }
)";

    const string frag =
            "    #version " + shaderVersion
                    + R"(
    precision lowp float;
    in vec3 f_color;
    out vec4 fragColor;

    void main(void) {
      fragColor = vec4(f_color.r, f_color.g, f_color.b, 1.0);
    }
)";

    cerr << "##### Vertex Shader #####" << endl;
    cerr << vert << endl;

    cerr << "##### Fragment Shader #####" << endl;
    cerr << frag << endl;

    program = init_shader(vert.c_str(), frag.c_str(), "fragColor");
}
int init_resources() {
    GLfloat cube_vertices[] = {
    // front
            -1.0, -1.0, 1.0, 1.0, -1.0, 1.0, 1.0, 1.0, 1.0, -1.0, 1.0, 1.0,
            // back
            -1.0, -1.0, -1.0, 1.0, -1.0, -1.0, 1.0, 1.0, -1.0, -1.0, 1.0, -1.0, };
    glGenBuffers(1, &vbo_cube_vertices);
    glBindBuffer(GL_ARRAY_BUFFER, vbo_cube_vertices);
    glBufferData(GL_ARRAY_BUFFER, sizeof(cube_vertices), cube_vertices, GL_STATIC_DRAW);

    GLfloat cube_colors[] = {
    // front colors
            1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0,
            // back colors
            1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, };
    glGenBuffers(1, &vbo_cube_colors);
    glBindBuffer(GL_ARRAY_BUFFER, vbo_cube_colors);
    glBufferData(GL_ARRAY_BUFFER, sizeof(cube_colors), cube_colors, GL_STATIC_DRAW);

    GLushort cube_elements[] = {
    // front
            0, 1, 2, 2, 3, 0,
            // top
            1, 5, 6, 6, 2, 1,
            // back
            7, 6, 5, 5, 4, 7,
            // bottom
            4, 0, 3, 3, 7, 4,
            // left
            4, 5, 1, 1, 0, 4,
            // right
            3, 2, 6, 6, 7, 3, };
    glGenBuffers(1, &ibo_cube_elements);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ibo_cube_elements);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(cube_elements), cube_elements, GL_STATIC_DRAW);

    GLint link_ok = GL_FALSE;

    GLuint vs, fs;

    load_shader();

    const char* attribute_name;
    attribute_name = "coord3d";
    attribute_coord3d = glGetAttribLocation(program, attribute_name);
    if (attribute_coord3d == -1) {
        fprintf(stderr, "Could not bind attribute %s\n", attribute_name);
        return 0;
    }
    attribute_name = "v_color";
    attribute_v_color = glGetAttribLocation(program, attribute_name);
    if (attribute_v_color == -1) {
        fprintf(stderr, "Could not bind attribute %s\n", attribute_name);
        return 0;
    }
    const char* uniform_name;
    uniform_name = "mvp";
    uniform_mvp = glGetUniformLocation(program, uniform_name);
    if (uniform_mvp == -1) {
        fprintf(stderr, "Could not bind uniform %s\n", uniform_name);
        return 0;
    }

    return 1;
}

void init_scene(const cv::Size& sz) {
    init_resources();
    glEnable (GL_BLEND);
    glEnable (GL_DEPTH_TEST);
//    glDepthFunc(GL_LESS);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
}

void render_scene(const cv::Size& sz) {
    glClearColor(1.0, 1.0, 1.0, 1.0);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glUseProgram(program);
    float angle = fmod(double(cv::getTickCount()) / double(cv::getTickFrequency()), 2 * M_PI);
    float scale = 0.25;

    cv::Matx44f rotXMat(
            1.0,        0.0,            0.0,    0.0,
            0.0,    cos(angle), -sin(angle),    0.0,
            0.0,    sin(angle), cos(angle),     0.0,
            0.0,        0.0,            0.0,    1.0
    );

    cv::Matx44f rotYMat(
            cos(angle),     0.0, sin(angle),    0.0,
            0.0,            1.0, 0.0,           0.0,
            -sin(angle),    0.0, cos(angle),    0.0,
            0.0,            0.0, 0.0,           1.0
    );

    cv::Matx44f rotZMat(
            cos(angle), -sin(angle),    0.0,    0.0,
            sin(angle), cos(angle),     0.0,    0.0,
            0.0,        0.0,            1.0,    0.0,
            0.0,        0.0,            0.0,    1.0
    );

    cv::Matx44f scaleMat(
            scale,  0.0,    0.0,    0.0,
            0.0,    scale,  0.0,    0.0,
            0.0,    0.0,    scale,  0.0,
            0.0,    0.0,    0.0,    1.0
    );

    cv::Matx44f trans = scaleMat * rotXMat * rotYMat * rotZMat;
    glUniformMatrix4fv(uniform_mvp, 1, GL_FALSE, trans.val);

    glEnableVertexAttribArray(attribute_coord3d);
    // Describe our vertices array to OpenGL (it can't guess its format automatically)
    glBindBuffer(GL_ARRAY_BUFFER, vbo_cube_vertices);
    glVertexAttribPointer(attribute_coord3d, // attribute
            3,                 // number of elements per vertex, here (x,y,z)
            GL_FLOAT,          // the type of each element
            GL_FALSE,          // take our values as-is
            0,                 // no extra data between each position
            0                  // offset of first element
            );

    glEnableVertexAttribArray(attribute_v_color);
    glBindBuffer(GL_ARRAY_BUFFER, vbo_cube_colors);
    glVertexAttribPointer(attribute_v_color, // attribute
            3,                 // number of elements per vertex, here (R,G,B)
            GL_FLOAT,          // the type of each element
            GL_FALSE,          // take our values as-is
            0,                 // no extra data between each position
            0                  // offset of first element
            );

    /* Push each element in buffer_vertices to the vertex shader */
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ibo_cube_elements);
    int size;
    glGetBufferParameteriv(GL_ELEMENT_ARRAY_BUFFER, GL_BUFFER_SIZE, &size);
    glDrawElements(GL_TRIANGLES, size / sizeof(GLushort), GL_UNSIGNED_SHORT, 0);

    glDisableVertexAttribArray(attribute_coord3d);
    glDisableVertexAttribArray(attribute_v_color);
}

void glow_effect(const cv::UMat& src, cv::UMat& dst, const int ksize) {
    static cv::UMat resize;
    static cv::UMat blur;
    static cv::UMat dst16;

    cv::bitwise_not(src, dst);

    //Resize for some extra performance
    cv::resize(dst, resize, cv::Size(), 0.5, 0.5);
    //Cheap blur
    cv::boxFilter(resize, resize, -1, cv::Size(ksize, ksize), cv::Point(-1, -1), true,
            cv::BORDER_REPLICATE);
    //Back to original size
    cv::resize(resize, blur, src.size());

    //Multiply the src image with a blurred version of itself
    cv::multiply(dst, blur, dst16, 1, CV_16U);
    //Normalize and convert back to CV_8U
    cv::divide(dst16, cv::Scalar::all(255.0), dst, 1, CV_8U);

    cv::bitwise_not(dst, dst);
}

bool iteration() {
    using namespace cv::viz;
    //Render using OpenGL
    v4d->gl(render_scene);

    //Aquire the frame buffer for use by OpenCL
    v4d->fb([&](cv::UMat& frameBuffer) {
        //Glow effect (OpenCL)
        glow_effect(frameBuffer, frameBuffer, GLOW_KERNEL_SIZE);
    });

    v4d->write();

    updateFps(v4d, true);

    //If onscreen rendering is enabled it displays the framebuffer in the native window. Returns false if the window was closed.
    if (!v4d->display())
        return false;

    return true;
}

int main(int argc, char** argv) {
    using namespace cv::viz;

    if (!v4d->isOffscreen())
        v4d->setVisible(true);

    printSystemInfo();

#ifndef __EMSCRIPTEN__
    Sink sink = makeWriterSink(OUTPUT_FILENAME, cv::VideoWriter::fourcc('V', 'P', '9', '0'), FPS,
            cv::Size(WIDTH, HEIGHT));
    v4d->setSink(sink);
#endif
    v4d->gl(init_scene);
    v4d->run(iteration);

    return 0;
}
