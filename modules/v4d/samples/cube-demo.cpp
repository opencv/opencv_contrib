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

const int GLOW_KERNEL_SIZE = std::max(int(DIAG / 138 % 2 == 0 ? DIAG / 138 + 1 : DIAG / 138), 1);

using std::cerr;
using std::endl;

const unsigned int triangles = 12;
const unsigned int vertices_index = 0;
const unsigned int colors_index = 1;
unsigned int shader_program;
unsigned int vao;
unsigned int uniform_transform;

cv::Ptr<cv::viz::V4D> v4d = cv::viz::V4D::make(cv::Size(WIDTH, HEIGHT), cv::Size(WIDTH, HEIGHT),
        OFFSCREEN, "Cube Demo");

GLuint load_shader() {
#ifndef OPENCV_V4D_USE_ES3
    const string shaderVersion = "330";
#else
    const string shaderVersion = "300 es";
#endif

    const string vert =
            "    #version " + shaderVersion
                    + R"(
    precision lowp float;
    layout(location = 0) in vec3 pos;
    layout(location = 1) in vec3 vertex_color;
    
    uniform mat4 transform;
    
    out vec3 color;
    void main() {
      gl_Position = transform * vec4(pos, 1.0);
      color = vertex_color;
    }
)";

    const string frag =
            "    #version " + shaderVersion
                    + R"(
    precision lowp float;
    in vec3 color;
    
    out vec4 frag_color;
    
    void main() {
      frag_color = vec4(color, 1.0);
    }
)";

    return cv::viz::init_shader(vert.c_str(), frag.c_str(), "fragColor");
}

void init_scene(const cv::Size& sz) {
    glEnable (GL_DEPTH_TEST);

    float vertices[] = {
    // Front face
            0.5, 0.5, 0.5, -0.5, 0.5, 0.5, -0.5, -0.5, 0.5, 0.5, -0.5, 0.5,

            // Back face
            0.5, 0.5, -0.5, -0.5, 0.5, -0.5, -0.5, -0.5, -0.5, 0.5, -0.5, -0.5, };

    float vertex_colors[] = { 1.0, 0.4, 0.6, 1.0, 0.9, 0.2, 0.7, 0.3, 0.8, 0.5, 0.3, 1.0,

    0.2, 0.6, 1.0, 0.6, 1.0, 0.4, 0.6, 0.8, 0.8, 0.4, 0.8, 0.8, };

    unsigned short triangle_indices[] = {
    // Front
            0, 1, 2, 2, 3, 0,

            // Right
            0, 3, 7, 7, 4, 0,

            // Bottom
            2, 6, 7, 7, 3, 2,

            // Left
            1, 5, 6, 6, 2, 1,

            // Back
            4, 7, 6, 6, 5, 4,

            // Top
            5, 1, 0, 0, 4, 5, };

    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);

    unsigned int triangles_ebo;
    glGenBuffers(1, &triangles_ebo);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, triangles_ebo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof triangle_indices, triangle_indices,
            GL_STATIC_DRAW);

    unsigned int verticies_vbo;
    glGenBuffers(1, &verticies_vbo);
    glBindBuffer(GL_ARRAY_BUFFER, verticies_vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof vertices, vertices, GL_STATIC_DRAW);

    glVertexAttribPointer(vertices_index, 3, GL_FLOAT, GL_FALSE, 0, NULL);
    glEnableVertexAttribArray(vertices_index);

    unsigned int colors_vbo;
    glGenBuffers(1, &colors_vbo);
    glBindBuffer(GL_ARRAY_BUFFER, colors_vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof vertex_colors, vertex_colors, GL_STATIC_DRAW);

    glVertexAttribPointer(colors_index, 3, GL_FLOAT, GL_FALSE, 0, NULL);
    glEnableVertexAttribArray(colors_index);

    // Unbind to prevent accidental modification
    glBindVertexArray(0);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    shader_program = load_shader();
    uniform_transform = glGetUniformLocation(shader_program, "transform");
}

void render_scene(const cv::Size& sz) {
    glClearColor(0.1, 0.12, 0.2, 1);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glUseProgram(shader_program);

    float angle = fmod(double(cv::getTickCount()) / double(cv::getTickFrequency()), 2 * M_PI);
    float scale = 0.25;

    cv::Matx44f scaleMat(scale, 0.0, 0.0, 0.0, 0.0, scale, 0.0, 0.0, 0.0, 0.0, scale, 0.0, 0.0, 0.0,
            0.0, 1.0);

    cv::Matx44f rotXMat(1.0, 0.0, 0.0, 0.0, 0.0, cos(angle), -sin(angle), 0.0, 0.0, sin(angle),
            cos(angle), 0.0, 0.0, 0.0, 0.0, 1.0);

    cv::Matx44f rotYMat(cos(angle), 0.0, sin(angle), 0.0, 0.0, 1.0, 0.0, 0.0, -sin(angle), 0.0,
            cos(angle), 0.0, 0.0, 0.0, 0.0, 1.0);

    cv::Matx44f rotZMat(cos(angle), -sin(angle), 0.0, 0.0, sin(angle), cos(angle), 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0);

    cv::Matx44f transform = scaleMat * rotXMat * rotYMat * rotZMat;
    glUniformMatrix4fv(uniform_transform, 1, GL_FALSE, transform.val);
    glBindVertexArray(vao);
    glDrawElements(GL_TRIANGLES, triangles * 3, GL_UNSIGNED_SHORT, NULL);

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

//If we have OpenCL and maybe even CL-GL sharing then this is faster than the glow shader. Without OpenCL this is very slow.
#ifndef __EMSCRIPTEN__
    //Aquire the frame buffer for use by OpenCL
    v4d->fb([&](cv::UMat& frameBuffer) {
        //Glow effect (OpenCL)
        glow_effect(frameBuffer, frameBuffer, GLOW_KERNEL_SIZE);
    });
#endif

    updateFps(v4d, true);

    v4d->write();

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
