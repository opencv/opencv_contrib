// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright Amir Hassan (kallaballa) <amir@viel-zu.org>

/*
 * Based on cube-demo. Only differs in two points:
 * - Uses a source to read a video.
 * - Doesn't clear the background so the cube is rendered on top of the video.
 */

#include <opencv2/v4d/v4d.hpp>

using std::cerr;
using std::endl;

/* Demo Parameters */
#ifndef __EMSCRIPTEN__
constexpr long unsigned int WIDTH = 1280;
constexpr long unsigned int HEIGHT = 720;
#else
constexpr long unsigned int WIDTH = 960;
constexpr long unsigned int HEIGHT = 960;
#endif
constexpr bool OFFSCREEN = false;
const unsigned long DIAG = hypot(double(WIDTH), double(HEIGHT));
const int glow_kernel_size = std::max(int(DIAG / 138 % 2 == 0 ? DIAG / 138 + 1 : DIAG / 138), 1);
#ifndef __EMSCRIPTEN__
constexpr double FPS = 60;
constexpr const char* OUTPUT_FILENAME = "video-demo.mkv";
#endif

/* OpenGL contants and variables */
static const unsigned int triangles = 12;
static const unsigned int vertices_index = 0;
static const unsigned int colors_index = 1;
static thread_local unsigned int shader_program;
static thread_local unsigned int vao;
static thread_local unsigned int uniform_transform;

static GLuint load_shader() {
#if !defined(__EMSCRIPTEN__) && !defined(OPENCV_V4D_USE_ES3)
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

    return cv::v4d::initShader(vert.c_str(), frag.c_str(), "fragColor");
}

static void init_scene() {
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

    glBindVertexArray(0);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    shader_program = load_shader();
    uniform_transform = glGetUniformLocation(shader_program, "transform");
}

static void render_scene() {
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
class VideoDemoPlan: public Plan {
cv::UMat frame_;
public:
	void setup(cv::Ptr<V4D> window) override {
		window->gl([]() {
			init_scene();
		});
	}
	void infer(cv::Ptr<V4D> window) override {
		window->capture(frame_);

		window->gl([]() {
			render_scene();
		});

		window->fb([](cv::UMat &framebuffer, cv::UMat& f) {
			glow_effect(framebuffer, framebuffer, glow_kernel_size);
			framebuffer.copyTo(f);
		}, frame_);

		//Ignored in WebAssmebly builds because there is no sink set.
		window->write(frame_);
	}
};

#ifndef __EMSCRIPTEN__
int main(int argc, char** argv) {
    if (argc != 2) {
        cerr << "Usage: video-demo <video-file>" << endl;
        exit(1);
    }
#else
int main() {
#endif
    using namespace cv::v4d;
    cv::Ptr<V4D> window = V4D::make(WIDTH, HEIGHT, "Video Demo", NONE, OFFSCREEN, false, 0);
    window->printSystemInfo();

#ifndef __EMSCRIPTEN__
    //Creates a source from a file or a device
    auto src = makeCaptureSource(window, argv[1]);
    window->setSource(src);

    auto sink = makeWriterSink(window, OUTPUT_FILENAME, src->fps(), cv::Size(WIDTH, HEIGHT));
    window->setSink(sink);
#else
    //Creates a webcam source is available
    auto src = makeCaptureSource(WIDTH, HEIGHT, window);
    window->setSource(src);
#endif

    window->run<VideoDemoPlan>(0);

    return 0;
}
