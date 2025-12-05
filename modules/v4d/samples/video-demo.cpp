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

using namespace cv::v4d;

class VideoDemoPlan: public Plan {
public:
	using Plan::Plan;
	/* Demo Parameters */
	int glowKernelSize_ = 0;
private:
	struct Cache {
		cv::UMat up_;
		cv::UMat down_;
		cv::UMat blur_;
		cv::UMat dst16_;
	} cache_;

	/* OpenGL constants */
	constexpr static GLuint TRIANGLES_ = 12;
	constexpr static GLuint VERTICES_INDEX_ = 0;
	constexpr static GLuint COLOR_INDEX_ = 1;

    constexpr static float VERTICES_[24] = {
            // Front face
            0.5, 0.5, 0.5, -0.5, 0.5, 0.5, -0.5, -0.5, 0.5, 0.5, -0.5, 0.5,

            // Back face
            0.5, 0.5, -0.5, -0.5, 0.5, -0.5, -0.5, -0.5, -0.5, 0.5, -0.5, -0.5, };

    constexpr static float VERTEX_COLORS[24] = { 1.0, 0.4, 0.6, 1.0, 0.9, 0.2, 0.7, 0.3, 0.8, 0.5, 0.3, 1.0,

    0.2, 0.6, 1.0, 0.6, 1.0, 0.4, 0.6, 0.8, 0.8, 0.4, 0.8, 0.8, };

    constexpr static unsigned short TRIANGLE_INDICES_[36] = {
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
	/* OpenGL variables */
	GLuint vao_ = 0;
	GLuint shader_ = 0;
	GLuint uniform_transform_ = 0;

	static GLuint load_shader() {
	#if !defined(OPENCV_V4D_USE_ES3)
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

        unsigned int handles[3];
        cv::v4d::initShader(handles, vert.c_str(), frag.c_str(), "fragColor");
        return handles[0];
	}

	static void init_scene(GLuint& vao, GLuint& shader, GLuint& uniformTrans) {
	    glEnable (GL_DEPTH_TEST);

	    glGenVertexArrays(1, &vao);
	    glBindVertexArray(vao);

	    unsigned int triangles_ebo;
	    glGenBuffers(1, &triangles_ebo);
	    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, triangles_ebo);
	    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof TRIANGLE_INDICES_, TRIANGLE_INDICES_,
	            GL_STATIC_DRAW);

	    unsigned int verticies_vbo;
	    glGenBuffers(1, &verticies_vbo);
	    glBindBuffer(GL_ARRAY_BUFFER, verticies_vbo);
	    glBufferData(GL_ARRAY_BUFFER, sizeof VERTICES_, VERTICES_, GL_STATIC_DRAW);

	    glVertexAttribPointer(VERTICES_INDEX_, 3, GL_FLOAT, GL_FALSE, 0, NULL);
	    glEnableVertexAttribArray(VERTICES_INDEX_);

	    unsigned int colors_vbo;
	    glGenBuffers(1, &colors_vbo);
	    glBindBuffer(GL_ARRAY_BUFFER, colors_vbo);
	    glBufferData(GL_ARRAY_BUFFER, sizeof VERTEX_COLORS, VERTEX_COLORS, GL_STATIC_DRAW);

	    glVertexAttribPointer(COLOR_INDEX_, 3, GL_FLOAT, GL_FALSE, 0, NULL);
	    glEnableVertexAttribArray(COLOR_INDEX_);

	    glBindVertexArray(0);
	    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
	    glBindBuffer(GL_ARRAY_BUFFER, 0);

	    shader = load_shader();
	    uniformTrans = glGetUniformLocation(shader, "transform");
	}

	static void render_scene(GLuint& vao, GLuint& shader, GLuint& uniformTrans) {
	    glUseProgram(shader);

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
	    glUniformMatrix4fv(uniformTrans, 1, GL_FALSE, transform.val);
	    glBindVertexArray(vao);
	    glDrawElements(GL_TRIANGLES, TRIANGLES_ * 3, GL_UNSIGNED_SHORT, NULL);

	}

	static void glow_effect(const cv::UMat& src, cv::UMat& dst, const int ksize, Cache& cache) {
	    cv::bitwise_not(src, dst);

	    cv::resize(dst, cache.down_, cv::Size(), 0.5, 0.5);
	    cv::boxFilter(cache.down_, cache.blur_, -1, cv::Size(ksize, ksize), cv::Point(-1, -1), true,
	            cv::BORDER_REPLICATE);
	    cv::resize(cache.blur_, cache.up_, src.size());

	    cv::multiply(dst, cache.up_, cache.dst16_, 1, CV_16U);
	    cv::divide(cache.dst16_, cv::Scalar::all(255.0), dst, 1, CV_8U);

	    cv::bitwise_not(dst, dst);
	}
public:
	void setup(cv::Ptr<V4D> window) override {
		int diag = hypot(double(size().width), double(size().height));
		glowKernelSize_ = std::max(int(diag / 138 % 2 == 0 ? diag / 138 + 1 : diag / 138), 1);

		window->gl([](GLuint& vao, GLuint& shader, GLuint& uniformTrans) {
			init_scene(vao, shader, uniformTrans);
		}, vao_, shader_, uniform_transform_);
	}
	void infer(cv::Ptr<V4D> window) override {
		window->capture();

		window->gl([](GLuint& vao, GLuint& shader, GLuint& uniformTrans) {
			render_scene(vao, shader, uniformTrans);
		}, vao_, shader_, uniform_transform_);

		window->fb([](cv::UMat& framebuffer, const cv::Rect& viewport, int glowKernelSize, Cache& cache) {
			cv::UMat roi = framebuffer(viewport);
			glow_effect(roi, roi, glowKernelSize, cache);
		}, viewport(), glowKernelSize_, cache_);

		window->write();
	}
};

int main(int argc, char** argv) {
	if (argc != 2) {
        cerr << "Usage: video-demo <video-file>" << endl;
        exit(1);
    }

	cv::Ptr<VideoDemoPlan> plan = new VideoDemoPlan(cv::Size(1280,720));
    cv::Ptr<V4D> window = V4D::make(plan->size(), "Video Demo", NONE);

    auto src = makeCaptureSource(window, argv[1]);
    auto sink = makeWriterSink(window, "video-demo.mkv", src->fps(), plan->size());
    window->setSource(src);
    window->setSink(sink);

    window->run(plan);

    return 0;
}
