// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright Amir Hassan (kallaballa) <amir@viel-zu.org>

#include <opencv2/v4d/v4d.hpp>
#include <map>
#include <fstream>
#include <sstream>
#include <chrono>

//adapted from https://gitlab.com/wikibooks-opengl/modern-tutorials/-/blob/master/tut05_cube/cube.cpp

using namespace cv::v4d;
class ManyCubesDemoPlan : public Plan {
public:
	/* Demo Parameters */
	constexpr static size_t MAX_NUM_CUBES_ = 100;
	constexpr static std::string subtitlesFile_ = "sub.txt";
	size_t numCubes_ = 0;
	int glowKernelSize_;

	/* OpenGL constants and variables */
	constexpr static GLuint TRIANGLES_ = 12;
	constexpr static GLuint VERTICES_INDEX_ = 0;
	constexpr static GLuint COLORS_INDEX_ = 1;

	//Cube vertices, colors and indices
	constexpr static float VERTICES_[24] = {
		// Front face
        0.5, 0.5, 0.5, -0.5, 0.5, 0.5, -0.5, -0.5, 0.5, 0.5, -0.5, 0.5,
        // Back face
        0.5, 0.5, -0.5, -0.5, 0.5, -0.5, -0.5, -0.5, -0.5, 0.5, -0.5, -0.5
	};

	constexpr static float VERTEX_COLORS_[24] = {
		0.125, 0.125, 0.125, 1.00, 1.00, 1.00, 0.125, 0.125, 0.125, 1.00, 1.00, 1.00,
		0.125, 0.125, 0.125, 1.00, 1.00, 1.00, 0.125, 0.125, 0.125, 1.00, 1.00, 1.00
	};

//	constexpr static float VERTEX_COLORS_[24] = {
//			1.0, 0.4, 0.6, 1.0, 0.9, 0.2, 0.7, 0.3, 0.8, 0.5, 0.3, 1.0,
//			0.2, 0.6, 1.0, 0.6, 1.0, 0.4, 0.6, 0.8, 0.8, 0.4, 0.8, 0.8
//	};

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
        5, 1, 0, 0, 4, 5
	};
private:
	struct Cache {
	    cv::UMat down_;
	    cv::UMat up_;
	    cv::UMat blur_;
	    cv::UMat dst16_;
	    cv::UMat invert_;
	    cv::UMat bgr_;
	    cv::UMat gray_;
	} cache_;
	GLuint vao_[MAX_NUM_CUBES_];
	GLuint shaderProgram_[MAX_NUM_CUBES_];
	GLuint uniformTransform_[MAX_NUM_CUBES_];
	float start_ = 0;
	std::map<long, std::pair<long, std::string>> subtitles_;
	cv::UMat cubes_;
	cv::UMat text_;

	//Simple transform & pass-through shaders
	static GLuint load_shader() {
		//Shader versions "330" and "300 es" are very similar.
		//If you are careful you can write the same code for both versions.
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

	    //Initialize the shaders and returns the program
	    return cv::v4d::initShader(vert.c_str(), frag.c_str(), "fragColor");
	}

	//Initializes objects, buffers, shaders and uniforms
	static void init_scene(const cv::Size& sz, GLuint& vao, GLuint& shaderProgram, GLuint& uniformTransform) {
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
	    glBufferData(GL_ARRAY_BUFFER, sizeof VERTEX_COLORS_, VERTEX_COLORS_, GL_STATIC_DRAW);

	    glVertexAttribPointer(COLORS_INDEX_, 3, GL_FLOAT, GL_FALSE, 0, NULL);
	    glEnableVertexAttribArray(COLORS_INDEX_);

	    glBindVertexArray(0);
	    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
	    glBindBuffer(GL_ARRAY_BUFFER, 0);

	    shaderProgram = load_shader();
	    uniformTransform = glGetUniformLocation(shaderProgram, "transform");
	    glViewport(0,0, sz.width, sz.height);
	}

	//Renders a rotating rainbow-colored cube on a blueish background
	static void render_scene(const cv::Size& sz, const double& x, const double& y, const double& angleMod, GLuint& vao, GLuint& shaderProgram, GLuint& uniformTransform) {
	    glViewport(0,0, sz.width, sz.height);
		//Use the prepared shader program
	    glUseProgram(shaderProgram);

	    //Scale and rotate the cube depending on the current time.
	    float angle =  fmod(double(cv::getTickCount()) / double(cv::getTickFrequency()) + angleMod, 2 * M_PI);
	    double scale = 0.25;
	    cv::Matx44f scaleMat(
	            scale, 0.0, 0.0, 0.0,
	            0.0, scale, 0.0, 0.0,
	            0.0, 0.0, scale, 0.0,
	            0.0, 0.0, 0.0, 1.0);

	    cv::Matx44f rotXMat(
	            1.0, 0.0, 0.0, 0.0,
	            0.0, cos(angle), -sin(angle), 0.0,
	            0.0, sin(angle), cos(angle), 0.0,
	            0.0, 0.0, 0.0, 1.0);

	    cv::Matx44f rotYMat(
	            cos(angle), 0.0, sin(angle), 0.0,
	            0.0, 1.0, 0.0, 0.0,
	            -sin(angle), 0.0,cos(angle), 0.0,
	            0.0, 0.0, 0.0, 1.0);

	    cv::Matx44f rotZMat(
	            cos(angle), -sin(angle), 0.0, 0.0,
	            sin(angle), cos(angle), 0.0, 0.0,
	            0.0, 0.0, 1.0, 0.0,
	            0.0, 0.0, 0.0, 1.0);

	    cv::Matx44f translateMat(
	            1.0, 0.0, 0.0, 0.0,
	            0.0, 1.0, 0.0, 0.0,
	            0.0, 0.0, 1.0, 0.0,
	              x,   y, 0.0, 1.0);

	    //calculate the transform
	    cv::Matx44f transform = scaleMat * rotXMat * rotYMat * rotZMat * translateMat;
	    //set the corresponding uniform
	    glUniformMatrix4fv(uniformTransform, 1, GL_FALSE, transform.val);
	    //Bind our vertex array
	    glBindVertexArray(vao);
	    //Draw
	    glDrawElements(GL_TRIANGLES, TRIANGLES_ * 3, GL_UNSIGNED_SHORT, NULL);
	}

#ifndef __EMSCRIPTEN__
	//applies a glow effect to an image
	static void glow_effect(const cv::UMat& src, cv::UMat& dst, const int ksize, Cache& cache) {
	    cv::bitwise_not(src, dst);

	    //Resize for some extra performance
	    cv::resize(dst, cache.down_, cv::Size(), 0.5, 0.5);
	    //Cheap blur
	    cv::boxFilter(cache.down_, cache.blur_, -1, cv::Size(ksize, ksize), cv::Point(-1, -1), true,
	            cv::BORDER_REPLICATE);
	    //Back to original size
	    cv::resize(cache.blur_, cache.up_, src.size());

	    //Multiply the src image with a blurred version of itself
	    cv::multiply(dst, cache.up_, cache.dst16_, 1, CV_16U);
	    //Normalize and convert back to CV_8U
	    cv::divide(cache.dst16_, cv::Scalar::all(255.0), dst, 1, CV_8U);

	    cv::bitwise_not(dst, dst);
	}
#endif
public:
	ManyCubesDemoPlan(cv::Size sz) : Plan(sz) {
		int diag = hypot(double(size().width), double(size().height));
		glowKernelSize_ = 3;
	}

	void setup(cv::Ptr<V4D> window) override {
		window->parallel([](float& start, const std::string& subtitlesFile, std::map<long, std::pair<long, std::string>>& subtitles){
			start = cv::getTickCount() / cv::getTickFrequency();

			std::ifstream ifs(subtitlesFile);
			std::string line;
			std::string hours;
			std::string minutes;
			std::string seconds;
			std::string millis;
			std::string text;
			size_t cnt = 0;
			while(ifs) {
				if(!getline(ifs, line))
						break;
				std::istringstream is(line);
				CV_Assert(getline(is, hours, ':'));
				CV_Assert(getline(is, minutes, ':'));
				CV_Assert(getline(is, seconds, ','));
				CV_Assert(getline(is, millis, ' '));
				long timepointMillis = ((stol(hours) * 60 * 60) + (stol(minutes) * 60) + stol(seconds)) * 1000 + stol(millis);
				getline(ifs, text);
//				cerr << timepointMillis << " = " << text << endl;
				long prev = 0;
				if(!subtitles.empty())
					std::prev(subtitles.end())->second.first = timepointMillis + 200;
				subtitles[5052 * cnt] = { 0, text };
				++cnt;
			}
			std::prev(subtitles.end())->second.first = std::prev(subtitles.end())->first + 5000;

		}, start_, subtitlesFile_, subtitles_);

		for(size_t i = 0; i < MAX_NUM_CUBES_; ++i) {
			window->gl(i, [](const size_t& ctxIdx, const cv::Size& sz, GLuint& vao, GLuint& shader, GLuint& uniformTrans){
				CV_UNUSED(ctxIdx);
				init_scene(sz, vao, shader, uniformTrans);
			}, size(), vao_[i], shaderProgram_[i], uniformTransform_[i]);
		}
	}

	void infer(cv::Ptr<V4D> window) override {
		window->gl([](){
			//Clear the background
			glClearColor(0.0, 0.0, 0.0, 1);
			glClear(GL_COLOR_BUFFER_BIT);
		});

		window->parallel([](float& start, size_t& numCubes){
			float t = cv::getTickCount() / cv::getTickFrequency() - start;
			numCubes = std::max(size_t(round(pow(t / 182.0f,2.0f) * MAX_NUM_CUBES_)), size_t(1));
			if(numCubes == MAX_NUM_CUBES_)
				exit(0);
		}, start_, numCubes_);

		//Render using multiple OpenGL contexts
		for(size_t i = 0; i < MAX_NUM_CUBES_; ++i) {
			window->gl(i, [](const int32_t& ctxIdx, const cv::Size& sz, const size_t& numCubes, GLuint& vao, GLuint& shader, GLuint& uniformTrans){
				double x = sin((double(ctxIdx) / numCubes) * 2 * M_PI) / 1.5;
				double y = cos((double(ctxIdx) / numCubes) * 2 * M_PI) / 1.5;
				double angle = sin((double(ctxIdx) / numCubes) * 2 * M_PI);
				if(ctxIdx < numCubes)
					render_scene(sz, x, y, angle, vao, shader, uniformTrans);
			}, size(), numCubes_, vao_[i], shaderProgram_[i], uniformTransform_[i]);
		}

		window->fb([](const cv::UMat& framebuffer, cv::UMat& cubes) {
			framebuffer.copyTo(cubes);
		}, cubes_);

		window->nvg([](const cv::Size& sz, float& start, std::map<long, std::pair<long, std::string>>& subtitles){
			if(subtitles.empty())
				return;
			float t = (cv::getTickCount() / cv::getTickFrequency() - start) * 1000;
			long first = ((*subtitles.begin()).first);
			long second = 0;
			if(subtitles.size() > 1)
				second = (std::next(subtitles.begin())->first);

//			cerr << "t:" << t << " f:" << first << " s:" << second << endl;
			if(second > 0 && t > second) {
				subtitles.erase(subtitles.begin());
				return;
			}

			if(t >= first) {
				using namespace cv::v4d::nvg;
				std::string txt = (*subtitles.begin()).second.second;
//				cerr << "DRAW: " << txt << endl;
				clear();
				fontSize(38);
				fontFace("sans-bold");
				fillColor(cv::Scalar::all(255));
				textAlign(NVG_ALIGN_CENTER | NVG_ALIGN_MIDDLE);
				text(sz.width / 2.0, sz.height / 2.0, txt.c_str(), txt.c_str() + txt.size());
			}
		}, size(), start_, subtitles_);

		//Aquire the frame buffer for use by OpenCV
		window->fb([](cv::UMat& framebuffer, cv::UMat& cubes, cv::UMat& text, int glowKernelSize, Cache& cache) {
			glow_effect(framebuffer, text, glowKernelSize, cache);
			cv::bitwise_not(text, cache.invert_);
			cv::add(cubes, text, framebuffer);
			glow_effect(text, text, glowKernelSize * 2, cache);
			cv::add(framebuffer, text, framebuffer);
			glow_effect(text, text, glowKernelSize * 4, cache);
			cv::add(framebuffer, text, framebuffer);
			cvtColor(cache.invert_, cache.gray_, cv::COLOR_BGRA2GRAY);
			cv::threshold(cache.gray_, cache.gray_, 254, 255, cv::THRESH_BINARY);
			cache.invert_.setTo(cv::Scalar::all(0), cache.gray_);
			cv::bitwise_xor(framebuffer, cache.invert_, framebuffer);
			glow_effect(framebuffer, framebuffer, glowKernelSize * 8, cache);
		}, cubes_, text_, glowKernelSize_, cache_);

		window->write();
	}
};

int main() {
	cv::Ptr<ManyCubesDemoPlan> plan = new ManyCubesDemoPlan(cv::Size(1920, 1080));
    cv::Ptr<V4D> window = V4D::make(plan->size(), "Many Cubes Demo", NANOVG);

	constexpr double FPS = 47;
	constexpr const char* OUTPUT_FILENAME = "many_cubes-demo.mkv";
    //Creates a writer sink (which might be hardware accelerated)
    auto sink = makeWriterSink(window, OUTPUT_FILENAME, FPS, plan->size());
    window->setPrintFPS(true);
    window->setShowFPS(false);
    window->setShowTracking(false);
    window->setSink(sink);
    window->run(plan, 0);

    return 0;
}
