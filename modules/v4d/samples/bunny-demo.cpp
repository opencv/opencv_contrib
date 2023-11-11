// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright Amir Hassan (kallaballa) <amir@viel-zu.org>

#include <opencv2/v4d/v4d.hpp>
//adapted from https://gitlab.com/wikibooks-opengl/modern-tutorials/-/blob/master/tut05_cube/cube.cpp
#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

using namespace cv::v4d;

class BunnyDemoPlan : public Plan {
public:
	using Plan::Plan;

	/* Demo Parameters */
	int glowKernelSize_ = 0;
private:
	struct Cache {
	    cv::UMat down_;
	    cv::UMat up_;
	    cv::UMat blur_;
	    cv::UMat dst16_;
	} cache_;
	GLuint vao_ = 0;
	GLuint shaderProgram_ = 0;
	GLuint uniProjection_= 0;
	GLuint uniView_= 0;
	GLuint uniModel_= 0;
    glm::mat4 projection_;
    glm::mat4 view_;
    glm::mat4 model_;
    inline static cv::Ptr<Assimp::Importer> importer_ = cv::Ptr<Assimp::Importer>(nullptr);
    inline static cv::Ptr<const aiScene> scene_ = cv::Ptr<const aiScene>(nullptr);
    inline static std::once_flag flag_;

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
    layout(location = 0) in vec3 aPos;
    layout(location = 1) in vec3 aNormal;
    out vec3 FragPos;
    out vec3 Normal;
    uniform mat4 model;
    uniform mat4 view;
    uniform mat4 projection;
    void main() {
        gl_Position = projection * view * model * vec4(aPos, 1.0);
        FragPos = vec3(model * vec4(aPos, 1.0));
        Normal = mat3(transpose(inverse(model))) * aNormal;
    }
	)";

	    const string frag =
	            "    #version " + shaderVersion
	                    + R"(
    precision mediump float;
    in vec3 FragPos;
    in vec3 Normal;
    out vec4 FragColor;
    uniform vec3 lightPos;
    uniform vec3 viewPos;
    void main() {
        vec3 ambient = 0.1 * vec3(1.0, 1.0, 1.0);
        vec3 lightDir = normalize(lightPos - FragPos);
        vec3 viewDir = normalize(viewPos - FragPos);
        float diff = max(dot(Normal, lightDir), 0.0);
        vec3 diffuse = diff * vec3(1.0, 1.0, 1.0);
        vec3 specular = vec3(0.0);
        float shininess = 32.0;
        FragColor = vec4(ambient + diffuse + specular, 1.0);
    }
	)";

	    //Initialize the shaders and returns the program
	    return cv::v4d::initShader(vert.c_str(), frag.c_str(), "fragColor");
	}

	//Initializes objects, buffers, shaders and uniforms
	static void init_scene(const cv::Size& sz, GLuint& vao, GLuint& shaderProgram, GLuint& uniProjection, GLuint& uniView, GLuint& uniModel, cv::Ptr<const aiScene>& scene) {
//	    glEnable (GL_DEPTH_TEST);
		    // Load and compile shaders, create program, and set uniforms
		    shaderProgram = load_shader();

		    GLuint VBO, EBO;
		    glGenVertexArrays(1, &vao);
		    glGenBuffers(1, &VBO);
		    glGenBuffers(1, &EBO);

		    glBindVertexArray(vao);

		    glBindBuffer(GL_ARRAY_BUFFER, VBO);
		    glBufferData(GL_ARRAY_BUFFER, sizeof(aiVector3D) * scene->mMeshes[0]->mNumVertices, scene->mMeshes[0]->mVertices, GL_STATIC_DRAW);

		    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
		    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(unsigned int) * scene->mMeshes[0]->mNumFaces * 3, reinterpret_cast<void*>(scene->mMeshes[0]->mFaces[0].mIndices), GL_STATIC_DRAW);

		    glEnableVertexAttribArray(0);
		    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), 0);

		    // Set up the projection and view matrices
		    glm::mat4 projection = glm::perspective(glm::radians(45.0f), 800.0f / 600.0f, 0.1f, 100.0f);
		    glm::mat4 view = glm::lookAt(glm::vec3(0.0f, 0.0f, 3.0f), glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 1.0f, 0.0f));

		    uniProjection = glGetUniformLocation(shaderProgram, "projection");
		    uniView = glGetUniformLocation(shaderProgram, "view");
		    uniModel = glGetUniformLocation(shaderProgram, "model");
	    glViewport(0,0, sz.width, sz.height);
	}

	//Renders a rotating rainbow-colored cube on a blueish background
	static void render_scene(GLuint &vao, GLuint &shaderProgram,
			GLuint &uniProjection, GLuint& uniView, GLuint& uniModel, const aiScene* scene) {
		//Clear the background
		glClearColor(0.2, 0.24, 0.4, 1);
		glClear(GL_COLOR_BUFFER_BIT);

		//Use the prepared shader program
		glUseProgram(shaderProgram);
		glm::mat4 model = glm::mat4(1.0f);
		    model = glm::rotate(model, (float)glfwGetTime(), glm::vec3(0.0f, 1.0f, 0.0f));
		    model = glm::scale(model, glm::vec3(5.0f)); // Adjust the scale

	    glm::mat4 projection = glm::perspective(glm::radians(45.0f), 800.0f / 600.0f, 0.1f, 100.0f);
	    glm::mat4 view = glm::lookAt(glm::vec3(0.0f, 0.0f, 3.0f), glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 1.0f, 0.0f));
	    glUniformMatrix4fv(uniProjection, 1, GL_FALSE, glm::value_ptr(projection));
	    glUniformMatrix4fv(uniView, 1, GL_FALSE, glm::value_ptr(view));
        glUniformMatrix4fv(uniModel, 1, GL_FALSE, glm::value_ptr(model));

		//Bind the prepared vertex array object
		glBindVertexArray(vao);
		//Draw
        glDrawElements(GL_TRIANGLES, scene->mMeshes[0]->mNumFaces * 3, GL_UNSIGNED_INT, 0);
	}

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

public:
	BunnyDemoPlan(const cv::Rect& viewport) : Plan(viewport) {
	}

	BunnyDemoPlan(const cv::Size& sz) : BunnyDemoPlan(cv::Rect(0, 0, sz.width, sz.height)) {
	}

	void setup(cv::Ptr<V4D> window) override {
		int diag = hypot(double(size().width), double(size().height));
		glowKernelSize_ = std::max(int(diag / 138 % 2 == 0 ? diag / 138 + 1 : diag / 138), 1);

		window->gl(0, [](const int32_t& ctxIdx, cv::Ptr<const aiScene>& scene, cv::Ptr<Assimp::Importer>& importer){
			std::call_once(flag_, [&](){
				importer.reset(new Assimp::Importer(), [](Assimp::Importer* s){});
				scene.reset(importer_->ReadFile("bunny.obj", aiProcess_Triangulate | aiProcess_GenNormals), [](const aiScene* s){});
				CV_Assert(scene && ((scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE) == 0) && scene->mRootNode && scene->mMeshes);
			});
			CV_Assert(scene && ((scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE) == 0) && scene->mRootNode && scene->mMeshes);

		}, scene_, importer_);

		window->gl([](const cv::Size& sz, GLuint& vao, GLuint& shaderProgram, GLuint& uniProjection, GLuint& uniView, GLuint& uniModel, cv::Ptr<const aiScene>& scene){
			CV_Assert(scene && ((scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE) == 0) && scene->mRootNode && scene->mMeshes);
			init_scene(sz, vao, shaderProgram, uniProjection, uniView, uniModel, scene);
		}, size(), vao_, shaderProgram_, uniProjection_, uniView_, uniModel_, scene_);

	}

	void infer(cv::Ptr<V4D> window) override {
//		window->gl([](){
//			//Clear the background
//			glClearColor(0.2f, 0.24f, 0.4f, 1.0f);
//			glClear(GL_COLOR_BUFFER_BIT);
//		});

		window->gl([](GLuint& vao, GLuint& shaderProgram, GLuint& uniProjection, GLuint& uniView, GLuint& uniModel, cv::Ptr<const aiScene>& scene){
			render_scene(vao, shaderProgram, uniProjection, uniView, uniModel, scene);
		}, vao_, shaderProgram_, uniProjection_, uniView_, uniModel_, scene_);

//		//Aquire the frame buffer for use by OpenCV
//		window->fb([](cv::UMat& framebuffer, const cv::Rect& viewport, int glowKernelSize, Cache& cache) {
//			cv::UMat roi = framebuffer(viewport);
//			glow_effect(roi, roi, glowKernelSize, cache);
//		}, viewport(), glowKernelSize_, cache_);

		window->write();
	}
};

int main() {

    cv::Ptr<V4D> window = V4D::make(cv::Size(1280, 720), "Bunny Demo", NONE);
	cv::Ptr<BunnyDemoPlan> plan = new BunnyDemoPlan(cv::Size(1280, 720));

    //Creates a writer sink (which might be hardware accelerated)
    auto sink = makeWriterSink(window, "bunny-demo.mkv", 60, plan->size());
    window->setSink(sink);
    window->run(plan, 0);

    return 0;
}
