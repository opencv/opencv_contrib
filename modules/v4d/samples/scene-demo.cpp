// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright Amir Hassan (kallaballa) <amir@viel-zu.org>

#include <opencv2/v4d/v4d.hpp>
#include <opencv2/v4d/scene.hpp>

using namespace cv::v4d;

class SceneDemoPlan : public Plan {
	const string filename_ = "gear.glb";
	gl::Scene scene_;
	gl::Scene pcaScene_;
	std::vector<cv::Point3f> pointCloud_;

	struct Transform {
		cv::Vec3f translate_;
		cv::Vec3f rotation_;
		cv::Vec3f scale_;
	    cv::Matx44f projection_;
		cv::Matx44f view_;
		cv::Matx44f model_;
	} transform_;
public:
	using Plan::Plan;

	void setup(cv::Ptr<V4D> window) override {
		window->gl([](gl::Scene& scene, const string& filename){
			CV_Assert(scene.load(filename));
		}, scene_, filename_);
	}

	void infer(cv::Ptr<V4D> window) override {
		window->gl(0,[](const int32_t& ctx, const cv::Rect& viewport, gl::Scene& scene, std::vector<cv::Point3f>& pointCloud, Transform& transform){
			glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
			double progress = (cv::getTickCount() / cv::getTickFrequency()) / 5.0;
			float angle =  fmod(double(cv::getTickCount()) / double(cv::getTickFrequency()), 2 * M_PI);
			int m = int(progress) % 3;

			float scale = scene.autoScale();
		    cv::Vec3f center = scene.autoCenter();
		    transform.rotation_ = {0, angle, 0};
		    transform.translate_ = {-center[0], -center[1], -center[2]};
		    transform.scale_ = { scale, scale, scale };
		    transform.projection_ = gl::perspective(45.0f * (CV_PI/180), float(viewport.width) / viewport.height, 0.1f, 100.0f);
		    transform.view_ = gl::lookAt(cv::Vec3f(0.0f, 0.0f, 3.0f), cv::Vec3f(0.0f, 0.0f, 0.0f), cv::Vec3f(0.0f, 1.0f, 0.0f));
		    transform.model_ = gl::modelView(transform.translate_, transform.rotation_, transform.scale_);

		    scene.setMode(static_cast<gl::Scene::RenderMode>(m));
			scene.render(viewport, transform.projection_, transform.view_, transform.model_);
		}, viewport(), scene_, pointCloud_, transform_);
		window->write();
	}
};

int main() {
    cv::Ptr<V4D> window = V4D::make(cv::Size(1280, 720), "Scene Demo", IMGUI);
	cv::Ptr<SceneDemoPlan> plan = new SceneDemoPlan(cv::Size(1280, 720));

    //Creates a writer sink (which might be hardware accelerated)
    auto sink = makeWriterSink(window, "scene-demo.mkv", 60, plan->size());
    window->setSink(sink);
    window->run(plan, 3);

    return 0;
}
