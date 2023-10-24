// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright Amir Hassan (kallaballa) <amir@viel-zu.org>

int v4d_cube_main();
int v4d_many_cubes_main();
int v4d_video_main(int argc, char **argv);
int v4d_nanovg_main(int argc, char **argv);
int v4d_shader_main(int argc, char **argv);
int v4d_font_main();
int v4d_pedestrian_main(int argc, char **argv);
int v4d_optflow_main(int argc, char **argv);
int v4d_beauty_main(int argc, char **argv);
#define main v4d_cube_main
#include "cube-demo.cpp"
#undef main
#define main v4d_many_cubes_main
#include "many_cubes-demo.cpp"
#undef main
#define main v4d_video_main
#include "video-demo.cpp"
#undef main
#define main v4d_nanovg_main
#include "nanovg-demo.cpp"
#undef main
#define main v4d_shader_main
#include "shader-demo.cpp"
#undef main
#define main v4d_font_main
#include "font-demo.cpp"
#undef main
#define main v4d_pedestrian_main
#include "pedestrian-demo.cpp"
#undef main
#define main v4d_optflow_main
#include "optflow-demo.cpp"
#undef main
#define main v4d_beauty_main
#include "beauty-demo.cpp"
#undef main


class MontageDemoPlan : public Plan {
	const cv::Size tileSz_ = cv::Size(640, 360);
	CubeDemoPlan cubePlan_ = CubeDemoPlan(tileSz_);
	ManyCubesDemoPlan manyCubesPlan_ = ManyCubesDemoPlan(tileSz_);
	VideoDemoPlan videoPlan_ = VideoDemoPlan(tileSz_);
	NanoVGDemoPlan nanovgPlan_ = NanoVGDemoPlan(tileSz_);
	ShaderDemoPlan shaderPlan_ = ShaderDemoPlan(tileSz_);
	FontDemoPlan fontPlan_ = FontDemoPlan(tileSz_);
	PedestrianDemoPlan pedestrianPlan_ = PedestrianDemoPlan(tileSz_);
	BeautyDemoPlan beautyPlan_ = BeautyDemoPlan(tileSz_);
	OptflowDemoPlan optflowPlan_ = OptflowDemoPlan(tileSz_);
	cv::Size_<float> scale_;
	struct Frames {
		cv::UMat cube_;
		cv::UMat many_cubes_;
		cv::UMat video_;
		cv::UMat nanovg_;
		cv::UMat shader_;
		cv::UMat font_;
		cv::UMat pedestrian_;
		cv::UMat beauty_;
		cv::UMat optflow_;
		cv::UMat captured;
	} frames_;
public:
	MontageDemoPlan(const cv::Size& sz) : Plan(sz) {
		scale_ = cv::Size_<float>(float(size().width) / tileSz_.width, float(size().height) / tileSz_.height);
	}

	virtual void setup(cv::Ptr<V4D> window) override {
		cubePlan_.setup(window);
		manyCubesPlan_.setup(window);
		videoPlan_.setup(window);
		nanovgPlan_.setup(window);
		shaderPlan_.setup(window);
		fontPlan_.setup(window);
		pedestrianPlan_.setup(window);
		beautyPlan_.setup(window);
		optflowPlan_.setup(window);
	}

	virtual void infer(cv::Ptr<V4D> window) override {
		window->nvg([](const cv::Size& sz, const cv::Size_<float>& scale) {
			CV_UNUSED(scale);
			glViewport(0, 0, sz.width, sz.height);
		}, size(), scale_);

		window->nvgCtx()->setScale(scale_);
		window->capture();
		window->setDisableIO(true);
		window->fb([](cv::UMat& framebuffer, const cv::Size& tileSize, cv::UMat& captured){
			cv::resize(framebuffer, captured, tileSize);
		}, tileSz_, frames_.captured);


		window->fb([](cv::UMat& framebuffer, const cv::Size& tileSize, const cv::UMat& captured){
			framebuffer = cv::Scalar::all(0);
			captured.copyTo(framebuffer(cv::Rect(0, tileSize.height * 2, tileSize.width, tileSize.height)));
		}, tileSz_, frames_.captured);
		cubePlan_.infer(window);
		window->fb([](const cv::UMat& framebuffer, cv::UMat& cube){
			framebuffer.copyTo(cube);
		}, frames_.cube_);

		window->fb([](cv::UMat& framebuffer, const cv::Size& tileSize, const cv::UMat& captured){
			framebuffer = cv::Scalar::all(0);
			captured.copyTo(framebuffer(cv::Rect(0, tileSize.height * 2, tileSize.width, tileSize.height)));
		}, tileSz_, frames_.captured);
		manyCubesPlan_.infer(window);
		window->fb([](const cv::UMat& framebuffer, cv::UMat& many_cubes){
			framebuffer.copyTo(many_cubes);
		}, frames_.many_cubes_);

		window->fb([](cv::UMat& framebuffer, const cv::Size& tileSize, const cv::UMat& captured){
			framebuffer = cv::Scalar::all(0);
			captured.copyTo(framebuffer(cv::Rect(0, tileSize.height * 2, tileSize.width, tileSize.height)));
		}, tileSz_, frames_.captured);
		videoPlan_.infer(window);
		window->fb([](const cv::UMat& framebuffer, cv::UMat& video){
			framebuffer.copyTo(video);
		}, frames_.video_);

		window->fb([](cv::UMat& framebuffer, const cv::Size& tileSize, const cv::UMat& captured){
			framebuffer = cv::Scalar::all(0);
			captured.copyTo(framebuffer(cv::Rect(0, tileSize.height * 2, tileSize.width, tileSize.height)));
		}, tileSz_, frames_.captured);
		nanovgPlan_.infer(window);
		window->fb([](const cv::UMat& framebuffer, cv::UMat& nanovg){
			framebuffer.copyTo(nanovg);
		}, frames_.nanovg_);

		window->fb([](cv::UMat& framebuffer, const cv::Size& tileSize, const cv::UMat& captured){
			framebuffer = cv::Scalar::all(0);
			captured.copyTo(framebuffer(cv::Rect(0, tileSize.height * 2, tileSize.width, tileSize.height)));
		}, tileSz_, frames_.captured);
		shaderPlan_.infer(window);
		window->fb([](const cv::UMat& framebuffer, cv::UMat& shader){
			framebuffer.copyTo(shader);
		}, frames_.shader_);

		window->fb([](cv::UMat& framebuffer, const cv::Size& tileSize, const cv::UMat& captured){
			framebuffer = cv::Scalar::all(0);
			captured.copyTo(framebuffer(cv::Rect(0, tileSize.height * 2, tileSize.width, tileSize.height)));
		}, tileSz_, frames_.captured);
		fontPlan_.infer(window);
		window->fb([](const cv::UMat& framebuffer, cv::UMat& font){
			framebuffer.copyTo(font);
		}, frames_.font_);

		window->fb([](cv::UMat& framebuffer, const cv::Size& tileSize, const cv::UMat& captured){
			framebuffer = cv::Scalar::all(0);
			captured.copyTo(framebuffer(cv::Rect(0, tileSize.height * 2, tileSize.width, tileSize.height)));
		}, tileSz_, frames_.captured);
		pedestrianPlan_.infer(window);
		window->fb([](const cv::UMat& framebuffer, cv::UMat& pedestrian){
			framebuffer.copyTo(pedestrian);
		}, frames_.pedestrian_);

		window->fb([](cv::UMat& framebuffer, const cv::Size& tileSize, const cv::UMat& captured){
			framebuffer = cv::Scalar::all(0);
			captured.copyTo(framebuffer(cv::Rect(0, tileSize.height * 2, tileSize.width, tileSize.height)));
		}, tileSz_, frames_.captured);
		optflowPlan_.infer(window);
		window->fb([](const cv::UMat& framebuffer, cv::UMat& optflow){
			framebuffer.copyTo(optflow);
		}, frames_.optflow_);

		window->fb([](cv::UMat& framebuffer, const cv::Size& tileSize, const cv::UMat& captured){
			framebuffer = cv::Scalar::all(0);
			captured.copyTo(framebuffer(cv::Rect(0, tileSize.height * 2, tileSize.width, tileSize.height)));
		}, tileSz_, frames_.captured);
		beautyPlan_.infer(window);
		window->fb([](const cv::UMat& framebuffer, cv::UMat& beauty){
			framebuffer.copyTo(beauty);
		}, frames_.beauty_);

		window->fb([](cv::UMat& framebuffer, const cv::Size& tileSz, const Frames& frames){
			int w = tileSz.width;
			int h = tileSz.height;
			framebuffer = cv::Scalar::all(0);
			frames.cube_		(cv::Rect(0, h * 2, w, h)).copyTo(framebuffer(cv::Rect(0	, 0, w, h)));
			frames.many_cubes_	(cv::Rect(0, h * 2, w, h)).copyTo(framebuffer(cv::Rect(w	, 0, w, h)));
			frames.video_		(cv::Rect(0, h * 2, w, h)).copyTo(framebuffer(cv::Rect(w * 2, 0, w, h)));
			frames.nanovg_		(cv::Rect(0, h * 2, w, h)).copyTo(framebuffer(cv::Rect(0	, h, w, h)));
			frames.shader_		(cv::Rect(0, h * 2, w, h)).copyTo(framebuffer(cv::Rect(w	, h, w, h)));
			frames.font_		(cv::Rect(0, h * 2, w, h)).copyTo(framebuffer(cv::Rect(w * 2, h, w, h)));
			frames.pedestrian_	(cv::Rect(0, h * 2, w, h)).copyTo(framebuffer(cv::Rect(0    , h * 2, w, h)));
			frames.optflow_		(cv::Rect(0, h * 2, w, h)).copyTo(framebuffer(cv::Rect(w    , h * 2, w, h)));
			frames.beauty_		(cv::Rect(0, h * 2, w, h)).copyTo(framebuffer(cv::Rect(w * 2, h * 2, w, h)));
		}, tileSz_, frames_);

		window->setDisableIO(false);
		window->write();
	}

	virtual void teardown(cv::Ptr<V4D> window) override {
		cubePlan_.teardown(window);
		manyCubesPlan_.teardown(window);
		videoPlan_.teardown(window);
		shaderPlan_.teardown(window);
		pedestrianPlan_.teardown(window);
		optflowPlan_.teardown(window);
		beautyPlan_.teardown(window);
	}
};

int main(int argc, char** argv) {
#ifndef __EMSCRIPTEN__
	if (argc != 2) {
        cerr << "Usage: montage-demo <video-file>" << endl;
        exit(1);
    }
	constexpr double FPS = 60;
	constexpr const char* OUTPUT_FILENAME = "montage-demo.mkv";
#else
	CV_UNUSED(argc);
	CV_UNUSED(argv);
#endif
    using namespace cv::v4d;
    cv::Ptr<MontageDemoPlan> plan = new MontageDemoPlan(cv::Size(1920, 1080));
    cv::Ptr<V4D> window = V4D::make(plan->size(), "Montage Demo", ALL);
#ifndef __EMSCRIPTEN__
    //Creates a source from a file or a device
    auto src = makeCaptureSource(window, argv[1]);
    window->setSource(src);
    //Creates a writer sink (which might be hardware accelerated)
    auto sink = makeWriterSink(window, OUTPUT_FILENAME, FPS, plan->size());
    window->setSink(sink);
#endif
    window->run(plan, 2);

    return 0;
}

