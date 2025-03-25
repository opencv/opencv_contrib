#include <opencv2/v4d/v4d.hpp>

using namespace cv;
using namespace cv::v4d;

class FontWithGuiPlan: public Plan {
	enum Names {
		SIZE,
		COLOR
	};
	using Params = ThreadSafeMap<Names>;
	inline static Params params_;

	//The text
	string hw_ = "hello world";
public:
	FontWithGuiPlan(const cv::Size& sz) : Plan(sz) {
		params_.set(SIZE, 40.0f);
		params_.set(COLOR, cv::Scalar_<float>(1.0f, 0.0f, 0.0f, 1.0f));
	}

	void gui(Ptr<V4D> window) override {
		window->imgui([](Ptr<V4D> win, ImGuiContext* ctx, Params& params) {
			CV_UNUSED(win);
			using namespace ImGui;
			SetCurrentContext(ctx);
			Begin("Settings");
			SliderFloat("Font Size", params.ptr<float>(SIZE), 1.0f, 100.0f);
			ColorPicker4("Text Color", params.ptr<cv::Scalar_<float>>(COLOR)->val);
			End();
		}, params_);
	}

	void infer(Ptr<V4D> window) override {
		//Render the text at the center of the screen using parameters from the GUI.
		window->nvg([](const Size& sz, const string& str, Params& params) {
			using namespace cv::v4d::nvg;
			clear();
			fontSize(params.get<float>(SIZE));
			fontFace("sans-bold");
			fillColor(params.get<cv::Scalar_<float>>(COLOR) * 255.0);
			textAlign(NVG_ALIGN_CENTER | NVG_ALIGN_TOP);
			text(sz.width / 2.0, sz.height / 2.0, str.c_str(), str.c_str() + str.size());
		}, window->fbSize(), hw_, params_);
	}
};

int main() {
	Ptr<FontWithGuiPlan> plan = new FontWithGuiPlan(cv::Size(960,960));
    Ptr<V4D> window = V4D::make(plan->size(), "Font Rendering with GUI");
	window->run(plan);
}

