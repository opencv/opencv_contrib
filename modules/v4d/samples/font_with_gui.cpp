#include <opencv2/v4d/v4d.hpp>

using namespace cv;
using namespace cv::v4d;

class FontWithGuiPlan: public Plan {
	struct Params {
	//The font size
	float size_ = 40.0f;
	//The text hue
	cv::Scalar_<float> color_ = {1.0f, 0.0f, 0.0f, 1.0f};
	} params_;
	//The text
	string hw_ = "hello world";
public:
	void gui(Ptr<V4D> window) override {
		window->imgui([](Ptr<V4D> win, ImGuiContext* ctx, Params& params) {
			using namespace ImGui;
			SetCurrentContext(ctx);
			Begin("Settings");
			SliderFloat("Font Size", &params.size_, 1.0f, 100.0f);
			ColorPicker4("Text Color", params.color_.val);
			End();
		}, params_);
	}

	void infer(Ptr<V4D> window) override {
		//Render the text at the center of the screen using parameters from the GUI.
		window->nvg([](const Size& sz, const string& str, const Params& params) {
			using namespace cv::v4d::nvg;
			clear();
			fontSize(params.size_);
			fontFace("sans-bold");
			fillColor(params.color_ * 255.0);
			textAlign(NVG_ALIGN_CENTER | NVG_ALIGN_TOP);
			text(sz.width / 2.0, sz.height / 2.0, str.c_str(), str.c_str() + str.size());
		}, window->fbSize(), hw_, params_);
	}
};

int main() {
    Ptr<V4D> window = V4D::make(960, 960, "Font Rendering with GUI");
	window->run<FontWithGuiPlan>(0);
}

