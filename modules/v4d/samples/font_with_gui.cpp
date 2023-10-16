#include <opencv2/v4d/v4d.hpp>

using namespace cv;
using namespace cv::v4d;

class FontWithGuiPlan: public Plan {
	//The font size
	float size_ = 40.0f;
	//The text hue
	std::vector<float> color_ = {1.0f, 0.0f, 0.0f};
	//The text
	string hw_ = "hello world";
public:
	void gui(Ptr<V4D> window) override {
		window->imgui([this](Ptr<V4D> win, ImGuiContext* ctx) {
			using namespace ImGui;
			SetCurrentContext(ctx);
			Begin("Settings");
			SliderFloat("Font Size", &size_, 1.0f, 100.0f);
			ColorPicker3("Text Color", color_.data());
			End();
		});
	}

	void infer(Ptr<V4D> window) override {
		//Render the text at the center of the screen using parameters from the GUI.
		window->nvg([](const Size& sz, const string& str, const float& s, const std::vector<float>& c) {
			using namespace cv::v4d::nvg;
			clear();
			fontSize(s);
			fontFace("sans-bold");
			fillColor(Scalar(c[2] * 255, c[1] * 255, c[0] * 255, 255));
			textAlign(NVG_ALIGN_CENTER | NVG_ALIGN_TOP);
			text(sz.width / 2.0, sz.height / 2.0, str.c_str(), str.c_str() + str.size());
		}, window->fbSize(), hw_, size_, color_);
	}
};

int main() {
    Ptr<V4D> window = V4D::make(960, 960, "Font Rendering with GUI");
	window->run<FontWithGuiPlan>(0);
}

