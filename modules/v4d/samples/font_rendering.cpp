#include <opencv2/v4d/v4d.hpp>

using namespace cv;
using namespace cv::v4d;

int main() {
	cv::Ptr<V4D> window = V4D::make(960, 960, "Font Rendering");

	class FontRenderingPlan: public Plan {
		//The text to render
		string hw_ = "Hello World";
	public:
		void infere(Ptr<V4D> win) override {
			//Render the text at the center of the screen. Note that you can load you own fonts.
			win->nvg([](const Size &sz, const string &str) {
				using namespace cv::v4d::nvg;
				clear();
				fontSize(40.0f);
				fontFace("sans-bold");
				fillColor(Scalar(255, 0, 0, 255));
				textAlign(NVG_ALIGN_CENTER | NVG_ALIGN_TOP);
				text(sz.width / 2.0, sz.height / 2.0, str.c_str(),
						str.c_str() + str.size());
			}, win->fbSize(), hw_);
		}
	};

	window->run<FontRenderingPlan>(0);
}

