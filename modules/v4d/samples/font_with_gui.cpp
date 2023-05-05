#include <opencv2/v4d/v4d.hpp>

using namespace cv;
using namespace cv::v4d;

static Ptr<V4D> window = V4D::make(Size(1280, 720), "Font Rendering with GUI");

int main() {
	//The text color. NanoGUI uses rgba with floating point
	nanogui::Color textColor = {0.0f, 0.0f, 1.0f, 1.0f};
	//The font size
	float size = 40.0f;
	//The text
	string hw = "hello world";
	//Setup the GUI
	window->nanogui([&](FormHelper& form) {
		//Create a light-weight dialog
		form.makeDialog(5, 30, "Settings");
		//Create a group
		form.makeGroup("Font");
		//Create a from variable. The type of widget is deduced from the variable type.
		form.makeFormVariable("Font Size", size, 1.0f, 100.0f, true, "pt", "Font size of the text crawl");
		//Create a color picker
		form.makeColorPicker("Text Color", textColor, "The text color");
	});

	window->run([&]() {
        window->clear();
		//Render the text at the center of the screen
		window->nvg([&](const Size& sz) {
			using namespace cv::v4d::nvg;
			fontSize(size);
			fontFace("sans-bold");
			fillColor(Scalar(textColor.b() * 255, textColor.g() * 255, textColor.r() * 255, 255));
			textAlign(NVG_ALIGN_CENTER | NVG_ALIGN_TOP);
			text(sz.width / 2.0, sz.height / 2.0, hw.c_str(), hw.c_str() + hw.size());
		});

        window->updateFps();
		//Display the framebuffer in the native window
		return window->display();
	});
}
