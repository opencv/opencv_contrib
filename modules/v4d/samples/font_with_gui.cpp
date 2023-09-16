#include <opencv2/v4d/v4d.hpp>

using namespace cv;
using namespace cv::v4d;

int main() {
    Ptr<V4D> window = V4D::make(Size(960, 960), cv::Size(), "Font Rendering with GUI");

    //The text color. NanoGUI uses rgba with floating point
//	nanogui::Color textColor = {0.0f, 0.0f, 1.0f, 1.0f};
	//The font size
	float size = 40.0f;
	//The text
	string hw = "hello world";
	//Setup the GUI. First thing you should do is create a light-weight dialog and add widgets as needed.
	//Variables passed to the FormHelper (e.g. via makeFormVariable) will be directly modified by the GUI.
	//Please note that you can build more complex GUIs if you use NanoGUI directly on the created dialog
	//instead of creating widgets through FormHelper::make* calls.
//	window->nanogui([&](FormHelper& form) {
//		//Create a light-weight dialog
//		form.makeDialog(5, 30, "Settings");
//		//Create a group
//		form.makeGroup("Font");
//		//Create a from variable. The type of widget is deduced from the variable type.
//		form.makeFormVariable("Font Size", size, 1.0f, 100.0f, true, "pt", "Font size of the text crawl");
//		//Create a color picker
//		form.makeColorPicker("Text Color", textColor, "The text color");
//	});

	window->run([&](Ptr<V4D> window) {
		//Render the text at the center of the screen using parameters from the GUI.
		window->nvg([&](const Size& sz) {
			using namespace cv::v4d::nvg;
			clear();
			fontSize(size);
			fontFace("sans-bold");
			fillColor(Scalar(255, 0, 0, 255));
			textAlign(NVG_ALIGN_CENTER | NVG_ALIGN_TOP);
			text(sz.width / 2.0, sz.height / 2.0, hw.c_str(), hw.c_str() + hw.size());
		});

		return window->display();
	});
}

