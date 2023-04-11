#include <opencv2/viz2d/viz2d.hpp>
#include <opencv2/viz2d/nvg.hpp>

using namespace cv;
using namespace cv::viz;

constexpr int WIDTH = 1280;
constexpr int HEIGHT = 720;

int main(int argc, char** argv) {
	Ptr<Viz2D> v2d = Viz2D::make(Size(WIDTH, HEIGHT), "Font Rendering with GUI");
	//The text color. NanoGUI uses rgba with floating point
	nanogui::Color textColor = {0.0f, 0.0f, 1.0f, 1.0f};
	//The font size
	float size = 40.0f;
	//The text
	string hw = "hello world";
	//Setup the GUI
	v2d->nanogui([&](FormHelper& form) {
		//Create a light-weight dialog
		form.makeDialog(5, 30, "Settings");
		//Create a group
		form.makeGroup("Font");
		//Create a from variable. The type of widget is deduced from the variable type.
		form.makeFormVariable("Font Size", size, 1.0f, 100.0f, true, "pt", "Font size of the text crawl");
		//Create a color picker
		form.makeColorPicker("Text Color", textColor, "The text color");
	});

	v2d->run([&]() {
		v2d->clear();
		//Render the text at the center of the screen
		v2d->nvg([&](const Size& sz) {
			using namespace cv::viz::nvg;
			fontSize(size);
			fontFace("sans-bold");
			fillColor(Scalar(textColor.b() * 255, textColor.g() * 255, textColor.r() * 255, 255));
			textAlign(NVG_ALIGN_CENTER | NVG_ALIGN_TOP);
			text(WIDTH / 2.0, HEIGHT / 2.0, hw.c_str(), hw.c_str() + hw.size());
		});
		return v2d->display();
	});
}

