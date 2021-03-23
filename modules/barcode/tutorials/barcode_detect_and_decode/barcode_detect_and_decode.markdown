Bar code Recognition{#tutorial_barcode_detect_and_decode}
======================

Goal
----

In this chapter,

-   We will familiarize with the bar code detection and decoding methods available in OpenCV.

Basics
----

Bar code is major technique to identify commodity in real life.  A common bar code is a pattern of parallel lines arranged by black bars and white bars with vastly different reflectivity. Bar code recognition is to scan the bar code in the horizontal direction to get a string of binary codes composed of bars of different widths and colors, that is, the code information of the bar code. The content of bar code can be decoded by matching with various bar code encoding methods. For current work, we only support EAN13 encoding method.

### EAN 13

To be done

### BarcodeDetector
Several algorithms were introduced for bar code recognition.

While coding, we firstly need to create a **cv::barcode::BarcodeDetector** object.  It has mainly three member functions, which will be introduced in the following.

#### detect

It is a algorithm based on directional consistency. First of all, we compute the average squared gradients of every pixels. It was proposed in the paper "Systematic methods for the computation of the directional  fields and singular points of fingerprints" by A.M. Bazen and S.H. Gerez in 2002. Then we divide the image into some square patches and compute the **gradient variance** and **mean gradient direction** of each patch. At last we connected the patches that have **low gradient variance** and **similar gradient direction**. In this stage, we use multi-size patches to capture the gradient distribution of multi-size bar codes, and apply non-maximum suppression to filter duplicate proposals.

See a simple example below:

@code{.cpp}
#include "opencv2/barcode.hpp"
#include "opencv2/imgproc.hpp"

using namespace cv;
barcode::BarcodeDetector bardet;
Mat frame = imread("bar_code.jpg");
std::vector<Point> corners,
bool ok = bardet.detect(frame, corners);
@endcode
( All the results are stored in `corners`, and `ok` would be true if there is  bar code detected ).

#### decode

To be done...

#### detectAndDecode

This function combines `detect`  and `decode`.  A simple example below to use this function showing recognized bar codes.

@code{.cpp}
#include <iostream>
#include "opencv2/barcode.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"

using namespace cv;
barcode::BarcodeDetector bardet;
Mat frame = imread("bar_code.jpg");
std::vector<Point> corners,
std::vector<std::string> decoded_info;
std::vector<barcode::BarcodeType> decoded_format;

ok = bardet.detectAndDecode(frame, decoded_info, decoded_format, corners);
if (ok)
{
	for (size_t i = 0; i < corners.size(); i += 4)
	{
		size_t bar_idx = i / 4;
		std::vector<Point> barcode_contour(corners.begin() + i, corners.begin() + i + 4);
		std::cout << decoded_info[bar_idx] << " " << decoded_format[bar_idx] << std::endl;
		putText(frame, decoded_info[bar_idx], barcode_contour[1], cv::FONT_HERSHEY_PLAIN, 			1, Scalar(255, 0, 0), 2);
		// use red bouding box to represent decoding failure
		if (decoded_format[bar_idx] == barcode::BarcodeType::NONE)
		{
			for (int j = 0; j < 4; j++)
			{
				line(frame, barcode_contour[j], barcode_contour[(j + 1) % 4], Scalar(0, 0, 						255), 2);
			}
		}
		// use green bouding box to represent decoding success
		else
		{
			for (int j = 0; j < 4; j++)
			{
				line(frame, barcode_contour[j], barcode_contour[(j + 1) % 4], Scalar(0, 255, 0), 					2);
			}
		}
	}
}
imshow("result", frame);
waitKey();
@endcode

Results
-------

**Original Image**

Below image shows four EAN 13 bar codes photoed by a smart phone.

![image](images/4_barcodes.jpg)

**Result of detectAndDecode**

Bar codes are bounded by green box, and decoded numbers are lying on the boxes.

![image](images/result.jpg)
