#include "opencv2/dehaze.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include <iostream>


int main(int argc, char** argv)
{
   if (argc < 2) {
       std::cout << "Usage: dehaze_demo <path_to_hazy_image>" << std::endl;
       return -1;
   }


   // Load hazy image
   cv::Mat src = cv::imread(argv[1]);
   if (src.empty()) {
       std::cout << "Error: could not load image: " << argv[1] << std::endl;
       return -1;
   }


   std::cout << "Image loaded: " << src.cols << "x" << src.rows << std::endl;


   // Run the full dehaze pipeline
   cv::Mat dst;
   cv::dehazeImage(src, dst);


   std::cout << "Dehazing complete!" << std::endl;


   // Show results side by side
   cv::imshow("Hazy Input", src);
   cv::imshow("Dehazed Output", dst);


   // Save output
   cv::imwrite("dehazed_output.jpg", dst);
   std::cout << "Saved to dehazed_output.jpg" << std::endl;


   cv::waitKey(0);
   return 0;
}