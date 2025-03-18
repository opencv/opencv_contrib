// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include <opencv2/dnn_superres.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>

using namespace std;
using namespace cv;
using namespace dnn_superres;

/**
 * @brief Demonstrate the use of SRGAN and RDN super resolution models
 * @author Contributed by Akalp
 */

const char* keys =
{
    "{ help  h     |                 | Print help message. }"
    "{ input i     |                 | Path to input image. }"
    "{ model m     |                 | Path to model weights. }"
    "{ scale s     | 4               | Scale factor (2, 3, 4). }"
    "{ model_type t| srgan           | Model type (srgan or rdn). }"
    "{ output o    | sr_result.png   | Path to output image. }"
    "{ cuda c      | false           | Use CUDA for GPU acceleration. }"
};

int main(int argc, char* argv[])
{
    // Parse command line arguments
    CommandLineParser parser(argc, argv, keys);
    parser.about("Super Resolution using SRGAN and RDN models");

    if (parser.has("help"))
    {
        parser.printMessage();
        return 0;
    }

    // Get the required parameters
    String input_path = parser.get<String>("input");
    String model_path = parser.get<String>("model");
    String model_type = parser.get<String>("model_type");
    String output_path = parser.get<String>("output");
    int scale = parser.get<int>("scale");
    bool use_cuda = parser.get<bool>("cuda");

    if (!parser.check())
    {
        parser.printErrors();
        return -1;
    }

    if (input_path.empty() || model_path.empty())
    {
        cerr << "Input image and model are required!" << endl;
        return -1;
    }

    // Convert model type to lowercase for comparison
    transform(model_type.begin(), model_type.end(), model_type.begin(), ::tolower);
    
    // Check if model type is valid
    if (model_type != "srgan" && model_type != "rdn")
    {
        cerr << "Invalid model type. Supported types: srgan, rdn" << endl;
        return -1;
    }

    try
    {
        // Load the image
        Mat img = imread(input_path);
        if (img.empty())
        {
            cerr << "Could not load the image: " << input_path << endl;
            return -1;
        }

        // Create the super resolution object
        DnnSuperResImpl sr;
        
        // Read the model
        sr.readModel(model_path);
        
        // Set the model and scale
        sr.setModel(model_type, scale);
        
        // Set GPU if requested
        if (use_cuda)
        {
#ifdef HAVE_CUDA
            sr.setPreferableBackend(dnn::DNN_BACKEND_CUDA);
            sr.setPreferableTarget(dnn::DNN_TARGET_CUDA);
            cout << "Using CUDA backend" << endl;
#else
            cerr << "CUDA is not available in this build. Using CPU." << endl;
#endif
        }

        cout << "Processing image with " << model_type << " model..." << endl;
        cout << "Original resolution: " << img.cols << "x" << img.rows << endl;
        
        // Create a window for the original image
        namedWindow("Original Image", WINDOW_NORMAL);
        imshow("Original Image", img);
        
        // Upscale the image
        Mat result;
        
        // Measure processing time
        double t = (double)getTickCount();
        sr.upsample(img, result);
        t = ((double)getTickCount() - t) / getTickFrequency();
        
        cout << "Done in " << t << " seconds" << endl;
        cout << "Upscaled resolution: " << result.cols << "x" << result.rows << endl;
        
        // Create a window for the super resolution result
        namedWindow("Super Resolution Result", WINDOW_NORMAL);
        imshow("Super Resolution Result", result);
        
        // Save the result
        imwrite(output_path, result);
        cout << "Result saved to: " << output_path << endl;
        
        waitKey(0);
        return 0;
    }
    catch (const cv::Exception& e)
    {
        cerr << "Error: " << e.what() << endl;
        return -1;
    }
} 
