/* edge_drawing.cpp

This example illustrates how to use cv.ximgproc.EdgeDrawing class.

It uses the OpenCV library to load an image, and then use the EdgeDrawing class
to detect edges, lines, and ellipses. The detected features are then drawn and displayed.

The main loop allows the user changing parameters of EdgeDrawing by pressing following keys:

to toggle the grayscale conversion press 'space' key
to increase MinPathLength value press '/' key
to decrease MinPathLength value press '*' key
to increase MinLineLength value press '+' key
to decrease MinLineLength value press '-' key
to toggle NFAValidation value press 'n' key
to toggle PFmode value press 'p' key
to save parameters to file press 's' key
to load parameters from file press 'l' key

The program exits when the Esc key is pressed.
*/

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/ximgproc.hpp>
#include <iostream>

void EdgeDrawingDemo(const cv::Mat src, cv::Ptr<cv::ximgproc::EdgeDrawing> ed, bool convert_to_gray);

void EdgeDrawingDemo(const cv::Mat src, cv::Ptr<cv::ximgproc::EdgeDrawing> ed, bool convert_to_gray)
{
    cv::Mat ssrc = cv::Mat::zeros(src.size(), src.type());
    cv::Mat lsrc = src.clone();
    cv::Mat esrc = src.clone();

    std::cout << std::endl << "convert_to_gray: " << convert_to_gray << std::endl;
    std::cout << "MinPathLength: " << ed->params.MinPathLength << std::endl;
    std::cout << "MinLineLength: " << ed->params.MinLineLength << std::endl;
    std::cout << "PFmode: " << ed->params.PFmode << std::endl;
    std::cout << "NFAValidation: " << ed->params.NFAValidation << std::endl;

    cv::TickMeter tm;
    tm.start();

    cv::Mat img_to_detect;

    if (convert_to_gray)
    {
        cv::cvtColor(src, img_to_detect, cv::COLOR_BGR2GRAY);
    }
    else
    {
        img_to_detect = src;
    }

    cv::imshow("source image", img_to_detect);

    tm.start();

    // Detect edges
    ed->detectEdges(img_to_detect);

    std::vector<std::vector<cv::Point>> segments = ed->getSegments();
    std::vector<cv::Vec4f> lines;
    ed->detectLines(lines);
    std::vector<cv::Vec6d> ellipses;
    ed->detectEllipses(ellipses);

    tm.stop();

    cv::RNG& rng = cv::theRNG();
    cv::setRNGSeed(0);

    // Draw detected edge segments
    for (const auto& segment : segments)
    {
        cv::Scalar color(rng.uniform(0, 256), rng.uniform(0, 256), rng.uniform(0, 256));
        cv::polylines(ssrc, segment, false, color, 1, cv::LINE_8);
    }

    cv::imshow("detected edge segments", ssrc);

    // Draw detected lines
    if (!lines.empty())  // Check if the lines have been found and only then iterate over these and add them to the image
    {
        for (size_t i = 0; i < lines.size(); i++)
        {
            cv::line(lsrc, cv::Point2d(lines[i][0], lines[i][1]), cv::Point2d(lines[i][2], lines[i][3]), cv::Scalar(0, 0, 255), 1, cv::LINE_AA);
        }
    }

    cv::imshow("detected lines", lsrc);

    // Draw detected circles and ellipses
    if (!ellipses.empty())  // Check if circles and ellipses have been found and only then iterate over these and add them to the image
    {
        for (const auto& ellipse : ellipses)
        {
            cv::Point center((int)ellipse[0], (int)ellipse[1]);
            cv::Size axes((int)ellipse[2] + (int)ellipse[3], (int)ellipse[2] + (int)ellipse[4]);
            double angle(ellipse[5]);
            cv::Scalar color = (ellipse[2] == 0) ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 0, 255);
            cv::ellipse(esrc, center, axes, angle, 0, 360, color, 1, cv::LINE_AA);
        }
    }

    cv::imshow("detected circles and ellipses", esrc);
    std::cout << "Total Detection Time : " << tm.getTimeMilli() << "ms." << std::endl;
}

int main(int argc, char** argv)
{
    std::string filename = (argc > 1) ? argv[1] : "board.jpg";
    cv::Mat src = cv::imread(cv::samples::findFile(filename));

    if (src.empty())
    {
        std::cerr << "Error: Could not open or find the image!" << std::endl;
        return -1;
    }

    cv::Ptr<cv::ximgproc::EdgeDrawing> ed = cv::ximgproc::createEdgeDrawing();

    // Set parameters (refer to the documentation for all parameters)
    ed->params.MinPathLength = 10;     // try changing this value by pressing '/' and '*' keys
    ed->params.MinLineLength = 10;     // try changing this value by pressing '+' and '-' keys
    ed->params.PFmode = false;         // default value is false, try switching by pressing 'p' key
    ed->params.NFAValidation = true;   // default value is true, try switching by pressing 'n' key

    bool convert_to_gray = true;
    int key = 0;

    while (key != 27)
    {
        EdgeDrawingDemo(src, ed, convert_to_gray);
        key = cv::waitKey(0);

        switch (key)
        {
        case 32:  // space key
            convert_to_gray = !convert_to_gray;
            break;
        case 'p':  // 'p' key
            ed->params.PFmode = !ed->params.PFmode;
            break;
        case 'n':  // 'n' key
            ed->params.NFAValidation = !ed->params.NFAValidation;
            break;
        case '+':  // '+' key
            ed->params.MinLineLength = std::max(0, ed->params.MinLineLength + 5);
            break;
        case '-':  // '-' key
            ed->params.MinLineLength = std::max(0, ed->params.MinLineLength - 5);
            break;
        case '/':  // '/' key
            ed->params.MinPathLength += 20;
            break;
        case '*':  // '*' key
            ed->params.MinPathLength = std::max(0, ed->params.MinPathLength - 20);
            break;
        case 's':  // 's' key
        {
            cv::FileStorage fs("ed-params.xml", cv::FileStorage::WRITE);
            ed->params.write(fs);
            fs.release();
            std::cout << "Parameters saved to ed-params.xml" << std::endl;
        }
        break;
        case 'l':  // 'l' key
        {
            cv::FileStorage fs("ed-params.xml", cv::FileStorage::READ);
            if (fs.isOpened())
            {
                ed->params.read(fs.root());
                fs.release();
                std::cout << "Parameters loaded from ed-params.xml" << std::endl;
            }
        }
        break;
        default:
            break;
        }
    }
    return 0;
}
