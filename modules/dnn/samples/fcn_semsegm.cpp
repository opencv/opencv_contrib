#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
using namespace cv;
using namespace cv::dnn;

#include <fstream>
#include <iostream>
#include <cstdlib>
using namespace std;

static std::vector<cv::Vec3b> readColors(const string &filename = "d:/dnn_opencv/pascal-classes.txt")
{
    std::vector<cv::Vec3b> colors;

    std::ifstream fp(filename);
    if (!fp.is_open())
    {
        std::cerr << "File with colors not found: " << filename << std::endl;
        exit(-1);
    }

    std::string line;
    while (!fp.eof())
    {
        std::getline(fp, line);
        if (line.length())
        {
            std::stringstream ss(line);

            std::string name; ss >> name;
            int temp;
            cv::Vec3b color;
            ss >> temp; color[0] = temp;
            ss >> temp; color[1] = temp;
            ss >> temp; color[2] = temp;
            colors.push_back(color);
        }
    }

    fp.close();
    return colors;
}

static void colorizeSegmentation(dnn::Blob &score, const std::vector<cv::Vec3b> &colors, cv::Mat &segm)
{
    const int rows = score.rows();
    const int cols = score.cols();
    const int chns = score.channels();

    cv::Mat maxCl(rows, cols, CV_8UC1);
    cv::Mat maxVal(rows, cols, CV_32FC1);
    for (int ch = 0; ch < chns; ch++)
    {
        for (int row = 0; row < rows; row++)
        {
            const float *ptrScore = score.ptrf(0, ch, row);
            uchar *ptrMaxCl = maxCl.ptr<uchar>(row);
            float *ptrMaxVal = maxVal.ptr<float>(row);
            for (int col = 0; col < cols; col++)
            {
                if (ptrScore[col] > ptrMaxVal[col])
                {
                    ptrMaxVal[col] = ptrScore[col];
                    ptrMaxCl[col] = ch;
                }
            }
        }
    }

    segm.create(rows, cols, CV_8UC3);
    for (int row = 0; row < rows; row++)
    {
        const uchar *ptrMaxCl = maxCl.ptr<uchar>(row);
        cv::Vec3b *ptrSegm = segm.ptr<cv::Vec3b>(row);
        for (int col = 0; col < cols; col++)
        {
            ptrSegm[col] = colors[ptrMaxCl[col]];
        }
    }

}

int main(int argc, char **argv)
{
    String modelTxt = "d:/dnn_opencv/fcn32s-heavy-pascal.prototxt";
    String modelBin = "d:/dnn_opencv/fcn32s-heavy-pascal.caffemodel";
    String imageFile = (argc > 1) ? argv[1] : "d:/dnn_opencv/rgb.jpg";

    std::vector<cv::Vec3b> colors = readColors();

    //! [Create the importer of Caffe model]
    Ptr<dnn::Importer> importer;
    try                                     //Try to import Caffe GoogleNet model
    {
        importer = dnn::createCaffeImporter(modelTxt, modelBin);
    }
    catch (const cv::Exception &err)        //Importer can throw errors, we will catch them
    {
        std::cerr << err.msg << std::endl;
    }
    //! [Create the importer of Caffe model]

    if (!importer)
    {
        std::cerr << "Can't load network by using the following files: " << std::endl;
        std::cerr << "prototxt:   " << modelTxt << std::endl;
        std::cerr << "caffemodel: " << modelBin << std::endl;
        std::cerr << "fcn32s-heavy-pascal.caffemodel can be downloaded here:" << std::endl;
        std::cerr << "http://dl.caffe.berkeleyvision.org/fcn32s-heavy-pascal.caffemodel" << std::endl;
        exit(-1);
    }

    //! [Initialize network]
    dnn::Net net;
    importer->populateNet(net);
    importer.release();                     //We don't need importer anymore
    //! [Initialize network]

    //! [Prepare blob]
    Mat img = imread(imageFile);
    if (img.empty())
    {
        std::cerr << "Can't read image from the file: " << imageFile << std::endl;
        exit(-1);
    }

    resize(img, img, Size(500, 500));       //FCN accepts 500x500 RGB-images
    dnn::Blob inputBlob = dnn::Blob(img);   //Convert Mat to dnn::Blob image batch
    //! [Prepare blob]

    //! [Set input blob]
    net.setBlob(".data", inputBlob);        //set the network input
    //! [Set input blob]

    //! [Make forward pass]
    net.forward();                          //compute output
    //! [Make forward pass]

    //! [Gather output]
    dnn::Blob score = net.getBlob("score");

    cv::Mat colorize;
    colorizeSegmentation(score, colors, colorize);
    cv::Mat show;
    cv::addWeighted(img, 0.4, colorize, 0.6, 0.0, show);
    cv::imshow("show", show);
    cv::waitKey(0);
    return 0;
} //main
