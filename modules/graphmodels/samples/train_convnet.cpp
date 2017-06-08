#include "opencv2/graphmodels/multigpu_convnet.hpp"
#include <opencv2/core.hpp>

#include <iostream>
#include <stdio.h>

using namespace std;
using namespace cv;
using namespace cv::graphmodels;

int main(int argc, char** argv)
{
    const char *keys =
            "{ board b || GPU board(s) }"
            "{ model m || Model pbtxt file }"
            "{ train t || Training data pbtxt file }"
            "{ val   v || Validation data pbtxt file }";
    CommandLineParser parser(argc, argv, keys);
    int board(parser.get<int>("board"));
    string model_file(parser.get<string>("model"));
    string train_data_file(parser.get<string>("train"));
    string val_data_file(parser.get<string>("val"));
    if (argc<4)
    {
        parser.printMessage();
        return -1;
    }

    vector<int> boards;
    boards.push_back(board);
    
    bool multi_gpu = boards.size() > 1; 
    
    // Setup GPU boards.
    if (multi_gpu)
    {
        Matrix::SetupCUDADevices(boards);
    } else
    {
      Matrix::SetupCUDADevice(boards[0]);
    }
    for (const int &b : boards)
    {
        cout << "Using board " << b << endl;
    }

    ConvNet *net = multi_gpu ? new MultiGPUConvNet(model_file) :
                               new ConvNet(model_file);
    if (!val_data_file.empty())
    {// Use a validation set.
        net->SetupDataset(train_data_file, val_data_file);
    } else
    {
        net->SetupDataset(train_data_file);
    }
    net->AllocateMemory(false);
    net->Train();
    delete net;

    return 0;
}
