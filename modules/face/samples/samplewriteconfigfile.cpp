#include "opencv2/core.hpp"
#include <iostream>
#include <string>

using namespace cv;
using namespace std;
int main(int argc,const char ** argv){
  CommandLineParser parser(argc, argv,
        "{ help h usage ?             |      | give the following arguments in following format }"
        "{ filename f                 |.     | (required) path to file which you want to create as config file [example - /data/config.xml] }"
        "{ cascade_depth cd           |  10  | (required) This stores the depth of cascade of regressors used for training.}"
        "{ tree_depth td              |  4   | (required) This stores the depth of trees created as weak learners during gradient boosting.}"
        "{ num_trees_per_cascade_level|  500 | (required) This stores number of trees required per cascade level.}"
        "{ learning_rate              |  0.1 | (required) This stores the learning rate for gradient boosting.}"
        "{ oversampling_amount        |  20  | (required) This stores the oversampling amount for the samples.}"
        "{ num_test_coordinates       |  400 | (required) This stores number of test coordinates required for making the split.}"
        "{ lambda                     |  0.1 | (required) This stores the value used for calculating the probabilty.}"
        "{ num_test_splits            |  20  | (required) This stores the number of test splits to be generated before making the best split.}"
        );
    // Read in the input arguments
    if (parser.has("help")){
        parser.printMessage();
        cerr << "TIP: Use absolute paths to avoid any problems with the software!" << endl;
        return 0;
    }
    //These variables have been initialised as defined in the research paper "One millisecond face alignment" CVPR 2014
    int cascade_depth = 15;
    int tree_depth = 4;
    int num_trees_per_cascade_level = 500;
    float learning_rate = float(0.1);
    int oversampling_amount = 20;
    int num_test_coordinates = 400;
    float lambda = float(0.1);
    int num_test_splits = 20;

    cascade_depth = parser.get<int>("cascade_depth");
    tree_depth = parser.get<int>("tree_depth");
    num_trees_per_cascade_level = parser.get<int>("num_trees_per_cascade_level");
    learning_rate = parser.get<float>("learning_rate");
    oversampling_amount = parser.get<int>("oversampling_amount");
    num_test_coordinates = parser.get<int>("num_test_coordinates");
    lambda = parser.get<float>("lambda");
    num_test_splits = parser.get<int>("num_test_splits");
    string filename(parser.get<string>("filename"));
    FileStorage fs(filename, FileStorage::WRITE);
    if (!fs.isOpened())
    {
        cerr << "Failed to open " << filename << endl;
        parser.printMessage();
        return -1;
    }
    fs << "cascade_depth" << cascade_depth;
    fs << "tree_depth"<< tree_depth;
    fs << "num_trees_per_cascade_level" << num_trees_per_cascade_level;
    fs << "learning_rate" << learning_rate;
    fs << "oversampling_amount" << oversampling_amount;
    fs << "num_test_coordinates" << num_test_coordinates;
    fs << "lambda" << lambda ;
    fs << "num_test_splits"<< num_test_splits;
    fs.release();
    cout << "Write Done." << endl;
    return 0;
}
