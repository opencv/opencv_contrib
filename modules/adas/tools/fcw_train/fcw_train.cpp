#include <cstdio>
#include <cstring>

#include <string>
using std::string;

#include <vector>
using std::vector;

#include <fstream>
using std::ifstream;
using std::getline;

#include <sstream>
using std::stringstream;

#include <iostream>
using std::cerr;
using std::endl;

#include <opencv2/core.hpp>
using cv::Rect;
using cv::Size;
#include <opencv2/highgui.hpp>
using cv::imread;
#include <opencv2/core/utility.hpp>
using cv::CommandLineParser;
using cv::FileStorage;
#include <opencv2/core/utility.hpp>

#include <ctime>        // std::time
#include <cstdlib>      // std::rand, std::srand

#include <opencv2/xobjdetect.hpp>


using cv::xobjdetect::ICFDetectorParams;
using cv::xobjdetect::ICFDetector;
using cv::xobjdetect::WaldBoost;
using cv::xobjdetect::WaldBoostParams;
using cv::Mat;

static bool read_model_size(const char *str, int *rows, int *cols)
{
    int pos = 0;
    if( sscanf(str, "%dx%d%n", rows, cols, &pos) != 2 || str[pos] != '\0' ||
        *rows <= 0 || *cols <= 0)
    {
        return false;
    }
    return true;
}

static int randomPred (int i) { return std::rand()%i;}

int main(int argc, char *argv[])
{
    
    const string keys =
        "{help           |           | print this message}"
        "{pos_path       |       pos | path to training object samples}"
        "{bg_path        |        bg | path to background images}"
        "{bg_per_image   |         5 | number of windows to sample per bg image}"
        "{feature_count  |     10000 | number of features to generate}"
        "{weak_count     |       100 | number of weak classifiers in cascade}"
        "{model_size     |     40x40 | model size in pixels}"
        "{model_filename | model.xml | filename for saving model}"
        "{features_type  |       icf | features type, \"icf\" or \"acf\"}"
        "{alpha          |      0.02 | alpha value}"
        "{is_grayscale   |     false | read the image as grayscale}"
        "{use_fast_log   |     false | use fast log function}"
        "{limit_ps       |        -1 | limit to positive samples (-1 means all)}"
        "{limit_bg       |        -1 | limit to negative samples (-1 means all)}"
        ;


    CommandLineParser parser(argc, argv, keys);
    parser.about("FCW trainer");

    if( parser.has("help") || argc == 1)
    {
        parser.printMessage();
        return 0;
    }

    string pos_path = parser.get<string>("pos_path");
    string bg_path = parser.get<string>("bg_path");
    string model_filename = parser.get<string>("model_filename");

    ICFDetectorParams params;
    params.feature_count = parser.get<int>("feature_count");
    params.weak_count = parser.get<int>("weak_count");
    params.bg_per_image = parser.get<int>("bg_per_image");
    params.features_type = parser.get<string>("features_type");
    params.alpha = parser.get<float>("alpha");
    params.is_grayscale = parser.get<bool>("is_grayscale");
    params.use_fast_log = parser.get<bool>("use_fast_log");
    
    int limit_ps = parser.get<int>("limit_ps");
    int limit_bg = parser.get<int>("limit_bg");    
    
    string model_size = parser.get<string>("model_size");
    if( !read_model_size(model_size.c_str(), &params.model_n_rows,
        &params.model_n_cols) )
    {
        cerr << "Error reading model size from `" << model_size << "`" << endl;
        return 1;
    }
    
    if( params.feature_count <= 0 )
    {
        cerr << "feature_count must be positive number" << endl;
        return 1;
    }

    if( params.weak_count <= 0 )
    {
        cerr << "weak_count must be positive number" << endl;
        return 1;
    }

    if( params.features_type != "icf" &&  params.features_type != "acf" )
    {
        cerr << "features_type must be \"icf\" or \"acf\"" << endl;
        return 1;
    }
    if( params.alpha <= 0 )
    {
        cerr << "alpha must be positive float number" << endl;
        return 1;
    }
    if( !parser.check() )
    {
        parser.printErrors();
        return 1;
    }
    
    std::vector<cv::String> pos_filenames;
    glob(pos_path, pos_filenames);

    std::vector<cv::String> bg_filenames;
    glob(bg_path, bg_filenames);
        
    if(limit_ps != -1 && (int)pos_filenames.size() > limit_ps)
      pos_filenames.erase(pos_filenames.begin()+limit_ps, pos_filenames.end());
    if(limit_bg != -1 && (int)bg_filenames.size() > limit_bg)
      bg_filenames.erase(bg_filenames.begin()+limit_bg, bg_filenames.end());
    
    //random pick input images
    bool random_shuffle = false;
    if(random_shuffle)
    {
      std::srand ( unsigned ( std::time(0) ) );
      std::random_shuffle ( pos_filenames.begin(), pos_filenames.end(), randomPred );
      std::random_shuffle ( bg_filenames.begin(), bg_filenames.end(), randomPred );
    }
    
    int samples_size = (int)((params.bg_per_image * bg_filenames.size()) + pos_filenames.size());
    int features_size = params.feature_count;
    int max_features_allowed = (int)(INT_MAX/(sizeof(int)* samples_size));
    int max_samples_allowed = (int)(INT_MAX/(sizeof(int)* features_size));
    int total_samples = (int)((params.bg_per_image * bg_filenames.size()) + pos_filenames.size());
    
    
    if(total_samples >max_samples_allowed)
    {
      CV_Error_(1, ("exceeded maximum number of samples. Maximum number of samples with %d features is %d, you have %d (%d positive samples + (%d bg * %d bg_per_image))\n",features_size,max_samples_allowed,total_samples,pos_filenames.size(),bg_filenames.size(),params.bg_per_image ));
    }
    
    if(params.feature_count >max_features_allowed)
    {
      CV_Error_(1, ("exceeded maximum number of features. Maximum number of features with %d samples is %d, you have %d\n",samples_size,max_features_allowed, features_size ));
    }
    
    std::cout<<pos_filenames.size()<<std::endl;
    std::cout<<bg_filenames.size()<<std::endl;

    ICFDetector detector;    

    
    detector.train(pos_filenames, bg_filenames, params);

    FileStorage fs(model_filename, FileStorage::WRITE);
    fs << "icfdetector";
    detector.write(fs);
    fs.release();
}
