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

#include <opencv2/core.hpp>
using cv::Rect;

#include "icfdetector.hpp"
#include "waldboost.hpp"

using cv::adas::ICFDetectorParams;
using cv::adas::ICFDetector;

static bool read_pos_int(const char *str, int *n)
{
    int pos = 0;
    if( sscanf(str, "%d%n", n, &pos) != 1 || str[pos] != '\0' || *n <= 0 )
    {
        return false;
    }
    return true;
}

static bool read_model_size(char *str, int *rows, int *cols)
{
    int pos = 0;
    if( sscanf(str, "%dx%d%n", rows, cols, &pos) != 2 || str[pos] != '\0' ||
        *rows <= 0 || *cols <= 0)
    {
        return false;
    }
    return true;
}

static bool read_overlap(const char *str, double *overlap)
{
    int pos = 0;
    if( sscanf(str, "%lf%n", overlap, &pos) != 1 || str[pos] != '\0' ||
        *overlap < 0 || *overlap > 1)
    {
        return false;
    }
    return true;
}

static bool read_labelling(const string& path,
    vector<string>& filenames, vector< vector<Rect> >& labelling)
{
    string labelling_path = path + "/gt.txt";
    string filename, line;
    int x1, y1, x2, y2;
    char delim;
    ifstream ifs(labelling_path.c_str());
    if( !ifs.good() )
        return false;

    while( getline(ifs, line) )
    {
        stringstream stream(line);
        stream >> filename;
        filenames.push_back(path + "/" + filename);
        vector<Rect> filename_labelling;
        while( stream >> x1 >> y1 >> x2 >> y2 >> delim )
        {
            filename_labelling.push_back(Rect(x1, y1, x2, y2));
        }
        labelling.push_back(filename_labelling);
        filename_labelling.clear();
    }
    return true;
}


int main(int argc, char *argv[])
{
    if( argc == 1 )
    {
        printf("Usage: %s OPTIONS, where OPTIONS are:\n"
               "\n"
               "--path <path> - path to dir with data and labelling\n"
               "    (labelling should have name gt.txt)\n"
               "\n"
               "--feature_count <count> - number of features to generate\n"
               "\n"
               "--weak_count <count> - number of weak classifiers in cascade\n"
               "\n"
               "--model_size <rowsxcols> - model size in pixels\n"
               "\n"
               "--overlap <measure> - number from [0, 1], means maximum\n"
               "    overlap with objects while sampling background\n"
               "\n"
               "--model_filename <path> - filename for saving model\n",
               argv[0]);
        return 0;
    }

    string path, model_path;
    ICFDetectorParams params;
    for( int i = 1; i < argc; ++i )
    {
        if( !strcmp("--path", argv[i]) )
        {
            i += 1;
            path = argv[i];
        }
        else if( !strcmp("--feature_count", argv[i]) )
        {
            i += 1;
            if( !read_pos_int(argv[i], &params.feature_count) )
            {
                fprintf(stderr, "Error reading feature count from `%s`\n",
                        argv[i]);
                return 1;
            }
        }
        else if( !strcmp("--weak_count", argv[i]) )
        {
            i += 1;
            if( !read_pos_int(argv[i], &params.weak_count) )
            {
                fprintf(stderr, "Error reading weak count from `%s`\n",
                        argv[i]);
                return 1;
            }
        }
        else if( !strcmp("--model_size", argv[i]) )
        {
            i += 1;
            if( !read_model_size(argv[i], &params.model_n_rows,
                                  &params.model_n_cols) )
            {
                fprintf(stderr, "Error reading model size from `%s`\n",
                        argv[i]);
                return 1;
            }
        }
        else if( !strcmp("--overlap", argv[i]) )
        {
            i += 1;
            if( !read_overlap(argv[i], &params.overlap) )
            {
                fprintf(stderr, "Error reading overlap from `%s`\n",
                        argv[i]);
                return 1;
            }
        }
        else if( !strcmp("--model_filename", argv[i]) )
        {
            i += 1;
            model_path = argv[i];
        }
        else
        {
            fprintf(stderr, "Error: unknown argument `%s`\n", argv[i]);
            return 1;
        }

    }

    try
    {
        ICFDetector detector;
        vector<string> filenames;
        vector< vector<Rect> > labelling;
        read_labelling(path, filenames, labelling);

        detector.train(filenames, labelling, params);
    }
    catch( const char *err )
    {
        fprintf(stderr, "%s", err);
    }
    catch( ... )
    {
        fprintf(stderr, "Unknown error\n");
    }
}
