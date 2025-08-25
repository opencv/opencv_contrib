// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "opencv2/datasets/saliency_mit1003.hpp"
#include "opencv2/datasets/util.hpp"

namespace cv
{
namespace datasets
{

using namespace std;

class SALIENCY_mit1003Imp : public SALIENCY_mit1003
{
public:
    SALIENCY_mit1003Imp() {}
    virtual ~SALIENCY_mit1003Imp() {}

    virtual void load(const string &path);
    virtual vector<vector<Mat> > getDataset();

private:
    void loadDataset(const string &path);
};

void SALIENCY_mit1003Imp::load(const string &path)
{
    loadDataset(path);
}

void SALIENCY_mit1003Imp::loadDataset(const string &path)
{
    train.push_back( vector< Ptr<Object> >() );
    test.push_back( vector< Ptr<Object> >() );
    validation.push_back( vector< Ptr<Object> >() );

    string imgPath( path + "/ALLSTIMULI/" );
    string fixPath( path + "/ALLFIXATIONMAPS/" );

    vector<string> imgNames;

    getDirList( imgPath, imgNames );
    for ( unsigned i = 0; i < imgNames.size(); i++ )
    {
        Ptr<SALIENCY_mit1003obj> curr( new SALIENCY_mit1003obj );
        curr->name = imgNames[i].substr( 0, imgNames[i].find_first_of('.') );
        curr->id = i;
        curr->img = imread( imgPath + curr->name + ".jpeg" );
        curr->fixMap = imread( fixPath + curr->name + "_fixMap.jpg", 0 );
        curr->fixPts = imread( fixPath + curr->name + "_fixPts.jpg", 0 );
        if ( curr->img.empty() || curr->fixMap.empty() || curr->fixPts.empty() ) continue;
        train.back().push_back(curr);
        test.back().push_back(curr);
        validation.back().push_back(curr);
    }

}

Ptr<SALIENCY_mit1003> SALIENCY_mit1003::create()
{
    return Ptr<SALIENCY_mit1003Imp>(new SALIENCY_mit1003Imp);
}

vector<vector<Mat> > SALIENCY_mit1003Imp::getDataset()
{
    vector<vector<Mat> > res = vector<vector<Mat> >( 3, vector<Mat>() );
    for ( unsigned i = 0; i < train.size() ;i++ )
    {
        for ( unsigned j = 0; j < train[i].size() ;j++ )
        {
            Ptr<SALIENCY_mit1003obj> curr(static_cast<SALIENCY_mit1003obj *>(train[i][j].get()));
            res[0].push_back( curr->img );
            res[1].push_back( curr->fixMap );
            res[2].push_back( curr->fixPts );
        }
    }
    return res;
}

}
}
