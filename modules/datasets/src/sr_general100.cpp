// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "opencv2/datasets/sr_general100.hpp"
#include "opencv2/datasets/util.hpp"

namespace cv
{
namespace datasets
{

using namespace std;

class SR_general100Imp CV_FINAL : public SR_general100
{
public:
    SR_general100Imp() {}
    //SR_general100Imp(const string &path);
    virtual ~SR_general100Imp() {}

    virtual void load(const string &path) CV_OVERRIDE;

private:
    void loadDataset(const string &path);
};

/*SR_general100Imp::SR_general100Imp(const string &path)
{
    loadDataset(path);
}*/

void SR_general100Imp::load(const string &path)
{
    loadDataset(path);
}

void SR_general100Imp::loadDataset(const string &path)
{
    train.push_back(vector< Ptr<Object> >());
    test.push_back(vector< Ptr<Object> >());
    validation.push_back(vector< Ptr<Object> >());

    vector<string> fileNames;
    getDirList(path, fileNames);
    for (vector<string>::iterator it=fileNames.begin(); it!=fileNames.end(); ++it)
    {
        string &imageName = *it;
        Ptr<SR_general100Obj> curr(new SR_general100Obj);
        curr->imageName = imageName;
        train.back().push_back(curr);

    }
}

Ptr<SR_general100> SR_general100::create()
{
    return Ptr<SR_general100Imp>(new SR_general100Imp);
}

}
}