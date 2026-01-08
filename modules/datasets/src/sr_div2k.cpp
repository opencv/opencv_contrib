// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "opencv2/datasets/sr_div2k.hpp"
#include "opencv2/datasets/util.hpp"

namespace cv
{
namespace datasets
{

using namespace std;

class SR_div2kImp CV_FINAL : public SR_div2k
{
public:
    SR_div2kImp() {}
    //SR_div2kImp(const string &path);
    virtual ~SR_div2kImp() {}

    virtual void load(const string &path) CV_OVERRIDE;

private:
    void loadDataset(const string &path);
};

/*SR_div2kImp::SR_div2kImp(const string &path)
{
    loadDataset(path);
}*/

void SR_div2kImp::load(const string &path)
{
    loadDataset(path);
}

void SR_div2kImp::loadDataset(const string &path)
{
    train.push_back(vector< Ptr<Object> >());
    test.push_back(vector< Ptr<Object> >());
    validation.push_back(vector< Ptr<Object> >());

    vector<string> fileNames;
    getDirList(path, fileNames);
    for (vector<string>::iterator it=fileNames.begin(); it!=fileNames.end(); ++it)
    {
        string &imageName = *it;
        Ptr<SR_div2kObj> curr(new SR_div2kObj);
        curr->imageName = imageName;
        train.back().push_back(curr);

    }
}

Ptr<SR_div2k> SR_div2k::create()
{
    return Ptr<SR_div2kImp>(new SR_div2kImp);
}

}
}