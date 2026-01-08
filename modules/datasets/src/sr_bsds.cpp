// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "opencv2/datasets/sr_bsds.hpp"
#include "opencv2/datasets/util.hpp"

namespace cv
{
namespace datasets
{

using namespace std;

class SR_bsdsImp CV_FINAL : public SR_bsds
{
public:
    SR_bsdsImp() {}
    //SR_bsdsImp(const string &path);
    virtual ~SR_bsdsImp() {}

    virtual void load(const string &path) CV_OVERRIDE;

private:
    void loadDataset(const string &path);

    void loadDatasetPart(const string &fileName, vector< Ptr<Object> > &dataset_);
};

void SR_bsdsImp::loadDatasetPart(const string &fileName, vector< Ptr<Object> > &dataset_)
{
    ifstream infile(fileName.c_str());
    string imageName;
    while (infile >> imageName)
    {
        Ptr<SR_bsdsObj> curr(new SR_bsdsObj);
        curr->imageName = imageName;
        dataset_.push_back(curr);
    }
}

/*SR_bsdsImp::SR_bsdsImp(const string &path)
{
    loadDataset(path);
}*/

void SR_bsdsImp::load(const string &path)
{
    loadDataset(path);
}

void SR_bsdsImp::loadDataset(const string &path)
{
    train.push_back(vector< Ptr<Object> >());
    test.push_back(vector< Ptr<Object> >());
    validation.push_back(vector< Ptr<Object> >());

    string trainName(path + "iids_train.txt");
    string testName(path + "iids_test.txt");

    // loading train
    loadDatasetPart(trainName, train.back());

    // loading test
    loadDatasetPart(testName, test.back());
}

Ptr<SR_bsds> SR_bsds::create()
{
    return Ptr<SR_bsdsImp>(new SR_bsdsImp);
}

}
}