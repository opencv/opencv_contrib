// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "precomp.hpp"
#include "face_alignmentimpl.hpp"

using namespace std;

namespace cv{
namespace face{
//Threading helper classes
class doSum : public ParallelLoopBody
{
    public:
        doSum(vector<training_sample>* samples_,vector<Point2f>* sum_) :
        samples(samples_),
        sum(sum_)
        {
        }
        virtual void operator()( const Range& range) const
        {
            for (int j = range.start; j < range.end; ++j){
                for(unsigned long k=0;k<(*samples)[j].shapeResiduals.size();k++){
                    (*sum)[k]=(*sum)[k]+(*samples)[j].shapeResiduals[k];
                }
            }
        }
    private:
        vector<training_sample>* samples;
        vector<Point2f>* sum;
};
class modifySamples : public ParallelLoopBody
{
    public:
        modifySamples(vector<training_sample>* samples_,vector<Point2f>* temp_) :
        samples(samples_),
        temp(temp_)
        {
        }
        virtual void operator()( const Range& range) const
        {
            for (int j = range.start; j < range.end; ++j){
                for(unsigned long k=0;k<(*samples)[j].shapeResiduals.size();k++){
                    (*samples)[j].shapeResiduals[k]=(*samples)[j].shapeResiduals[k]-(*temp)[k];
                    (*samples)[j].current_shape[k]=(*samples)[j].actual_shape[k]-(*samples)[j].shapeResiduals[k];
                }
            }
        }
    private:
        vector<training_sample>* samples;
        vector<Point2f>* temp;
};
class splitSamples : public ParallelLoopBody
{
    public:
        splitSamples(vector<training_sample>* samples_,vector< vector<Point2f> >* leftsumresiduals_,vector<unsigned long>* left_count_,unsigned long* num_test_splits_,vector<splitr>* feats_) :
        samples(samples_),
        leftsumresiduals(leftsumresiduals_),
        left_count(left_count_),
        num_test_splits(num_test_splits_),
        feats(feats_)
        {
        }
        virtual void operator()( const Range& range) const
        {
            for (int i = range.start; i < range.end; ++i){
                for(unsigned long j=0;j<*(num_test_splits);j++){
                    (*left_count)[j]++;
                    if ((float)(*samples)[i].pixel_intensities[(unsigned long)(*feats)[j].index1] - (float)(*samples)[i].pixel_intensities[(unsigned long)(*feats)[j].index2] > (*feats)[j].thresh){
                        for(unsigned long k=0;k<(*samples)[i].shapeResiduals.size();k++){
                            (*leftsumresiduals)[j][k]=(*leftsumresiduals)[j][k]+(*samples)[i].shapeResiduals[k];
                        }
                    }
                }
            }
        }
    private:
        vector<training_sample>* samples;
        vector< vector<Point2f> >* leftsumresiduals;
        vector<unsigned long>* left_count;
        unsigned long* num_test_splits;
        vector<splitr>* feats;
};
splitr FacemarkKazemiImpl::getTestSplits(vector<Point2f> pixel_coordinates,int seed)
{
    splitr feat;
    //generates splits whose probability is above a particular threshold.
    //P(u,v)=e^(-distance/lambda) as described in the research paper
    //cited above. This helps to select closer pixels hence make efficient
    //splits.
    double probability;
    double check;
    RNG rng(seed);
    do
    {
        //select random pixel coordinate
        feat.index1   = rng.uniform(0,params.num_test_coordinates);
        //select another random coordinate
        feat.index2   = rng.uniform(0,params.num_test_coordinates);
        Point2f pt = pixel_coordinates[(unsigned long)feat.index1]-pixel_coordinates[(unsigned long)feat.index2];
        double distance = sqrt((pt.x*pt.x)+(pt.y*pt.y));
        //calculate the probability
        probability = exp(-distance/params.lambda);
        check = rng.uniform(double(0),double(1));
    }
    while(check>probability||feat.index1==feat.index2);
    feat.thresh =(float)(((rng.uniform(double(0),double(1)))*256 - 128)/2.0);
    return feat;
}
bool FacemarkKazemiImpl:: getBestSplit(vector<Point2f> pixel_coordinates, vector<training_sample>& samples,unsigned long start ,
                                        unsigned long end,splitr& split,vector< vector<Point2f> >& sum,long node_no)
{
    if(samples[0].shapeResiduals.size()!=samples[0].current_shape.size()){
        String error_message = "Error while generating split.Residuals are not complete.Aborting....";
        CV_ErrorNoReturn(Error::StsBadArg, error_message);
        return false;
    }
    //This vector stores the matrices where each matrix represents
    //sum of the residuals of shapes of samples which go to the left
    //child after split
    vector< vector<Point2f> > leftsumresiduals;
    leftsumresiduals.resize(params.num_test_splits);
    vector<splitr> feats;
    //generate random splits and selects the best split amongst them.
    for (unsigned long i = 0; i < params.num_test_splits; ++i){
        feats.push_back(getTestSplits(pixel_coordinates,i+(int)time(0)));
        leftsumresiduals[i].resize(samples[0].shapeResiduals.size());
    }
    vector<unsigned long> left_count;
    left_count.resize(params.num_test_splits);
    parallel_for_(Range(start,end),splitSamples(&samples,&leftsumresiduals,&left_count,&params.num_test_splits,&feats));
    //Selecting the best split
    double best_score =-1;
    unsigned long best_feat = 0;
    double score = -1;
    vector<Point2f> right_sum;
    right_sum.resize(sum[node_no].size());
    vector<Point2f> left_sum;
    left_sum.resize(sum[node_no].size());
    unsigned long right_cnt;
    for(unsigned long i=0;i<leftsumresiduals.size();i++){
        right_cnt = (end-start+1)-left_count[i];
        for(unsigned long k=0;k<leftsumresiduals[i].size();k++){
            if (right_cnt!=0){
                right_sum[k].x=(sum[node_no][k].x-leftsumresiduals[i][k].x)/right_cnt;
                right_sum[k].y=(sum[node_no][k].y-leftsumresiduals[i][k].y)/right_cnt;
            }
            else
                right_sum[k]=Point2f(0,0);
            if(left_count[i]!=0){
                left_sum[k].x=leftsumresiduals[i][k].x/left_count[i];
                left_sum[k].y=leftsumresiduals[i][k].y/left_count[i];
            }
            else
                left_sum[k]=Point2f(0,0);
        }
        Point2f pt1(0,0);
        Point2f pt2(0,0);
        for(unsigned long k=0;k<left_sum.size();k++){
            pt1.x = pt1.x + (float)(left_sum[k].x*left_sum[k].x);
            pt2.x = pt2.x + (float)(right_sum[k].x*right_sum[k].x);
            pt1.y = pt1.y + (float)(left_sum[k].y*left_sum[k].y);
            pt2.y = pt2.y + (float)(right_sum[k].y*right_sum[k].y);
        }
        score = (double)sqrt(pt1.x+pt1.y)*(double)left_count[i] + (double)sqrt(pt2.x+pt2.y)*(double)right_cnt;
        if(score > best_score){
            best_score = score;
            best_feat = i;
        }
    }
    sum[2*node_no+1] = leftsumresiduals[best_feat];
    sum[2*node_no+2].resize(sum[node_no].size());
    for(unsigned long k=0;k<sum[node_no].size();k++){
        sum[2*node_no+2][k].x = sum[node_no][k].x-sum[2*node_no+1][k].x;
        sum[2*node_no+2][k].y = sum[node_no][k].y-sum[2*node_no+1][k].y;
    }
    split = feats[best_feat];
    return true;
}
void FacemarkKazemiImpl::createSplitNode(regtree& tree, splitr split,long node_no){
    tree_node node;
    node.split = split;
    node.leaf.clear();
    tree.nodes[node_no]=node;
}
void FacemarkKazemiImpl::createLeafNode(regtree& tree,long node_no,vector<Point2f> assign){
    tree_node node;
    node.split.index1 = (uint64_t)(-1);
    node.split.index2 = (uint64_t)(-1);
    node.leaf = assign;
    tree.nodes[node_no] = node;
}
bool FacemarkKazemiImpl :: generateSplit(queue<node_info>& curr,vector<Point2f> pixel_coordinates, vector<training_sample>& samples,
                                        splitr &split , vector< vector<Point2f> >& sum){

    long start = curr.front().index1;
    long end = curr.front().index2;
    long _depth = curr.front().depth;
    long node_no =curr.front().node_no;
    curr.pop();
    if(start == end)
        return false;
    getBestSplit(pixel_coordinates,samples,start,end,split,sum,node_no);
    long mid = divideSamples(split, samples, start, end);
    //cout<<mid<<endl;
    if(mid==start||mid==end+1)
        return false;
    node_info _left,_right;
    _left.index1 = start;
    _left.index2 = mid-1;
    _left.depth = _depth +1;
    _left.node_no = 2*node_no+1;
    _right.index1 = mid;
    _right.index2 = end;
    _right.depth = _depth +1;
    _right.node_no = 2*node_no+2;
    curr.push(_left);
    curr.push(_right);
    return true;
}
bool FacemarkKazemiImpl :: buildRegtree(regtree& tree,vector<training_sample>& samples,vector<Point2f> pixel_coordinates){
    if(samples.size()==0){
        String error_message = "Error while building regression tree.Empty samples. Aborting....";
        CV_ErrorNoReturn(Error::StsBadArg, error_message);
        return false;
    }
    if(pixel_coordinates.size()==0){
        String error_message = "Error while building regression tree.No pixel coordinates. Aborting....";
        CV_ErrorNoReturn(Error::StsBadArg, error_message);
        return false;
    }
    queue<node_info> curr;
    node_info parent;
    vector< vector<Point2f> > sum;
    const long numNodes =(long)pow(2,params.tree_depth);
    const long numSplitNodes = numNodes/2 - 1;
    sum.resize(numNodes+1);
    sum[0].resize(samples[0].shapeResiduals.size());
    parallel_for_(cv::Range(0,(int)samples.size()), doSum(&(samples),&(sum[0])));
    parent.index1=0;
    parent.index2=(long)samples.size()-1;
    parent.node_no=0;
    parent.depth=0;
    curr.push(parent);
    tree.nodes.resize(numNodes+1);
    //Total number of split nodes
    while(!curr.empty()){
        pair<long,long> range= make_pair(curr.front().index1,curr.front().index2);
        long node_no = curr.front().node_no;
        splitr split;
        //generate a split
        if(node_no<=numSplitNodes){
            if(generateSplit(curr,pixel_coordinates,samples,split,sum)){
                createSplitNode(tree,split,node_no);
            }
        //create leaf
            else{
                long count = range.second-range.first +1;
                vector<Point2f> temp;
                temp.resize(samples[range.first].shapeResiduals.size());
                parallel_for_(Range(range.first, range.second), doSum(&(samples),&(temp)));
                for(unsigned long k=0;k<temp.size();k++){
                    temp[k].x=(temp[k].x/count)*params.learning_rate;
                    temp[k].y=(temp[k].y/count)*params.learning_rate;
                }
                // Modify current shape according to the weak learners.
                parallel_for_(Range(range.first,range.second), modifySamples(&(samples),&(temp)));
                createLeafNode(tree,node_no,temp);
            }
        }
        else
        {
            unsigned long count = range.second-range.first +1;
            vector<Point2f> temp;
            temp.resize(samples[range.first].shapeResiduals.size());
            parallel_for_(Range(range.first, range.second), doSum(&(samples),&(temp)));
            for(unsigned long k=0;k<temp.size();k++){
                temp[k].x=(temp[k].x/count)*params.learning_rate;
                temp[k].y=(temp[k].y/count)*params.learning_rate;
            }
            // Modify current shape according to the weak learners.
            parallel_for_(Range(range.first,range.second), modifySamples(&(samples),&(temp)));
            createLeafNode(tree,node_no,temp);
            curr.pop();
        }
    }
    return true;
}
unsigned long FacemarkKazemiImpl::divideSamples (splitr split,vector<training_sample>& samples,unsigned long start,unsigned long end)
{
    if(samples.size()==0){
        String error_message = "Error while dividing samples. Sample array empty. Aborting....";
        CV_ErrorNoReturn(Error::StsBadArg, error_message);
        return 0;
    }
    unsigned long i = start;
    training_sample temp;
    //partition samples according to the split
    for (unsigned long j = start; j < end; ++j)
    {
        if ((float)samples[j].pixel_intensities[(unsigned long)split.index1] - (float)samples[j].pixel_intensities[(unsigned long)split.index2] > split.thresh)
        {
            temp=samples[i];
            samples[i]=samples[j];
            samples[j]=temp;
            ++i;
        }
    }
    return i;
}
}//cv
}//face