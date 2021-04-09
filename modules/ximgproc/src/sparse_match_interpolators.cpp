/*
 *  By downloading, copying, installing or using the software you agree to this license.
 *  If you do not agree to this license, do not download, install,
 *  copy or use the software.
 *
 *
 *  License Agreement
 *  For Open Source Computer Vision Library
 *  (3 - clause BSD License)
 *
 *  Redistribution and use in source and binary forms, with or without modification,
 *  are permitted provided that the following conditions are met :
 *
 *  * Redistributions of source code must retain the above copyright notice,
 *  this list of conditions and the following disclaimer.
 *
 *  * Redistributions in binary form must reproduce the above copyright notice,
 *  this list of conditions and the following disclaimer in the documentation
 *  and / or other materials provided with the distribution.
 *
 *  * Neither the names of the copyright holders nor the names of the contributors
 *  may be used to endorse or promote products derived from this software
 *  without specific prior written permission.
 *
 *  This software is provided by the copyright holders and contributors "as is" and
 *  any express or implied warranties, including, but not limited to, the implied
 *  warranties of merchantability and fitness for a particular purpose are disclaimed.
 *  In no event shall copyright holders or contributors be liable for any direct,
 *  indirect, incidental, special, exemplary, or consequential damages
 *  (including, but not limited to, procurement of substitute goods or services;
 *  loss of use, data, or profits; or business interruption) however caused
 *  and on any theory of liability, whether in contract, strict liability,
 *  or tort(including negligence or otherwise) arising in any way out of
 *  the use of this software, even if advised of the possibility of such damage.
 */

#include "precomp.hpp"
#include "opencv2/ximgproc/sparse_match_interpolator.hpp"
#include "opencv2/video.hpp"
#include <algorithm>
#include <vector>
#include "opencv2/core/hal/hal.hpp"

using namespace std;

#define INF 1E+20F

namespace cv {
namespace ximgproc {

struct SparseMatch
{
    Point2f reference_image_pos;
    Point2f target_image_pos;
    SparseMatch(){}
    SparseMatch(Point2f ref_point, Point2f target_point): reference_image_pos(ref_point), target_image_pos(target_point) {}
};

bool operator<(const SparseMatch& lhs,const SparseMatch& rhs);

static void computeGradientMagnitude(Mat& src, Mat& dst);
static void weightedLeastSquaresAffineFit(int* labels, float* weights, int count, float lambda, const SparseMatch* matches, Mat& dst);
static void generateHypothesis(int* labels, int count, RNG& rng, unsigned char* is_used, SparseMatch* matches, Mat& dst);
static void verifyHypothesis(int* labels, float* weights, int count, SparseMatch* matches, float eps, float lambda, Mat& hypothesis_transform, Mat& old_transform, float& old_weighted_num_inliers);

struct node
{
    float dist;
    int label;
    node() {}
    node(int l,float d): dist(d), label(l) {}
};



class EdgeAwareInterpolatorImpl CV_FINAL : public EdgeAwareInterpolator
{
public:
    static Ptr<EdgeAwareInterpolatorImpl> create();
    void interpolate(InputArray from_image, InputArray from_points, InputArray to_image, InputArray to_points, OutputArray dense_flow) CV_OVERRIDE;

protected:
    int match_num;
    int w, h;
    //internal buffers:
    vector<node>* g;
    Mat NNlabels;
    Mat NNdistances;
    Mat labels;
    Mat costMap;
    //tunable parameters:
    float lambda;
    int k;
    float sigma;
    bool use_post_proc;
    float fgs_lambda;
    float fgs_sigma;

    // static parameters:
    static const int ransac_interpolation_num_iter = 1;
    static const int distance_transform_num_iter = 1;
    float regularization_coef;
    static const int ransac_num_stripes = 4;
    RNG rngs[ransac_num_stripes];

    void init();
    void preprocessData(Mat& src, vector<SparseMatch>& matches);
    void geodesicDistanceTransform(Mat& distances, Mat& cost_map);
    void buildGraph(Mat& distances, Mat& cost_map);
    void ransacInterpolation(vector<SparseMatch>& matches, Mat& dst_dense_flow);

protected:
    struct GetKNNMatches_ParBody : public ParallelLoopBody
    {
        EdgeAwareInterpolatorImpl* inst;
        int num_stripes;
        int stripe_sz;

        GetKNNMatches_ParBody(EdgeAwareInterpolatorImpl& _inst, int _num_stripes);
        void operator () (const Range& range) const CV_OVERRIDE;
    };

    struct RansacInterpolation_ParBody : public ParallelLoopBody
    {
        EdgeAwareInterpolatorImpl* inst;
        Mat* transforms;
        float* weighted_inlier_nums;
        float* eps;
        SparseMatch* matches;
        int num_stripes;
        int stripe_sz;
        int inc;

        RansacInterpolation_ParBody(EdgeAwareInterpolatorImpl& _inst, Mat* _transforms, float* _weighted_inlier_nums, float* _eps, SparseMatch* _matches, int _num_stripes, int _inc);
        void operator () (const Range& range) const CV_OVERRIDE;
    };

public:
    void setCostMap(const Mat & _costMap) CV_OVERRIDE { _costMap.copyTo(costMap); }
    void  setK(int _k) CV_OVERRIDE {k=_k;}
    int   getK() CV_OVERRIDE {return k;}
    void  setSigma(float _sigma) CV_OVERRIDE {sigma=_sigma;}
    float getSigma() CV_OVERRIDE {return sigma;}
    void  setLambda(float _lambda) CV_OVERRIDE {lambda=_lambda;}
    float getLambda() CV_OVERRIDE {return lambda;}
    void  setUsePostProcessing(bool _use_post_proc) CV_OVERRIDE {use_post_proc=_use_post_proc;}
    bool  getUsePostProcessing() CV_OVERRIDE {return use_post_proc;}
    void  setFGSLambda(float _lambda) CV_OVERRIDE {fgs_lambda=_lambda;}
    float getFGSLambda() CV_OVERRIDE {return fgs_lambda;}
    void  setFGSSigma(float _sigma) CV_OVERRIDE {fgs_sigma = _sigma;}
    float getFGSSigma() CV_OVERRIDE {return fgs_sigma;}
};

void EdgeAwareInterpolatorImpl::init()
{
    lambda        = 999.0f;
    k             = 128;
    sigma         = 0.05f;
    use_post_proc = true;
    fgs_lambda    = 500.0f;
    fgs_sigma     = 1.5f;
    regularization_coef = 0.01f;
    costMap = Mat();
}

Ptr<EdgeAwareInterpolatorImpl> EdgeAwareInterpolatorImpl::create()
{
    EdgeAwareInterpolatorImpl *eai = new EdgeAwareInterpolatorImpl();
    eai->init();
    return Ptr<EdgeAwareInterpolatorImpl>(eai);
}

void EdgeAwareInterpolatorImpl::interpolate(InputArray from_image, InputArray from_points, InputArray, InputArray to_points, OutputArray dense_flow)
{
    CV_Assert( !from_image.empty() && (from_image.depth() == CV_8U) && (from_image.channels() == 3 || from_image.channels() == 1) );
    CV_Assert( !from_points.empty() && !to_points.empty() && from_points.sameSize(to_points) );
    CV_Assert((from_points.isVector() || from_points.isMat()) && from_points.depth() == CV_32F);
    CV_Assert((to_points.isVector() || to_points.isMat()) && to_points.depth() == CV_32F);
    CV_Assert(from_points.sameSize(to_points));

    w = from_image.cols();
    h = from_image.rows();

    Mat from_mat = from_points.getMat();
    Mat to_mat = to_points.getMat();
    int npoints = from_mat.checkVector(2, CV_32F, false);
    if (from_mat.channels() != 2)
        from_mat = from_mat.reshape(2, npoints);

    if (to_mat.channels() != 2){
        to_mat = to_mat.reshape(2, npoints);
        npoints = from_mat.checkVector(2, CV_32F, false);
    }


    vector<SparseMatch> matches_vector(npoints);
    for(unsigned int i=0;i<matches_vector.size();i++)
        matches_vector[i] = SparseMatch(from_mat.at<Point2f>(i), to_mat.at<Point2f>(i));


    sort(matches_vector.begin(),matches_vector.end());

    match_num = (int)matches_vector.size();
    CV_Assert(match_num<SHRT_MAX);

    Mat src = from_image.getMat();
    labels = Mat(h,w,CV_32S);
    labels = Scalar(-1);
    NNlabels = Mat(match_num,k,CV_32S);
    NNlabels = Scalar(-1);
    NNdistances = Mat(match_num,k,CV_32F);
    NNdistances = Scalar(0.0f);
    g = new vector<node>[match_num];

    preprocessData(src,matches_vector);

    dense_flow.create(from_image.size(),CV_32FC2);
    Mat dst = dense_flow.getMat();
    ransacInterpolation(matches_vector,dst);
    if(use_post_proc)
        fastGlobalSmootherFilter(src,dst,dst,fgs_lambda,fgs_sigma);

    costMap.release();
    delete[] g;
}

void EdgeAwareInterpolatorImpl::preprocessData(Mat& src, vector<SparseMatch>& matches)
{
    Mat distances(h,w,CV_32F);
    distances = Scalar(INF);

    int x,y;
    for(unsigned int i=0;i<matches.size();i++)
    {
        x = min((int)(matches[i].reference_image_pos.x+0.5f),w-1);
        y = min((int)(matches[i].reference_image_pos.y+0.5f),h-1);

        distances.at<float>(y,x) = 0.0f;
        labels.at<int>(y,x) = (int)i;
    }

    if (costMap.empty())
    {
        costMap.create(h, w, CV_32FC1);
        computeGradientMagnitude(src, costMap);
    }
    else
        CV_Assert(costMap.cols == w && costMap.rows == h);
    costMap = (1000.0f-lambda) + lambda* costMap;
    geodesicDistanceTransform(distances, costMap);
    buildGraph(distances, costMap);
    parallel_for_(Range(0,getNumThreads()),GetKNNMatches_ParBody(*this,getNumThreads()));
}

void EdgeAwareInterpolatorImpl::geodesicDistanceTransform(Mat& distances, Mat& cost_map)
{
    float c1 = 1.0f / 2.0f;
    float c2 = sqrt(2.0f) / 2.0f;
    float d = 0.0f;
    int i, j;
    float *dist_row, *cost_row;
    float *dist_row_prev, *cost_row_prev;
    int *label_row;
    int *label_row_prev;

#define CHECK(cur_dist,cur_label,cur_cost,prev_dist,prev_label,prev_cost,coef)\
{\
    d = prev_dist + coef*(cur_cost+prev_cost);\
    if(cur_dist>d){\
        cur_dist=d;\
        cur_label = prev_label;}\
}

    for (int it = 0; it < distance_transform_num_iter; it++)
    {
        //first pass (left-to-right, top-to-bottom):
        dist_row = distances.ptr<float>(0);
        label_row = labels.ptr<int>(0);
        cost_row = cost_map.ptr<float>(0);
        for (j = 1; j < w; j++)
            CHECK(dist_row[j], label_row[j], cost_row[j], dist_row[j - 1], label_row[j - 1], cost_row[j - 1], c1);

        for (i = 1; i < h; i++)
        {
            dist_row = distances.ptr<float>(i);
            dist_row_prev = distances.ptr<float>(i - 1);

            label_row = labels.ptr<int>(i);
            label_row_prev = labels.ptr<int>(i - 1);

            cost_row = cost_map.ptr<float>(i);
            cost_row_prev = cost_map.ptr<float>(i - 1);

            j = 0;
            CHECK(dist_row[j], label_row[j], cost_row[j], dist_row_prev[j], label_row_prev[j], cost_row_prev[j], c1);
            CHECK(dist_row[j], label_row[j], cost_row[j], dist_row_prev[j + 1], label_row_prev[j + 1], cost_row_prev[j + 1], c2);
            j++;
            for (; j < w - 1; j++)
            {
                CHECK(dist_row[j], label_row[j], cost_row[j], dist_row[j - 1], label_row[j - 1], cost_row[j - 1], c1);
                CHECK(dist_row[j], label_row[j], cost_row[j], dist_row_prev[j - 1], label_row_prev[j - 1], cost_row_prev[j - 1], c2);
                CHECK(dist_row[j], label_row[j], cost_row[j], dist_row_prev[j], label_row_prev[j], cost_row_prev[j], c1);
                CHECK(dist_row[j], label_row[j], cost_row[j], dist_row_prev[j + 1], label_row_prev[j + 1], cost_row_prev[j + 1], c2);
            }
            CHECK(dist_row[j], label_row[j], cost_row[j], dist_row[j - 1], label_row[j - 1], cost_row[j - 1], c1);
            CHECK(dist_row[j], label_row[j], cost_row[j], dist_row_prev[j - 1], label_row_prev[j - 1], cost_row_prev[j - 1], c2);
            CHECK(dist_row[j], label_row[j], cost_row[j], dist_row_prev[j], label_row_prev[j], cost_row_prev[j], c1);
        }

        //second pass (right-to-left, bottom-to-top):
        dist_row = distances.ptr<float>(h - 1);
        label_row = labels.ptr<int>(h - 1);
        cost_row = cost_map.ptr<float>(h - 1);
        for (j = w - 2; j >= 0; j--)
            CHECK(dist_row[j], label_row[j], cost_row[j], dist_row[j + 1], label_row[j + 1], cost_row[j + 1], c1);

        for (i = h - 2; i >= 0; i--)
        {
            dist_row = distances.ptr<float>(i);
            dist_row_prev = distances.ptr<float>(i + 1);

            label_row = labels.ptr<int>(i);
            label_row_prev = labels.ptr<int>(i + 1);

            cost_row = cost_map.ptr<float>(i);
            cost_row_prev = cost_map.ptr<float>(i + 1);

            j = w - 1;
            CHECK(dist_row[j], label_row[j], cost_row[j], dist_row_prev[j], label_row_prev[j], cost_row_prev[j], c1);
            CHECK(dist_row[j], label_row[j], cost_row[j], dist_row_prev[j - 1], label_row_prev[j - 1], cost_row_prev[j - 1], c2);
            j--;
            for (; j > 0; j--)
            {
                CHECK(dist_row[j], label_row[j], cost_row[j], dist_row[j + 1], label_row[j + 1], cost_row[j + 1], c1);
                CHECK(dist_row[j], label_row[j], cost_row[j], dist_row_prev[j + 1], label_row_prev[j + 1], cost_row_prev[j + 1], c2);
                CHECK(dist_row[j], label_row[j], cost_row[j], dist_row_prev[j], label_row_prev[j], cost_row_prev[j], c1);
                CHECK(dist_row[j], label_row[j], cost_row[j], dist_row_prev[j - 1], label_row_prev[j - 1], cost_row_prev[j - 1], c2);
            }
            CHECK(dist_row[j], label_row[j], cost_row[j], dist_row[j + 1], label_row[j + 1], cost_row[j + 1], c1);
            CHECK(dist_row[j], label_row[j], cost_row[j], dist_row_prev[j + 1], label_row_prev[j + 1], cost_row_prev[j + 1], c2);
            CHECK(dist_row[j], label_row[j], cost_row[j], dist_row_prev[j], label_row_prev[j], cost_row_prev[j], c1);
        }
    }
#undef CHECK
}

void EdgeAwareInterpolatorImpl::buildGraph(Mat& distances, Mat& cost_map)
{
    float *dist_row,      *cost_row;
    float *dist_row_prev, *cost_row_prev;
    int *label_row;
    int *label_row_prev;
    int i,j;
    const float c1 = 1.0f/2.0f;
    const float c2 = sqrt(2.0f)/2.0f;
    float d;
    bool found;

#define CHECK(cur_dist,cur_label,cur_cost,prev_dist,prev_label,prev_cost,coef)\
    if(cur_label!=prev_label)\
    {\
        d = prev_dist + cur_dist + coef*(cur_cost+prev_cost);\
        found = false;\
        for(unsigned int n=0;n<g[prev_label].size();n++)\
        {\
            if(g[prev_label][n].label==cur_label)\
            {\
                g[prev_label][n].dist = min(g[prev_label][n].dist,d);\
                found=true;\
                break;\
            }\
        }\
        if(!found)\
            g[prev_label].push_back(node(cur_label ,d));\
    }

    dist_row  = distances.ptr<float>(0);
    label_row = labels.ptr<int>(0);
    cost_row  = cost_map.ptr<float>(0);
    for(j=1;j<w;j++)
        CHECK(dist_row[j],label_row[j],cost_row[j],dist_row[j-1],label_row[j-1],cost_row[j-1],c1);

    for(i=1;i<h;i++)
    {
        dist_row       = distances.ptr<float>(i);
        dist_row_prev  = distances.ptr<float>(i-1);

        label_row      = labels.ptr<int>(i);
        label_row_prev = labels.ptr<int>(i-1);

        cost_row      = cost_map.ptr<float>(i);
        cost_row_prev = cost_map.ptr<float>(i-1);

        j=0;
        CHECK(dist_row[j],label_row[j],cost_row[j],dist_row_prev[j]  ,label_row_prev[j]  ,cost_row_prev[j]  ,c1);
        CHECK(dist_row[j],label_row[j],cost_row[j],dist_row_prev[j+1],label_row_prev[j+1],cost_row_prev[j+1],c2);
        j++;
        for(;j<w-1;j++)
        {
            CHECK(dist_row[j],label_row[j],cost_row[j],dist_row[j-1]     ,label_row[j-1]     ,cost_row[j-1]     ,c1);
            CHECK(dist_row[j],label_row[j],cost_row[j],dist_row_prev[j-1],label_row_prev[j-1],cost_row_prev[j-1],c2);
            CHECK(dist_row[j],label_row[j],cost_row[j],dist_row_prev[j]  ,label_row_prev[j]  ,cost_row_prev[j]  ,c1);
            CHECK(dist_row[j],label_row[j],cost_row[j],dist_row_prev[j+1],label_row_prev[j+1],cost_row_prev[j+1],c2);
        }
        CHECK(dist_row[j],label_row[j],cost_row[j],dist_row[j-1]     ,label_row[j-1]     ,cost_row[j-1]     ,c1);
        CHECK(dist_row[j],label_row[j],cost_row[j],dist_row_prev[j-1],label_row_prev[j-1],cost_row_prev[j-1],c2);
        CHECK(dist_row[j],label_row[j],cost_row[j],dist_row_prev[j]  ,label_row_prev[j]  ,cost_row_prev[j]  ,c1);
    }
#undef CHECK

    // force equal distances in both directions:
    node* neighbors;
    for(i=0;i<match_num;i++)
    {
        if(g[i].empty())
            continue;
        neighbors = &g[i].front();
        for(j=0;j<(int)g[i].size();j++)
        {
            found = false;

            for(unsigned int n=0;n<g[neighbors[j].label].size();n++)
            {
                if(g[neighbors[j].label][n].label==i)
                {
                    neighbors[j].dist = g[neighbors[j].label][n].dist = min(neighbors[j].dist,g[neighbors[j].label][n].dist);
                    found = true;
                    break;
                }
            }

            if(!found)
                g[neighbors[j].label].push_back(node((int)i,neighbors[j].dist));
        }
    }
}

struct nodeHeap
{
    // start indexing from 1 (root)
    // children: 2*i, 2*i+1
    // parent: i>>1
    node* heap;
    int* heap_pos;
    node tmp_node;
    int size;
    int num_labels;

    nodeHeap(int _num_labels)
    {
        num_labels = _num_labels;
        heap = new node[num_labels+1];
        heap[0] = node(-1,-1.0f);
        heap_pos = new int[num_labels];
        memset(heap_pos,0,sizeof(int)*num_labels);
        size=0;
    }

    ~nodeHeap()
    {
        delete[] heap;
        delete[] heap_pos;
    }

    void clear()
    {
        size=0;
        memset(heap_pos,0,sizeof(int)*num_labels);
    }

    inline bool empty()
    {
        return (size==0);
    }

    inline void nodeSwap(int idx1, int idx2)
    {
        heap_pos[heap[idx1].label] = idx2;
        heap_pos[heap[idx2].label] = idx1;

        tmp_node   = heap[idx1];
        heap[idx1] = heap[idx2];
        heap[idx2] = tmp_node;
    }

    void add(node n)
    {
        size++;
        heap[size] = n;
        heap_pos[n.label] = size;
        int i = size;
        int parent_i = i>>1;
        while(heap[i].dist<heap[parent_i].dist)
        {
            nodeSwap(i,parent_i);
            i=parent_i;
            parent_i = i>>1;
        }
    }

    node getMin()
    {
        node res = heap[1];
        heap_pos[res.label] = 0;

        int i=1;
        int left,right;
        while( (left=i<<1) < size )
        {
            right = left+1;
            if(heap[left].dist<heap[right].dist)
            {
                heap[i] = heap[left];
                heap_pos[heap[i].label] = i;
                i = left;
            }
            else
            {
                heap[i] = heap[right];
                heap_pos[heap[i].label] = i;
                i = right;
            }
        }

        if(i==size)
        {
            size--;
            return res;
        }

        heap[i] = heap[size];
        heap_pos[heap[i].label] = i;

        int parent_i = i>>1;
        while(heap[i].dist<heap[parent_i].dist)
        {
            nodeSwap(i,parent_i);
            i=parent_i;
            parent_i = i>>1;
        }

        size--;
        return res;
    }

    //checks if node is already in the heap
    //if not - add it
    //if it is - update it with the min dist of the two
    void updateNode(node n)
    {
        if(heap_pos[n.label])
        {
            int i = heap_pos[n.label];
            heap[i].dist = min(heap[i].dist,n.dist);
            int parent_i = i>>1;
            while(heap[i].dist<heap[parent_i].dist)
            {
                nodeSwap(i,parent_i);
                i=parent_i;
                parent_i = i>>1;
            }
        }
        else
            add(n);
    }
};

EdgeAwareInterpolatorImpl::GetKNNMatches_ParBody::GetKNNMatches_ParBody(EdgeAwareInterpolatorImpl& _inst, int _num_stripes):
inst(&_inst),num_stripes(_num_stripes)
{
    stripe_sz = (int)ceil(inst->match_num/(double)num_stripes);
}

void EdgeAwareInterpolatorImpl::GetKNNMatches_ParBody::operator() (const Range& range) const
{
    int start = std::min(range.start * stripe_sz, inst->match_num);
    int end   = std::min(range.end   * stripe_sz, inst->match_num);
    nodeHeap q((int)inst->match_num);
    int num_expanded_vertices;
    unsigned char* expanded_flag = new unsigned char[inst->match_num];
    node* neighbors;

    for(int i=start;i<end;i++)
    {
        if(inst->g[i].empty())
            continue;

        num_expanded_vertices = 0;
        memset(expanded_flag,0,inst->match_num);
        q.clear();
        q.add(node((int)i,0.0f));
        int* NNlabels_row    = inst->NNlabels.ptr<int>(i);
        float* NNdistances_row = inst->NNdistances.ptr<float>(i);
        while(num_expanded_vertices<inst->k && !q.empty())
        {
            node vert_for_expansion = q.getMin();
            expanded_flag[vert_for_expansion.label] = 1;

            //write the expanded vertex to the dst:
            NNlabels_row[num_expanded_vertices] = vert_for_expansion.label;
            NNdistances_row[num_expanded_vertices] = vert_for_expansion.dist;
            num_expanded_vertices++;

            //update the heap:
            neighbors = &inst->g[vert_for_expansion.label].front();
            for(int j=0;j<(int)inst->g[vert_for_expansion.label].size();j++)
            {
                if(!expanded_flag[neighbors[j].label])
                    q.updateNode(node(neighbors[j].label,vert_for_expansion.dist+neighbors[j].dist));
            }
        }
    }
    delete[] expanded_flag;
}

static void weightedLeastSquaresAffineFit(int* labels, float* weights, int count, float lambda, const SparseMatch* matches, Mat& dst)
{
    double sa[6][6]={{0.}}, sb[6]={0.};
    Mat A (6, 6, CV_64F, &sa[0][0]),
        B (6, 1, CV_64F, sb),
        MM(1, 6, CV_64F);
    Point2f a,b;
    float w;

    for( int i = 0; i < count; i++ )
    {
        a = matches[labels[i]].reference_image_pos;
        b = matches[labels[i]].target_image_pos;
        w = weights[i];

        sa[0][0] += w*a.x*a.x;
        sa[0][1] += w*a.y*a.x;
        sa[0][2] += w*a.x;
        sa[1][1] += w*a.y*a.y;
        sa[1][2] += w*a.y;
        sa[2][2] += w;

        sb[0] += w*a.x*b.x;
        sb[1] += w*a.y*b.x;
        sb[2] += w*b.x;
        sb[3] += w*a.x*b.y;
        sb[4] += w*a.y*b.y;
        sb[5] += w*b.y;
    }
    sa[0][0] += lambda;
    sa[1][1] += lambda;

    sa[3][4] = sa[4][3] = sa[1][0] = sa[0][1];
    sa[3][5] = sa[5][3] = sa[2][0] = sa[0][2];
    sa[4][5] = sa[5][4] = sa[2][1] = sa[1][2];

    sa[3][3] = sa[0][0];
    sa[4][4] = sa[1][1];
    sa[5][5] = sa[2][2];

    sb[0] += lambda;
    sb[4] += lambda;

    solve(A, B, MM, DECOMP_EIG);
    MM.reshape(2,3).convertTo(dst,CV_32F);
}

static void generateHypothesis(int* labels, int count, RNG& rng, unsigned char* is_used, SparseMatch* matches, Mat& dst)
{
    int idx;
    Point2f src_points[3];
    Point2f dst_points[3];
    memset(is_used,0,count);

    // randomly get 3 distinct matches
    idx = rng.uniform(0,count-2);
    is_used[idx] = true;
    src_points[0] = matches[labels[idx]].reference_image_pos;
    dst_points[0] = matches[labels[idx]].target_image_pos;

    idx = rng.uniform(0,count-1);
    if (is_used[idx])
        idx = count-2;
    is_used[idx] = true;
    src_points[1] = matches[labels[idx]].reference_image_pos;
    dst_points[1] = matches[labels[idx]].target_image_pos;

    idx = rng.uniform(0,count);
    if (is_used[idx])
        idx = count-1;
    is_used[idx] = true;
    src_points[2] = matches[labels[idx]].reference_image_pos;
    dst_points[2] = matches[labels[idx]].target_image_pos;

    // compute an affine transform:
    getAffineTransform(src_points,dst_points).convertTo(dst,CV_32F);
}

static void verifyHypothesis(int* labels, float* weights, int count, SparseMatch* matches, float eps, float lambda, Mat& hypothesis_transform, Mat& old_transform, float& old_weighted_num_inliers)
{
    float* tr = hypothesis_transform.ptr<float>(0);
    Point2f a,b;
    float weighted_num_inliers = -lambda*((tr[0]-1)*(tr[0]-1)+tr[1]*tr[1]+tr[3]*tr[3]+(tr[4]-1)*(tr[4]-1));

    for(int i=0;i<count;i++)
    {
        a = matches[labels[i]].reference_image_pos;
        b = matches[labels[i]].target_image_pos;
        if(abs(tr[0]*a.x + tr[1]*a.y + tr[2] - b.x) +
           abs(tr[3]*a.x + tr[4]*a.y + tr[5] - b.y) < eps)
            weighted_num_inliers += weights[i];
    }

    if(weighted_num_inliers>=old_weighted_num_inliers)
    {
        old_weighted_num_inliers = weighted_num_inliers;
        hypothesis_transform.copyTo(old_transform);
    }
}

static void computeGradientMagnitude(Mat& src, Mat& dst)
{
    Mat dx, dy;
    Sobel(src, dx, CV_16S, 1, 0);
    Sobel(src, dy, CV_16S, 0, 1);
    float norm_coef = src.channels() * 4 * 255.0f;

    if (src.channels() == 1)
    {
        for (int i = 0; i < src.rows; i++)
        {
            short* dx_row = dx.ptr<short>(i);
            short* dy_row = dy.ptr<short>(i);
            float* dst_row = dst.ptr<float>(i);

            for (int j = 0; j < src.cols; j++)
                dst_row[j] = ((float)abs(dx_row[j]) + abs(dy_row[j])) / norm_coef;
        }
    }
    else
    {
        for (int i = 0; i < src.rows; i++)
        {
            Vec3s* dx_row = dx.ptr<Vec3s>(i);
            Vec3s* dy_row = dy.ptr<Vec3s>(i);
            float* dst_row = dst.ptr<float>(i);

            for (int j = 0; j < src.cols; j++)
                dst_row[j] = (float)(abs(dx_row[j][0]) + abs(dy_row[j][0]) +
                    abs(dx_row[j][1]) + abs(dy_row[j][1]) +
                    abs(dx_row[j][2]) + abs(dy_row[j][2])) / norm_coef;
        }
    }
}

EdgeAwareInterpolatorImpl::RansacInterpolation_ParBody::RansacInterpolation_ParBody(EdgeAwareInterpolatorImpl& _inst, Mat* _transforms, float* _weighted_inlier_nums, float* _eps, SparseMatch* _matches, int _num_stripes, int _inc):
inst(&_inst), transforms(_transforms), weighted_inlier_nums(_weighted_inlier_nums), eps(_eps), matches(_matches), num_stripes(_num_stripes), inc(_inc)
{
    stripe_sz = (int)ceil(inst->match_num/(double)num_stripes);
}

void EdgeAwareInterpolatorImpl::RansacInterpolation_ParBody::operator() (const Range& range) const
{
    if(range.end>range.start+1)
    {
        for(int n=range.start;n<range.end;n++)
            (*this)(Range(n,n+1));
        return;
    }

    int start = std::min(range.start * stripe_sz, inst->match_num);
    int end   = std::min(range.end   * stripe_sz, inst->match_num);
    if(inc<0)
    {
        int tmp = end;
        end = start-1;
        start = tmp-1;
    }

    int* KNNlabels;
    float* KNNdistances;
    unsigned char* is_used = new unsigned char[inst->k];
    Mat hypothesis_transform;

    int* inlier_labels    = new int[inst->k];
    float* inlier_distances = new float[inst->k];
    float* tr;
    int num_inliers;
    Point2f a,b;

    for(int i=start;i!=end;i+=inc)
    {
        if(inst->g[i].empty())
            continue;

        KNNlabels    = inst->NNlabels.ptr<int>(i);
        KNNdistances = inst->NNdistances.ptr<float>(i);
        if(inc>0) //forward pass
        {
            cv::hal::exp32f(KNNdistances,KNNdistances,inst->k);

            Point2f average = Point2f(0.0f,0.0f);
            for(int j=0;j<inst->k;j++)
                average += matches[KNNlabels[j]].target_image_pos - matches[KNNlabels[j]].reference_image_pos;
            average/=inst->k;
            float average_dist = 0.0;
            Point2f vec;
            for(int j=0;j<inst->k;j++)
            {
                vec = (matches[KNNlabels[j]].target_image_pos - matches[KNNlabels[j]].reference_image_pos);
                average_dist += abs(vec.x-average.x) + abs(vec.y-average.y);
            }
            eps[i] = min(0.5f*(average_dist/inst->k),2.0f);
        }

        for(int it=0;it<inst->ransac_interpolation_num_iter;it++)
        {
            generateHypothesis(KNNlabels,inst->k,inst->rngs[range.start],is_used,matches,hypothesis_transform);
            verifyHypothesis(KNNlabels,KNNdistances,inst->k,matches,eps[i],inst->regularization_coef,hypothesis_transform,transforms[i],weighted_inlier_nums[i]);
        }

        //propagate hypotheses from neighbors:
        node* neighbors = &inst->g[i].front();
        for(int j=0;j<(int)inst->g[i].size();j++)
        {
            if((inc*neighbors[j].label)<(inc*i) && (inc*neighbors[j].label)>=(inc*start)) //already processed this neighbor
                verifyHypothesis(KNNlabels,KNNdistances,inst->k,matches,eps[i],inst->regularization_coef,transforms[neighbors[j].label],transforms[i],weighted_inlier_nums[i]);
        }

        if(inc<0) //backward pass
        {
            // determine inliers and compute a least squares fit:
            tr = transforms[i].ptr<float>(0);
            num_inliers = 0;

            for(int j=0;j<inst->k;j++)
            {
                a = matches[KNNlabels[j]].reference_image_pos;
                b = matches[KNNlabels[j]].target_image_pos;
                if(abs(tr[0]*a.x + tr[1]*a.y + tr[2] - b.x) +
                   abs(tr[3]*a.x + tr[4]*a.y + tr[5] - b.y) < eps[i])
                {
                    inlier_labels[num_inliers]    = KNNlabels[j];
                    inlier_distances[num_inliers] = KNNdistances[j];
                    num_inliers++;
                }
            }

            weightedLeastSquaresAffineFit(inlier_labels,inlier_distances,num_inliers,inst->regularization_coef,matches,transforms[i]);
        }
    }

    delete[] inlier_labels;
    delete[] inlier_distances;
    delete[] is_used;
}

void EdgeAwareInterpolatorImpl::ransacInterpolation(vector<SparseMatch>& matches, Mat& dst_dense_flow)
{
    NNdistances *= (-sigma*sigma);

    Mat* transforms = new Mat[match_num];
    float* weighted_inlier_nums = new float[match_num];
    float* eps = new float[match_num];
    for(int i=0;i<match_num;i++)
        weighted_inlier_nums[i] = -std::numeric_limits<float>::max();

    for(int i=0;i<ransac_num_stripes;i++)
        rngs[i] = RNG(0);

    //forward pass:
    parallel_for_(Range(0,ransac_num_stripes),RansacInterpolation_ParBody(*this,transforms,weighted_inlier_nums,eps,&matches.front(),ransac_num_stripes,1));
    //backward pass:
    parallel_for_(Range(0,ransac_num_stripes),RansacInterpolation_ParBody(*this,transforms,weighted_inlier_nums,eps,&matches.front(),ransac_num_stripes,-1));

    //construct the final piecewise-affine interpolation:
    int* label_row;
    float* tr;
    for(int i=0;i<h;i++)
    {
        label_row = labels.ptr<int>(i);
        Point2f* dst_row = dst_dense_flow.ptr<Point2f>(i);
        for(int j=0;j<w;j++)
        {
            tr = transforms[label_row[j]].ptr<float>(0);
            dst_row[j] = Point2f(tr[0]*j+tr[1]*i+tr[2],tr[3]*j+tr[4]*i+tr[5]) - Point2f((float)j,(float)i);
        }
    }

    delete[] transforms;
    delete[] weighted_inlier_nums;
    delete[] eps;
}

CV_EXPORTS_W
Ptr<EdgeAwareInterpolator> createEdgeAwareInterpolator()
{
    return Ptr<EdgeAwareInterpolator>(EdgeAwareInterpolatorImpl::create());
}

bool operator<(const SparseMatch& lhs,const SparseMatch& rhs)
{
    if((int)(lhs.reference_image_pos.y+0.5f)!=(int)(rhs.reference_image_pos.y+0.5f))
        return (lhs.reference_image_pos.y<rhs.reference_image_pos.y);
    else
        return (lhs.reference_image_pos.x<rhs.reference_image_pos.x);
}

class RICInterpolatorImpl CV_FINAL : public RICInterpolator
{
public:
    static Ptr<RICInterpolatorImpl> create();
    void interpolate(InputArray from_image, InputArray from_points, InputArray to_image, InputArray to_points, OutputArray dense_flow) CV_OVERRIDE;

protected:
    // internal buffers
    int match_num;
    vector<vector<node>> g;
    Mat NNlabels;
    Mat NNdistances;
    Mat labels;
    Mat costMap;
    static const int distance_transform_num_iter = 1;
    float lambda;

    //tunable parameters:
    int max_neighbors;
    float alpha;
    int sp_size;
    int sp_nncnt;
    float sp_ruler;
    int model_iter;
    bool refine_models;
    bool use_variational_refinement;
    float cost_suffix;
    float max_flow;
    bool use_global_smoother_filter;
    float fgs_lambda;
    float fgs_sigma;
    SLICType slic_type;

    void init();
    void buildGraph(Mat& distances, Mat& cost_map);
    void geodesicDistanceTransform(Mat& distances, Mat& cost_map);
    int  overSegmentaion(const Mat & img, Mat & outLabels, const int spSize);
    void superpixelNeighborConstruction(const Mat & labels, int labelCnt, Mat& outNeighbor);
    void superpixelLayoutAnalysis(const Mat & labels, int labelCnt, Mat & outCenterPositions, Mat & outNodeItemLists);
    void findSupportMatches(vector<int> & srcIds, int srcCnt, int supportCnt, Mat & matNN,
        Mat & matNNDis, vector<int> & outSupportIds, vector<float> &  outSupportDis);
    float GetWeightFromDistance(float dis);
    int PropagateModels(int spCnt, Mat & spNN, vector<int> & supportMatchIds, vector<float> & supportMatchDis, int supportCnt,
        const vector<SparseMatch> &inputMatches, Mat & outModels);
    float HypothesisEvaluation(const Mat & inModel, const int * matNodes, const float * matDis, int matCnt, const vector<SparseMatch> & inputMatches, Mat & outInLier);
    int HypothesisGeneration(int* matNodes, int matCnt, const vector<SparseMatch> & inputMatches, Mat & outModel);

public:
    void setCostMap(const Mat & _costMap) CV_OVERRIDE { _costMap.copyTo(costMap); }
    void setK(int val) CV_OVERRIDE { max_neighbors  = val; }
    int  getK() const CV_OVERRIDE { return max_neighbors; }
    void setSuperpixelSize(int val) CV_OVERRIDE { sp_size = val; }
    int  getSuperpixelSize() const CV_OVERRIDE { return sp_size; }
    void setSuperpixelNNCnt(int val) CV_OVERRIDE { sp_nncnt = val; }
    int  getSuperpixelNNCnt() const CV_OVERRIDE { return sp_nncnt; }
    void setSuperpixelRuler(float val) CV_OVERRIDE { sp_ruler = val; }
    float getSuperpixelRuler() const CV_OVERRIDE { return sp_ruler; }
    void setSuperpixelMode(int val) CV_OVERRIDE
    {
        slic_type = static_cast<SLICType>(val);
        CV_Assert(slic_type == SLICO || slic_type == SLIC || slic_type == MSLIC);
    }
    int  getSuperpixelMode() const CV_OVERRIDE { return slic_type; }
    void setAlpha(float val) CV_OVERRIDE { alpha = val; }
    float getAlpha() const CV_OVERRIDE { return alpha; }
    void setModelIter(int val) CV_OVERRIDE { model_iter = val; }
    int  getModelIter() const CV_OVERRIDE { return model_iter; }
    void setRefineModels(bool val) CV_OVERRIDE { refine_models = static_cast<int>(val); }
    bool getRefineModels() const CV_OVERRIDE { return refine_models; }
    void setMaxFlow(float val) CV_OVERRIDE { max_flow = val; }
    float getMaxFlow() const CV_OVERRIDE { return max_flow; }
    void setUseVariationalRefinement(bool val) CV_OVERRIDE { use_variational_refinement = val; }
    bool getUseVariationalRefinement() const CV_OVERRIDE { return use_variational_refinement; }
    void  setUseGlobalSmootherFilter(bool val) CV_OVERRIDE { use_global_smoother_filter = val; }
    bool  getUseGlobalSmootherFilter() const CV_OVERRIDE { return use_global_smoother_filter; }
    void  setFGSLambda(float _lambda) CV_OVERRIDE { fgs_lambda = _lambda; }
    float getFGSLambda() const CV_OVERRIDE { return fgs_lambda; }
    void  setFGSSigma(float _sigma) CV_OVERRIDE { fgs_sigma = _sigma; }
    float getFGSSigma() const CV_OVERRIDE { return fgs_sigma; }
};

Ptr<RICInterpolatorImpl> RICInterpolatorImpl::create()
{
    auto eai = makePtr<RICInterpolatorImpl>();
    eai->init();
    return eai;
}

void RICInterpolatorImpl::init()
{
    max_neighbors = 32;
    alpha = 0.7f;
    lambda = 999.0f;
    sp_size = 15;
    sp_nncnt = 150;
    sp_ruler = 15.f;
    model_iter = 4;
    refine_models = true;
    use_variational_refinement = false;
    cost_suffix = 0.001f;
    max_flow = 250.f;
    use_global_smoother_filter = true;
    fgs_lambda = 500.0f;
    fgs_sigma = 1.5f;
    slic_type = SLIC;
    costMap = Mat();
}

struct MinHeap
{
public:
    MinHeap(int size)
    {
        //memset(this, 0, sizeof(*this));
        Init(size);
    }

    int Init(int size)
    {
        m_index.resize(size);
        m_weight.resize(size);
        m_size = size;
        m_validSize = 0;

        return 0;
    }
    int Push(float index, float weight)
    {
        if (m_validSize >= m_size)
        {
            CV_Error(CV_StsOutOfRange, " m_validSize >= m_size this problem can be resolved my decreasig k parameter");
        }
        m_index[m_validSize] = index;
        m_weight[m_validSize] = weight;
        m_validSize++;

        int i = m_validSize - 1;
        while (prior(m_weight[i], m_weight[(i - 1) / 2])) {
            swap(m_weight[i], m_weight[(i - 1) / 2]);
            swap(m_index[i], m_index[(i - 1) / 2]);
            i = (i - 1) / 2; // jump up to the parent
        }

        return i;
    }

    float Pop(float* weight = NULL)
    {
        if (m_validSize == 0) {
            return -1;
        }

        if (weight) {
            *weight = m_weight[0];
        }
        float outIdx = m_index[0];

        // use the last item to overwrite the first
        m_index[0] = m_index[m_validSize - 1];
        m_weight[0] = m_weight[m_validSize - 1];
        m_validSize--;

        // adjust the heap from top to bottom
        float rawIdx = m_index[0];
        float rawWt = m_weight[0];
        int candiPos = 0; // the root
        int i = 1; // left child of the root
        while (i < m_validSize) {
            // test right child
            if (i + 1 < m_validSize && prior(m_weight[i + 1], m_weight[i])) {
                i++;
            }
            if (prior(rawWt, m_weight[i])) {
                break;
            }
            m_index[candiPos] = m_index[i];
            m_weight[candiPos] = m_weight[i];
            candiPos = i;

            i = (i + 1) * 2 - 1; // left child
        }
        m_index[candiPos] = rawIdx;
        m_weight[candiPos] = rawWt;

        return outIdx;
    }
    void Clear()
    {
        m_validSize = 0;
    }
    int Size()
    {
        return m_validSize;
    }

private:
    inline bool prior(float w1, float w2)
    {
            return w1 < w2;
    }

    vector<float> m_index;
    vector<float> m_weight;
    int m_size;
    int m_validSize;
};

void RICInterpolatorImpl::interpolate(InputArray from_image, InputArray from_points, InputArray to_image, InputArray to_points, OutputArray dense_flow)
{
    CV_Assert(!from_image.empty());
    CV_Assert(from_image.depth() == CV_8U);
    CV_Assert((from_image.channels() == 3 || from_image.channels() == 1));
    CV_Assert(use_variational_refinement == false || !to_image.empty());
    CV_Assert(use_variational_refinement == false || to_image.depth() == CV_8U);
    CV_Assert(use_variational_refinement == false || to_image.channels() == 3 || to_image.channels() == 1);
    CV_Assert(!from_points.empty());
    CV_Assert(!to_points.empty());
    CV_Assert((from_points.isVector() || from_points.isMat()) && from_points.depth() == CV_32F);
    CV_Assert((to_points.isVector() || to_points.isMat()) && to_points.depth() == CV_32F);
    CV_Assert(from_points.sameSize(to_points));

    Mat from_mat = from_points.getMat();
    Mat to_mat = to_points.getMat();
    int npoints = from_mat.checkVector(2, CV_32F, false);

    if (from_mat.channels() != 2)
        from_mat = from_mat.reshape(2, npoints);

    if (to_mat.channels() != 2 ){
        to_mat = to_mat.reshape(2, npoints);
        npoints = from_mat.checkVector(2, CV_32F, false);
    }

    vector<SparseMatch> matches_vector(npoints);
    for(unsigned int i=0;i<matches_vector.size();i++)
        matches_vector[i] = SparseMatch(from_mat.at<Point2f>(i),to_mat.at<Point2f>(i));

    match_num = static_cast<int>(matches_vector.size());

    Mat src = from_image.getMat();
    Size src_size = src.size();

    labels = Mat(src_size, CV_32SC1);
    labels.setTo(-1);
    NNlabels = Mat(match_num, max_neighbors, CV_32S);
    NNlabels = Scalar(-1);
    NNdistances = Mat(match_num, max_neighbors, CV_32F);
    NNdistances = Scalar(0.0f);

    Mat matDistanceMap(src_size, CV_32FC1);
    matDistanceMap.setTo(1e10);

    if (costMap.empty())
    {
        costMap.create(src_size, CV_32FC1);
        computeGradientMagnitude(src, costMap);
    }
    else
        CV_Assert(costMap.rows == src.rows && costMap.cols == src.cols );

    costMap = (1000.0f - lambda) + lambda * costMap;

    for (unsigned int i = 0; i < matches_vector.size(); i++)
    {
        const SparseMatch & p = matches_vector[i];
        Point pos(static_cast<int>(p.reference_image_pos.x), static_cast<int>(p.reference_image_pos.y));
        labels.at<int>(pos) = i;
        matDistanceMap.at<float>(pos) = costMap.at<float>(pos);
    }

    geodesicDistanceTransform(matDistanceMap, costMap);

    g.resize(match_num);
    buildGraph(matDistanceMap, costMap);
    parallel_for_(Range(0, getNumThreads()), [&](const Range & range)
    {
        int stripe_sz = (int)ceil(match_num / (double)getNumThreads());
        int start = std::min(range.start * stripe_sz, match_num);
        int end = std::min(range.end   * stripe_sz, match_num);
        nodeHeap q((int)match_num);
        int num_expanded_vertices;
        vector<unsigned int> expanded_flag(match_num);
        node* neighbors;
        for (int i = start; i < end; i++)
        {
            if (g[i].empty())
                continue;
            num_expanded_vertices = 0;
            fill(expanded_flag.begin(), expanded_flag.end(), 0);
            q.clear();
            q.add(node((int)i, 0.0f));
            int* NNlabels_row = NNlabels.ptr<int>(i);
            float* NNdistances_row = NNdistances.ptr<float>(i);
            while (num_expanded_vertices < max_neighbors && !q.empty())
            {
                node vert_for_expansion = q.getMin();
                expanded_flag[vert_for_expansion.label] = 1;

                //write the expanded vertex to the dst:
                NNlabels_row[num_expanded_vertices] = vert_for_expansion.label;
                NNdistances_row[num_expanded_vertices] = vert_for_expansion.dist;
                num_expanded_vertices++;

                //update the heap:
                neighbors = &g[vert_for_expansion.label].front();
                for (int j = 0; j < (int)g[vert_for_expansion.label].size(); j++)
                {
                    if (!expanded_flag[neighbors[j].label])
                        q.updateNode(node(neighbors[j].label, vert_for_expansion.dist + neighbors[j].dist));
                }
            }
        }
    });

    Mat spLabels;
    Mat spNN;
    Mat spPos;
    Mat spItems;

    int spCnt = overSegmentaion(src, spLabels, sp_size);
    superpixelNeighborConstruction(spLabels, spCnt, spNN);
    superpixelLayoutAnalysis(spLabels, spCnt, spPos, spItems);

    vector<int> srcMatchIds(spCnt);
    for (int i = 0; i < spCnt; i++)
    {
        Point pos = static_cast<Point>(spPos.at<Point2f>(i) + Point2f(0.5f,0.5f));
        pos.x = MIN(labels.cols - 1, pos.x);
        pos.y = MIN(labels.rows - 1, pos.y);
        srcMatchIds[i] = labels.at<int>(pos);
    }

    int supportCnt = sp_nncnt;
    vector<int> supportMatchIds(spCnt*supportCnt);  // support matches for each superpixel
    vector<float> supportMatchDis(spCnt*supportCnt);

    findSupportMatches(srcMatchIds, spCnt, supportCnt, NNlabels, NNdistances, supportMatchIds, supportMatchDis);

    Mat transModels(spCnt,6, CV_32FC1);
    Mat fitModels(spCnt, 6, CV_32FC1);
    Mat rawModel(1, 6, CV_32FC1);
    rawModel.setTo(0);
    rawModel.at<float>(0) = 1;
    rawModel.at<float>(4) = 1;

    for (int i = 0; i < spCnt; i++) {
        int matId = supportMatchIds[i*supportCnt + 0];

        const SparseMatch & p = matches_vector[matId];
        float u = p.target_image_pos.x - p.reference_image_pos.x;
        float v = p.target_image_pos.y - p.reference_image_pos.y;

        rawModel.copyTo(transModels.row(i));
        transModels.at<float>(i,2) = u;
        transModels.at<float>(i,5) = v;
        transModels.row(i).copyTo(fitModels.row(i));
    }

    PropagateModels(spCnt, spNN, supportMatchIds, supportMatchDis, supportCnt, matches_vector, fitModels);

    dense_flow.create(from_image.size(), CV_32FC2);

    Mat U = Mat(src.rows, src.cols, CV_32FC1);
    Mat V = Mat(src.rows, src.cols, CV_32FC1);
    for (int i = 0; i < spCnt; i++) {
        for (int k = 0; k < spItems.cols; k++) {
            int x = spItems.at<Point>(i,k).x;
            int y = spItems.at<Point>(i,k).y;
            if (x < 0 || y < 0) {
                break;
            }
            float fx = fitModels.at<float>(i, 0) * x + fitModels.at<float>(i, 1) * y + fitModels.at<float>(i, 2);
            float fy = fitModels.at<float>(i, 3) * x + fitModels.at<float>(i, 4) * y + fitModels.at<float>(i, 5);
            //cout << fitModels << endl;
            U.at<float>(y, x) = fx - x;
            V.at<float>(y, x) = fy - y;
            if (abs(fx - x) > max_flow || abs(fy - y) > max_flow)
            {
                // use the translational model directly
                fx = transModels.at<float>(i, 0) * x + transModels.at<float>(i, 1) * y + transModels.at<float>(i, 2);
                fy = transModels.at<float>(i, 3) * x + transModels.at<float>(i, 4) * y + transModels.at<float>(i, 5);
                U.at<float>(y, x) = fx - x;
                V.at<float>(y, x) = fy - y;
            }
        }
    }

    Mat dst;
    Mat prevGrey, currGrey;

    if (use_variational_refinement)
    {
        Mat src2 = to_image.getMat();
        cv::medianBlur(U, U, 3);
        cv::medianBlur(V, V, 3);
        Ptr<VariationalRefinement> variationalrefine = VariationalRefinement::create();
        cvtColor(src, prevGrey, COLOR_BGR2GRAY);
        cvtColor(src2, currGrey, COLOR_BGR2GRAY);
        variationalrefine->setOmega(1.9f);
        variationalrefine->calcUV(prevGrey, currGrey, U,V);
    }
    Mat UV[2] = { U,V };
    merge(UV, 2, dst);

    if (use_global_smoother_filter)
    {
        if (prevGrey.empty())
            cvtColor(src, prevGrey, COLOR_BGR2GRAY);
        fastGlobalSmootherFilter(prevGrey, dst, dst, fgs_lambda, fgs_sigma);
    }
    dst.copyTo(dense_flow.getMat());
    costMap.release();
}

void RICInterpolatorImpl::geodesicDistanceTransform(Mat& distances, Mat& cost_map)
{
    float c1 = 1.0f / 2.0f;
    float c2 = sqrt(2.0f) / 2.0f;
    float d = 0.0f;
    int i, j;
    float *dist_row, *cost_row;
    float *dist_row_prev, *cost_row_prev;
    int *label_row;
    int *label_row_prev;

#define CHK_GD(cur_dist,cur_label,cur_cost,prev_dist,prev_label,prev_cost,coef)\
{\
    d = prev_dist + coef*(cur_cost+prev_cost);\
    if(cur_dist>d){\
        cur_dist=d;\
        cur_label = prev_label;}\
}

    for (int it = 0; it < distance_transform_num_iter; it++)
    {
        //first pass (left-to-right, top-to-bottom):
        dist_row = distances.ptr<float>(0);
        label_row = labels.ptr<int>(0);
        cost_row = cost_map.ptr<float>(0);
        for (j = 1; j < distances.cols; j++)
            CHK_GD(dist_row[j], label_row[j], cost_row[j], dist_row[j - 1], label_row[j - 1], cost_row[j - 1], c1);

        for (i = 1; i < distances.rows; i++)
        {
            dist_row = distances.ptr<float>(i);
            dist_row_prev = distances.ptr<float>(i - 1);

            label_row = labels.ptr<int>(i);
            label_row_prev = labels.ptr<int>(i - 1);

            cost_row = cost_map.ptr<float>(i);
            cost_row_prev = cost_map.ptr<float>(i - 1);

            j = 0;
            CHK_GD(dist_row[j], label_row[j], cost_row[j], dist_row_prev[j], label_row_prev[j], cost_row_prev[j], c1);
            CHK_GD(dist_row[j], label_row[j], cost_row[j], dist_row_prev[j + 1], label_row_prev[j + 1], cost_row_prev[j + 1], c2);
            j++;
            for (; j < distances.cols - 1; j++)
            {
                CHK_GD(dist_row[j], label_row[j], cost_row[j], dist_row[j - 1], label_row[j - 1], cost_row[j - 1], c1);
                CHK_GD(dist_row[j], label_row[j], cost_row[j], dist_row_prev[j - 1], label_row_prev[j - 1], cost_row_prev[j - 1], c2);
                CHK_GD(dist_row[j], label_row[j], cost_row[j], dist_row_prev[j], label_row_prev[j], cost_row_prev[j], c1);
                CHK_GD(dist_row[j], label_row[j], cost_row[j], dist_row_prev[j + 1], label_row_prev[j + 1], cost_row_prev[j + 1], c2);
            }
            CHK_GD(dist_row[j], label_row[j], cost_row[j], dist_row[j - 1], label_row[j - 1], cost_row[j - 1], c1);
            CHK_GD(dist_row[j], label_row[j], cost_row[j], dist_row_prev[j - 1], label_row_prev[j - 1], cost_row_prev[j - 1], c2);
            CHK_GD(dist_row[j], label_row[j], cost_row[j], dist_row_prev[j], label_row_prev[j], cost_row_prev[j], c1);
        }

        //second pass (right-to-left, bottom-to-top):
        dist_row = distances.ptr<float>(distances.rows - 1);
        label_row = labels.ptr<int>(distances.rows - 1);
        cost_row = cost_map.ptr<float>(distances.rows - 1);
        for (j = distances.cols - 2; j >= 0; j--)
            CHK_GD(dist_row[j], label_row[j], cost_row[j], dist_row[j + 1], label_row[j + 1], cost_row[j + 1], c1);

        for (i = distances.rows - 2; i >= 0; i--)
        {
            dist_row = distances.ptr<float>(i);
            dist_row_prev = distances.ptr<float>(i + 1);

            label_row = labels.ptr<int>(i);
            label_row_prev = labels.ptr<int>(i + 1);

            cost_row = cost_map.ptr<float>(i);
            cost_row_prev = cost_map.ptr<float>(i + 1);

            j = distances.cols - 1;
            CHK_GD(dist_row[j], label_row[j], cost_row[j], dist_row_prev[j], label_row_prev[j], cost_row_prev[j], c1);
            CHK_GD(dist_row[j], label_row[j], cost_row[j], dist_row_prev[j - 1], label_row_prev[j - 1], cost_row_prev[j - 1], c2);
            j--;
            for (; j > 0; j--)
            {
                CHK_GD(dist_row[j], label_row[j], cost_row[j], dist_row[j + 1], label_row[j + 1], cost_row[j + 1], c1);
                CHK_GD(dist_row[j], label_row[j], cost_row[j], dist_row_prev[j + 1], label_row_prev[j + 1], cost_row_prev[j + 1], c2);
                CHK_GD(dist_row[j], label_row[j], cost_row[j], dist_row_prev[j], label_row_prev[j], cost_row_prev[j], c1);
                CHK_GD(dist_row[j], label_row[j], cost_row[j], dist_row_prev[j - 1], label_row_prev[j - 1], cost_row_prev[j - 1], c2);
            }
            CHK_GD(dist_row[j], label_row[j], cost_row[j], dist_row[j + 1], label_row[j + 1], cost_row[j + 1], c1);
            CHK_GD(dist_row[j], label_row[j], cost_row[j], dist_row_prev[j + 1], label_row_prev[j + 1], cost_row_prev[j + 1], c2);
            CHK_GD(dist_row[j], label_row[j], cost_row[j], dist_row_prev[j], label_row_prev[j], cost_row_prev[j], c1);
        }
    }
#undef CHK_GD
}

void RICInterpolatorImpl::buildGraph(Mat& distances, Mat& cost_map)
{
    float *dist_row, *cost_row;
    float *dist_row_prev, *cost_row_prev;
    const int *label_row;
    const int *label_row_prev;
    int i, j;
    const float c1 = 1.0f / 2.0f;
    const float c2 = sqrt(2.0f) / 2.0f;
    float d;
    bool found;
    int h = labels.rows;
    int w = labels.cols;
#define CHK_GD(cur_dist,cur_label,cur_cost,prev_dist,prev_label,prev_cost,coef)\
    if(cur_label!=prev_label)\
    {\
        d = prev_dist + cur_dist + coef*(cur_cost+prev_cost);\
        found = false;\
        for(unsigned int n=0;n<g[prev_label].size();n++)\
        {\
            if(g[prev_label][n].label==cur_label)\
            {\
                g[prev_label][n].dist = min(g[prev_label][n].dist,d);\
                found=true;\
                break;\
            }\
        }\
        if(!found)\
            g[prev_label].push_back(node(cur_label ,d));\
    }

    dist_row = distances.ptr<float>(0);
    label_row = labels.ptr<int>(0);
    cost_row = cost_map.ptr<float>(0);
    for (j = 1; j < w; j++)
        CHK_GD(dist_row[j], label_row[j], cost_row[j], dist_row[j - 1], label_row[j - 1], cost_row[j - 1], c1);

    for (i = 1; i < h; i++)
    {
        dist_row = distances.ptr<float>(i);
        dist_row_prev = distances.ptr<float>(i - 1);

        label_row = labels.ptr<int>(i);
        label_row_prev = labels.ptr<int>(i - 1);

        cost_row = cost_map.ptr<float>(i);
        cost_row_prev = cost_map.ptr<float>(i - 1);

        j = 0;
        CHK_GD(dist_row[j], label_row[j], cost_row[j], dist_row_prev[j], label_row_prev[j], cost_row_prev[j], c1);
        CHK_GD(dist_row[j], label_row[j], cost_row[j], dist_row_prev[j + 1], label_row_prev[j + 1], cost_row_prev[j + 1], c2);
        j++;
        for (; j < w - 1; j++)
        {
            CHK_GD(dist_row[j], label_row[j], cost_row[j], dist_row[j - 1], label_row[j - 1], cost_row[j - 1], c1);
            CHK_GD(dist_row[j], label_row[j], cost_row[j], dist_row_prev[j - 1], label_row_prev[j - 1], cost_row_prev[j - 1], c2);
            CHK_GD(dist_row[j], label_row[j], cost_row[j], dist_row_prev[j], label_row_prev[j], cost_row_prev[j], c1);
            CHK_GD(dist_row[j], label_row[j], cost_row[j], dist_row_prev[j + 1], label_row_prev[j + 1], cost_row_prev[j + 1], c2);
        }
        CHK_GD(dist_row[j], label_row[j], cost_row[j], dist_row[j - 1], label_row[j - 1], cost_row[j - 1], c1);
        CHK_GD(dist_row[j], label_row[j], cost_row[j], dist_row_prev[j - 1], label_row_prev[j - 1], cost_row_prev[j - 1], c2);
        CHK_GD(dist_row[j], label_row[j], cost_row[j], dist_row_prev[j], label_row_prev[j], cost_row_prev[j], c1);
    }
#undef CHK_GD

    // force equal distances in both directions:
    node* neighbors;
    for (i = 0; i < match_num; i++)
    {
        if (g[i].empty())
            continue;
        neighbors = &g[i].front();
        for (j = 0; j < (int)g[i].size(); j++)
        {
            found = false;

            for (unsigned int n = 0; n < g[neighbors[j].label].size(); n++)
            {
                if (g[neighbors[j].label][n].label == i)
                {
                    neighbors[j].dist = g[neighbors[j].label][n].dist = min(neighbors[j].dist, g[neighbors[j].label][n].dist);
                    found = true;
                    break;
                }
            }

            if (!found)
                g[neighbors[j].label].push_back(node((int)i, neighbors[j].dist));
        }
    }
}

int RICInterpolatorImpl::overSegmentaion(const Mat & img, Mat & outLabels, const int spSize)
{
    Mat labImg;
    cvtColor(img, labImg, COLOR_BGR2Lab);
    Ptr< SuperpixelSLIC> slic = createSuperpixelSLIC(labImg, slic_type, spSize, sp_ruler);
    slic->iterate(5);
    slic->getLabels(outLabels);
    return slic->getNumberOfSuperpixels();
}

void RICInterpolatorImpl::superpixelNeighborConstruction(const Mat & _labels, int labelCnt, Mat& outNeighbor)
{
    // init
    outNeighbor = Mat(labelCnt, max_neighbors, CV_32SC1); // only support 32 neighbors
    outNeighbor.setTo(-1);

    vector<int> diffPairs(labels.cols*_labels.rows * 4, 0);
    int diffPairCnt = 0;
    for (int i = 1; i < _labels.rows; i++) {
        for (int j = 1; j < _labels.cols; j++) {

            int l0 = _labels.at<int>(i,j);
            int l1 = _labels.at<int>(i,j- 1);
            int l2 = _labels.at<int>(i-1,j);

            if (l0 != l1) {
                diffPairs[2 * diffPairCnt] = l0;
                diffPairs[2 * diffPairCnt + 1] = l1;
                diffPairCnt++;
            }

            if (l0 != l2) {
                diffPairs[2 * diffPairCnt] = l0;
                diffPairs[2 * diffPairCnt + 1] = l2;
                diffPairCnt++;
            }
        }
    }

    for (int i = 0; i < diffPairCnt; i++) {
        int a = diffPairs[2 * i];
        int b = diffPairs[2 * i + 1];
        int k = 0;

        // add to neighbor list of a
        for (k = 0; k < max_neighbors; k++) {
            if (outNeighbor.at<int>(a, k) < 0) {
                break;
            }
            if (outNeighbor.at<int>(a,k) == b) {
                k = -1;
                break;
            }
        }
        if (k >= 0 && k < max_neighbors) {
            outNeighbor.at<int>(a, k) = b;
        }

        // add to neighbor list of b
        for (k = 0; k < max_neighbors; k++) {
            if (outNeighbor.at<int>(b, k) < 0) {
                break;
            }
            if (outNeighbor.at<int>(b, k) == a) {
                k = -1;
                break;
            }
        }
        if (k >= 0 && k < max_neighbors) {
            outNeighbor.at<int>(b, k) = a;
        }
    }

}

void RICInterpolatorImpl::superpixelLayoutAnalysis(const Mat & _labels, int labelCnt, Mat & outCenterPositions, Mat & outNodeItemLists)
{
    outCenterPositions = Mat(labelCnt,1,CV_32FC2); // x and y
    outCenterPositions.setTo(0);

    vector<int> itemCnt(labelCnt, 0);

    for (int i = 0; i < _labels.rows; i++) {
        for (int j = 0; j < _labels.cols; j++) {
            int id = _labels.at<int>(i,j);
            outCenterPositions.at<Point2f>(id) += Point2f(static_cast<float>(j),static_cast<float>(i));
            itemCnt[id]++;
        }
    }
    int maxItemCnt = 0;
    for (int i = 0; i < labelCnt; i++) {
        if (itemCnt[i] > maxItemCnt) {
            maxItemCnt = itemCnt[i];
        }
        if (itemCnt[i] > 0) {
            outCenterPositions.at<Point2f>(i).x /= itemCnt[i];
            outCenterPositions.at<Point2f>(i).y /= itemCnt[i];
        }
        else {
            outCenterPositions.at<Point2f>(i) = Point2f(-1,-1);
        }
    }

    // get node item lists
    outNodeItemLists = Mat(labelCnt, maxItemCnt, CV_32SC2);
    outNodeItemLists.setTo(-1);
    fill(itemCnt.begin(), itemCnt.end(), 0);
    for (int i = 0; i < _labels.rows; i++) {
        for (int j = 0; j < _labels.cols; j++) {
            int id = _labels.at<int>(i,j);
            int cnt = itemCnt[id];
            outNodeItemLists.at<Point>(id, cnt) = Point(j,i);
            itemCnt[id]++;
        }
    }

}

void RICInterpolatorImpl::findSupportMatches(vector<int> & srcIds, int srcCnt, int supportCnt, Mat & matNN,
                                            Mat & matNNDis, vector<int> & outSupportIds, vector<float> &  outSupportDis)
{
    fill(outSupportIds.begin(), outSupportIds.end(), -1); // -1
    fill(outSupportDis.begin(), outSupportDis.end(), -1.f); // -1

    int allNodeCnt = matNN.rows;
    MinHeap H(allNodeCnt); // min-heap
    vector<float> currDis(allNodeCnt);

    for (int i = 0; i < srcCnt; i++)
    {
        int id = srcIds[i];
        int* pSupportIds   = &outSupportIds[i * supportCnt];
        float* pSupportDis = &outSupportDis[i * supportCnt];

        H.Clear();
        fill(currDis.begin(), currDis.end(), numeric_limits<float>::max());

        int validSupportCnt = 0;

        H.Push(static_cast<float>(id), 0); // min distance
        currDis[id] = 0;

        while (H.Size()) {
            float dis;
            int idx = static_cast<int>(H.Pop(&dis));

            if (dis > currDis[idx]) {
                continue;
            }

            pSupportIds[validSupportCnt] = idx;
            pSupportDis[validSupportCnt] = dis;
            validSupportCnt++;
            if (validSupportCnt >= supportCnt) {
                break;
            }

            for (int k = 0; k < matNN.cols; k++) {
                int nb = matNN.at<int>(idx, k);
                if (nb < 0) {
                    break;
                }
                float newDis = dis + matNNDis.at<float>(idx,k);
                if (newDis < currDis[nb]) {
                    H.Push(static_cast<float>(nb), newDis);
                    currDis[nb] = newDis;
                }
            }
        }
    }
}

int RICInterpolatorImpl::PropagateModels(int spCnt, Mat & spNN, vector<int> & supportMatchIds, vector<float> & supportMatchDis, int supportCnt,
    const vector<SparseMatch> &inputMatches, Mat & outModels)
{
    int iterCnt = model_iter;

    srand(0);

    Mat inLierFlag(spCnt, supportCnt, CV_32SC1);
    Mat tmpInlierFlag(1, supportCnt, CV_32SC1);
    Mat tmpModel(1, 6, CV_32FC1);

    // prepare data
    vector<float> bestCost(spCnt);
    parallel_for_(Range(0, spCnt), [&](const Range& range)
    {
        for (int i = range.start; i < range.end; i++)
        {
            Mat outModelRow = outModels.row(i);
            Mat inlierFlagRow = inLierFlag.row(i);
            bestCost[i] =
                HypothesisEvaluation(
                outModelRow,
                &supportMatchIds[i * supportCnt],
                &supportMatchDis[i * supportCnt],
                supportCnt,
                inputMatches,
                inlierFlagRow);
        }
    }
    );
    parallel_for_(Range(0, iterCnt), [&](const Range& range)
    {
        vector<int> vFlags(spCnt);
        for (int iter = range.start; iter < range.end; iter++)
        {
            fill(vFlags.begin(), vFlags.end(), 0);

            int startPos = 0, endPos = spCnt, step = 1;
            if (iter % 2 == 1) {
                startPos = spCnt - 1; endPos = -1; step = -1;
            }

            for (int idx = startPos; idx != endPos; idx += step)
            {
                for (int i = 0; i < spNN.cols; i++) {
                    int nb = spNN.at<int>(idx, i);
                    if (nb < 0) break;
                    if (!vFlags[nb]) continue;
                    Mat NNModel = outModels.row(nb);
                    float cost = HypothesisEvaluation(
                        outModels.row(nb),
                        &supportMatchIds[idx * supportCnt],
                        &supportMatchDis[idx * supportCnt],
                        supportCnt,
                        inputMatches,
                        tmpInlierFlag);
                    if (cost < bestCost[idx])
                    {
                        outModels.row(nb).copyTo(outModels.row(idx));
                        tmpInlierFlag.copyTo(inLierFlag.row(idx));
                        bestCost[idx] = cost;
                    }
                }

                // random test
                int testCnt = 1;
                for (int i = 0; i < testCnt; i++) {
                    if (HypothesisGeneration(&supportMatchIds[idx * supportCnt], supportCnt, inputMatches, tmpModel) == 0)
                    {
                        float cost = HypothesisEvaluation(
                            tmpModel,
                            &supportMatchIds[idx * supportCnt],
                            &supportMatchDis[idx * supportCnt],
                            supportCnt,
                            inputMatches,
                            tmpInlierFlag);
                        if (cost < bestCost[idx])
                        {
                            tmpModel.copyTo(outModels.row(idx));
                            tmpInlierFlag.copyTo(inLierFlag.row(idx));
                            bestCost[idx] = cost;
                        }
                    }
                }
                vFlags[idx] = 1;
            }
        }
    }
    );
    // refinement
    if (refine_models)
    {
        parallel_for_(Range(0, spCnt), [&](const Range& range)
        {
            int averInlier = 0;
            int minPtCnt = 30;
            for (int i = range.start; i < range.end; i++)
            {

                Mat pt1(supportCnt, 1, CV_32FC2);
                Mat pt2(supportCnt, 1, CV_32FC2);
                vector<int>   inlier_labels(supportCnt);
                vector<float> inlier_distances(supportCnt);
                Mat fitModel;

                int* matNodes = &supportMatchIds[i * supportCnt];
                float* matDis = &supportMatchDis[i * supportCnt];

                int inlierCnt = 0;
                for (int k = 0; k < supportCnt; k++) {
                    if (inLierFlag.at<int>(i, k))
                    {
                        int matId = matNodes[k];
                        inlier_labels[inlierCnt] = matId;
                        inlier_distances[inlierCnt] = GetWeightFromDistance(matDis[k]);
                        inlierCnt++;
                    }
                }
                if (inlierCnt >= minPtCnt)
                {
                    weightedLeastSquaresAffineFit(&inlier_labels[0], &inlier_distances[0], inlierCnt, 0.01f, &inputMatches[0], fitModel);
                    fitModel.reshape(1, 1).copyTo(outModels.row(i));

                }
                averInlier += inlierCnt;
            }
        }
        );
    }

    return 0;
}

float RICInterpolatorImpl::GetWeightFromDistance(float dis)
{
    return exp(-dis / (alpha * 1000));
}

float RICInterpolatorImpl::HypothesisEvaluation(const Mat & inModel, const int * matNodes, const float * matDis, int matCnt, const vector<SparseMatch> & inputMatches, Mat & outInLier)
{
    float errTh = 5.;

    // find inliers
    int inLierCnt = 0;
    float cost = 0;
    for (int k = 0; k < matCnt; k++) {
        int matId = matNodes[k];
        const SparseMatch & p = inputMatches[matId];
        float x1 = p.reference_image_pos.x;
        float y1 = p.reference_image_pos.y;
        float xp = inModel.at<float>(0) * x1 + inModel.at<float>(1) * y1 + inModel.at<float>(2);
        float yp = inModel.at<float>(3) * x1 + inModel.at<float>(4) * y1 + inModel.at<float>(5);
        float pu = xp - x1, pv = yp - y1;

        float tu = p.target_image_pos.x - p.reference_image_pos.x;
        float tv = p.target_image_pos.y - p.reference_image_pos.y;
        float wt = GetWeightFromDistance(matDis[k]);

        if (std::isnan(pu) || std::isnan(pv)) {
            outInLier.at<int>(k) = 0;
            cost += wt * errTh;
            continue;
        }

        float dis = sqrt((tu - pu)*(tu - pu) + (tv - pv)*(tv - pv));
        if (dis < errTh) {
            outInLier.at<int>(k) = 1;
            inLierCnt++;
            cost += wt * dis;
        }
        else {
            outInLier.at<int>(k) = 0;
            cost += wt * errTh;
        }
    }
    return cost;
}

int RICInterpolatorImpl::HypothesisGeneration(int* matNodes, int matCnt, const vector<SparseMatch> & inputMatches, Mat & outModel)
{
    if (matCnt < 3)
    {
        return -1;
    }

    int pickTimes = 0;
    int maxPickTimes = 10;
    float p1[6], p2[6]; // 3 pairs
    bool pick_data = true;
    float deter = 0;
    while (pick_data)
    {
        pick_data = false;
        // pick 3 group of points randomly
        for (int k = 0; k < 3; k++) {
            int matId = matNodes[rand() % matCnt];
            const SparseMatch & p = inputMatches[matId];
            p1[2 * k] = p.reference_image_pos.x;
            p1[2 * k + 1] = p.reference_image_pos.y;
            p2[2 * k] = p.target_image_pos.x;
            p2[2 * k + 1] = p.target_image_pos.y;
        }
        // are the 3 points on the same line ?
        deter = 0; // determinant
        deter = p1[0] * p1[3] + p1[2] * p1[5] + p1[4] * p1[1]
              - p1[4] * p1[3] - p1[0] * p1[5] - p1[2] * p1[1];
        if (abs(deter) <= FLT_EPSILON)
        {
            pickTimes++;
            if (pickTimes > maxPickTimes) {
                return -1;
            }
            pick_data = true;
        }
    }
    // estimate the model
    float inv[9];
    inv[0] = (p1[3] - p1[5]) / deter;
    inv[1] = (p1[5] - p1[1]) / deter;
    inv[2] = (p1[1] - p1[3]) / deter;
    inv[3] = (p1[4] - p1[2]) / deter;
    inv[4] = (p1[0] - p1[4]) / deter;
    inv[5] = (p1[2] - p1[0]) / deter;
    inv[6] = (p1[2] * p1[5] - p1[3] * p1[4]) / deter;
    inv[7] = (p1[1] * p1[4] - p1[0] * p1[5]) / deter;
    inv[8] = (p1[0] * p1[3] - p1[1] * p1[2]) / deter;

    outModel.at<float>(0) = inv[0] * p2[0] + inv[1] * p2[2] + inv[2] * p2[4];
    outModel.at<float>(1) = inv[3] * p2[0] + inv[4] * p2[2] + inv[5] * p2[4];
    outModel.at<float>(2) = inv[6] * p2[0] + inv[7] * p2[2] + inv[8] * p2[4];
    outModel.at<float>(3) = inv[0] * p2[1] + inv[1] * p2[3] + inv[2] * p2[5];
    outModel.at<float>(4) = inv[3] * p2[1] + inv[4] * p2[3] + inv[5] * p2[5];
    outModel.at<float>(5) = inv[6] * p2[1] + inv[7] * p2[3] + inv[8] * p2[5];

    return 0;
}
CV_EXPORTS_W

Ptr<RICInterpolator> createRICInterpolator()
{
    return Ptr<RICInterpolator>(RICInterpolatorImpl::create());
}

}
}
