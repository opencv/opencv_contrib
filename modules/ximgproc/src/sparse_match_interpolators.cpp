#include "opencv2/ximgproc/sparse_match_interpolator.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include <unordered_set>
#include <unordered_map>
#include <algorithm>
#include <queue>
#include <bitset>

#define INF 1E+20F

namespace cv {
namespace ximgproc {

SparseMatch::SparseMatch(Point2f ref_point, Point2f target_point)
{
    reference_image_pos = ref_point;
    target_image_pos = target_point;
}

bool operator<(const SparseMatch& lhs,const SparseMatch& rhs)
{
    if((int)(lhs.reference_image_pos.y+0.5f)!=(int)(rhs.reference_image_pos.y+0.5f))
        return (lhs.reference_image_pos.y<rhs.reference_image_pos.y);
    else
        return (lhs.reference_image_pos.x<rhs.reference_image_pos.x);
}

struct node
{
    float dist;
    short label;
    node() {}
    node(short l,float d): label(l), dist(d){}
};

class EdgeAwareInterpolatorImpl : public EdgeAwareInterpolator
{
public:
    static Ptr<EdgeAwareInterpolatorImpl> create(bool useSED);
    void interpolate(InputArray reference_image, InputArray target_image, InputArray matches, OutputArray dense_flow);

protected:
    int w,h;
    bool useSED;
    vector<node>* g;

    Ptr<StructuredEdgeDetection> sed;
    void init();
    void geodesicDistanceTransform(Mat& distances, Mat& labels, Mat& cost_map, int num_iter=1);
    void getKNNMatches(Mat& distances, Mat& labels, Mat& cost_map, int k, int match_num, Mat& dstNNlabels, Mat& dstNNdistances);
    void leastSquaresInterpolation(Mat& labels, Mat& NNlabels, Mat& NNdistances, vector<SparseMatch>& matches, Mat& dst_dense_flow);
    void ransacInterpolation(Mat& labels, Mat& NNlabels, Mat& NNdistances, vector<SparseMatch>& matches, Mat& dst_dense_flow, int num_iter);
    void computeSED(Mat src, Mat& dst);
    void computeGradientMagnitude(Mat src, Mat& dst, bool useL2=false);
};

void EdgeAwareInterpolatorImpl::init() {}

Ptr<EdgeAwareInterpolatorImpl> EdgeAwareInterpolatorImpl::create(bool useSED)
{
    EdgeAwareInterpolatorImpl *gd = new EdgeAwareInterpolatorImpl();
    gd->init();
    gd->useSED = useSED;
    if(useSED)
        gd->sed = createStructuredEdgeDetection("model.yml");
    return Ptr<EdgeAwareInterpolatorImpl>(gd);
}

void EdgeAwareInterpolatorImpl::interpolate(InputArray reference_image, InputArray, InputArray matches, OutputArray dense_flow)
{
    float lambda = 999.0f; //controls edge sensitivity
    int k=128; //num of nearest-neighbor matches
    w = reference_image.cols();
    h = reference_image.rows();

    Mat distances(h,w,CV_32F);
    Mat labels   (h,w,CV_16S);
    Mat cost_map (h,w,CV_32F);
    distances = Scalar(INF);
    labels    = Scalar(-1);

    vector<SparseMatch> matches_vector = *(const std::vector<SparseMatch>*)matches.getObj();
    std::sort(matches_vector.begin(),matches_vector.end());
    int x,y;
    int match_num = matches_vector.size();
    CV_Assert(match_num<SHRT_MAX);
    int x_prev=-1;
    for(unsigned int i=0;i<matches_vector.size();i++)
    {
        x = min((int)(matches_vector[i].reference_image_pos.x+0.5f),w-1);
        y = min((int)(matches_vector[i].reference_image_pos.y+0.5f),h-1);

        x_prev = x;
        distances.at<float>(y,x) = 0.0f;
        labels.at<short>(y,x) = (short)i;
    }

    if(useSED)
        computeSED(reference_image.getMat(),cost_map);
    else
        computeGradientMagnitude(reference_image.getMat(),cost_map);

    cost_map = (1000.0f-lambda) + lambda*cost_map;
    geodesicDistanceTransform(distances,labels,cost_map,1);

    g = new vector<node>[match_num];
    Mat NNlabels(match_num,k,CV_16S);
    Mat NNdistances(match_num,k,CV_32F);
    getKNNMatches(distances,labels,cost_map,k,match_num,NNlabels,NNdistances);

    dense_flow.create(reference_image.size(),CV_32FC2);
    Mat dst = dense_flow.getMat();

    ransacInterpolation(labels,NNlabels,NNdistances,matches_vector,dst,2);
    fastGlobalSmootherFilter(reference_image.getMat(),dst,dst,500.0,1.5);
    delete[] g;
}

// distances - CV_32F - zero in keypoints, INF in other points 
// labels - CV_16S - indexed keypoints, -1 in other points
// cost_map - CV_32F, gradient magnitude or more some other edge detector response
void EdgeAwareInterpolatorImpl::geodesicDistanceTransform(Mat& distances, Mat& labels, Mat& cost_map, int num_iter)
{
    float c1 = 1.0f/2.0f;
    float c2 = sqrt(2.0f)/2.0f;
    float d = 0.0f;
    int i,j;
    float *dist_row,      *cost_row;
    float *dist_row_prev, *cost_row_prev;
    short *label_row;
    short *label_row_prev;

#define CHECK(cur_dist,cur_label,cur_cost,prev_dist,prev_label,prev_cost,coef)\
    d = prev_dist + coef*(cur_cost+prev_cost);\
    if(cur_dist>d){\
        cur_dist=d;\
        cur_label = prev_label;}

    for(int it=0;it<num_iter;it++)
    {
        //first pass (left-to-right, top-to-bottom):
        dist_row  = distances.ptr<float>(0);
        label_row = labels.ptr<short>(0);
        cost_row  = cost_map.ptr<float>(0);
        for(j=1;j<w;j++)
            CHECK(dist_row[j],label_row[j],cost_row[j],dist_row[j-1],label_row[j-1],cost_row[j-1],c1);

        for(i=1;i<h;i++)
        {
            dist_row       = distances.ptr<float>(i);
            dist_row_prev  = distances.ptr<float>(i-1);

            label_row      = labels.ptr<short>(i);
            label_row_prev = labels.ptr<short>(i-1);

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

        //second pass (right-to-left, bottom-to-top):
        dist_row  = distances.ptr<float>(h-1);
        label_row = labels.ptr<short>(h-1);
        cost_row  = cost_map.ptr<float>(h-1);
        for(j=w-2;j>=0;j--)
            CHECK(dist_row[j],label_row[j],cost_row[j],dist_row[j+1],label_row[j+1],cost_row[j+1],c1);

        for(i=h-2;i>=0;i--)
        {
            dist_row       = distances.ptr<float>(i);
            dist_row_prev  = distances.ptr<float>(i+1);

            label_row      = labels.ptr<short>(i);
            label_row_prev = labels.ptr<short>(i+1);

            cost_row      = cost_map.ptr<float>(i);
            cost_row_prev = cost_map.ptr<float>(i+1);

            j=w-1;
            CHECK(dist_row[j],label_row[j],cost_row[j],dist_row_prev[j]  ,label_row_prev[j]  ,cost_row_prev[j]  ,c1);
            CHECK(dist_row[j],label_row[j],cost_row[j],dist_row_prev[j-1],label_row_prev[j-1],cost_row_prev[j-1],c2);            
            j--;
            for(;j>0;j--)
            {
                CHECK(dist_row[j],label_row[j],cost_row[j],dist_row[j+1]     ,label_row[j+1]     ,cost_row[j+1]     ,c1);
                CHECK(dist_row[j],label_row[j],cost_row[j],dist_row_prev[j+1],label_row_prev[j+1],cost_row_prev[j+1],c2);
                CHECK(dist_row[j],label_row[j],cost_row[j],dist_row_prev[j]  ,label_row_prev[j]  ,cost_row_prev[j]  ,c1);
                CHECK(dist_row[j],label_row[j],cost_row[j],dist_row_prev[j-1],label_row_prev[j-1],cost_row_prev[j-1],c2);            
            }
            CHECK(dist_row[j],label_row[j],cost_row[j],dist_row[j+1]     ,label_row[j+1]     ,cost_row[j+1]     ,c1);
            CHECK(dist_row[j],label_row[j],cost_row[j],dist_row_prev[j+1],label_row_prev[j+1],cost_row_prev[j+1],c2);
            CHECK(dist_row[j],label_row[j],cost_row[j],dist_row_prev[j]  ,label_row_prev[j]  ,cost_row_prev[j]  ,c1);
        }
    }
#undef CHECK
}

void EdgeAwareInterpolatorImpl::computeSED(Mat src, Mat& dst)
{
    Mat src_CV32F;
    src.convertTo(src_CV32F, CV_32F, 1/255.0);
    sed->detectEdges(src_CV32F, dst);
}

void EdgeAwareInterpolatorImpl::computeGradientMagnitude(Mat src, Mat& dst, bool useL2)
{
    Mat dx,dy,src_gray;
    cvtColor(src,src_gray,COLOR_BGR2GRAY);
    Sobel(src_gray, dx, CV_16SC1, 1, 0);
    Sobel(src_gray, dy, CV_16SC1, 0, 1);
    float norm_coef = 4.0f*255.0f;

    if(useL2)
    {
        norm_coef*=norm_coef;
        for(int i=0;i<h;i++)
        {
            short* dx_row  = dx.ptr<short>(i);
            short* dy_row  = dy.ptr<short>(i);
            float* dst_row = dst.ptr<float>(i);

            for(int j=0;j<w;j++)
                dst_row[j] = ((float)dx_row[j]*dx_row[j]+dy_row[j]*dy_row[j])/norm_coef;

            hal::sqrt(dst_row,dst_row,h);
        }
    }
    else
    {
        for(int i=0;i<h;i++)
        {
            short* dx_row  = dx.ptr<short>(i);
            short* dy_row  = dy.ptr<short>(i);
            float* dst_row = dst.ptr<float>(i);

            for(int j=0;j<w;j++)
                dst_row[j] = ((float)abs(dx_row[j])+abs(dy_row[j]))/norm_coef;
        }
    }
}

struct nodeHeap
{
    // start indexing from 1 (root)
    // children: 2*i, 2*i+1
    // parent: i>>1
    node* heap;
    short* heap_pos;
    node tmp_node;
    short size;
    short num_labels;

    nodeHeap(short _num_labels)
    {
        num_labels = _num_labels;
        heap = new node[num_labels+1];
        heap[0] = node(-1,-1.0f);
        heap_pos = new short[num_labels];
        memset(heap_pos,0,sizeof(short)*num_labels);
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
        memset(heap_pos,0,sizeof(short)*num_labels);
    }

    inline bool empty()
    {
        return (size==0);
    }

    inline void nodeSwap(short idx1, short idx2)
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
        short i = size;
        short parent_i = i>>1;
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

        short i=1;
        short left,right;
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

        short parent_i = i>>1;
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
            short i = heap_pos[n.label];
            heap[i].dist = min(heap[i].dist,n.dist);
            short parent_i = i>>1;
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

//dstNNlabels, dstNNdistances - each row contains all NN matches of a particular match
//k - number of NN matches
void EdgeAwareInterpolatorImpl::getKNNMatches(Mat& distances, Mat& labels, Mat& cost_map, int k, int match_num, Mat& dstNNlabels, Mat& dstNNdistances)
{
    //Step 1: build a graph out of the matches
    float *dist_row,      *cost_row;
    float *dist_row_prev, *cost_row_prev;
    short *label_row;
    short *label_row_prev;
    int i,j;
    float c1 = 1.0f/2.0f;
    float c2 = sqrt(2.0f)/2.0f;
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
    label_row = labels.ptr<short>(0);
    cost_row  = cost_map.ptr<float>(0);
    for(j=1;j<w;j++)
        CHECK(dist_row[j],label_row[j],cost_row[j],dist_row[j-1],label_row[j-1],cost_row[j-1],c1);

    for(i=1;i<h;i++)
    {
        dist_row       = distances.ptr<float>(i);
        dist_row_prev  = distances.ptr<float>(i-1);

        label_row      = labels.ptr<short>(i);
        label_row_prev = labels.ptr<short>(i-1);

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
        neighbors = &g[i].front();
        for(j=0;j<g[i].size();j++)
        {
            found = false;

            for(int k=0;k<g[neighbors[j].label].size();k++)
            {
                if(g[neighbors[j].label][k].label==i)
                {
                    neighbors[j].dist = g[neighbors[j].label][k].dist = min(neighbors[j].dist,g[neighbors[j].label][k].dist);
                    found = true;
                    break;
                }
            }

            if(!found)
                g[neighbors[j].label].push_back(node(i,neighbors[j].dist));
        }        
    }

    //Step 2: for each match find k nearest-neighbor matches with corresponding distances 
    //        using Dijkstra algorithm

    nodeHeap q(match_num);
    int num_expanded_vertices;
    bitset<SHRT_MAX> expanded_flag;
    dstNNlabels = Scalar(-1);
    dstNNdistances = Scalar(0.0f);

    for(i=0;i<match_num;i++)
    {
        if(g[i].empty())
            continue;
        //find k closest matches:
        num_expanded_vertices = 0;
        expanded_flag.reset();
        q.clear();
        q.add(node(i,0.0f));
        short* NNlabels_row = dstNNlabels.ptr<short>(i);
        float* NNdistances_row = dstNNdistances.ptr<float>(i);
        while(num_expanded_vertices<k && !q.empty())
        {
            node vert_for_expansion = q.getMin();
            expanded_flag[vert_for_expansion.label] = 1;

            //write the expanded vertex to the dst:
            NNlabels_row[num_expanded_vertices] = vert_for_expansion.label;
            NNdistances_row[num_expanded_vertices] = vert_for_expansion.dist;
            num_expanded_vertices++;

            //update the heap:
            neighbors = &g[vert_for_expansion.label].front();
            for(j=0;j<g[vert_for_expansion.label].size();j++)
            {
                if(!expanded_flag[neighbors[j].label])
                    q.updateNode(node(neighbors[j].label,vert_for_expansion.dist+neighbors[j].dist));
            }
        }
    }
}

void weightedLeastSquaresAffineFit(short* labels, float* weights, int count, SparseMatch* matches, Mat& dst)
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

    sa[3][4] = sa[4][3] = sa[1][0] = sa[0][1];
    sa[3][5] = sa[5][3] = sa[2][0] = sa[0][2];
    sa[4][5] = sa[5][4] = sa[2][1] = sa[1][2];

    sa[3][3] = sa[0][0];
    sa[4][4] = sa[1][1];
    sa[5][5] = sa[2][2];

    bool res = solve(A, B, MM, DECOMP_EIG);
    MM.reshape(2,3).convertTo(dst,CV_32F);
}

void EdgeAwareInterpolatorImpl::leastSquaresInterpolation(Mat& labels, Mat& NNlabels, Mat& NNdistances, vector<SparseMatch>& matches, Mat& dst_dense_flow)
{
    // for each match compute an affine transform with distance-weighted least squares and
    // fill the the voronoi diagram cell correspondingly
    int n = matches.size();
    int k = NNlabels.cols;
    float sigma = 0.05;
    short* KNNlabels;
    float* KNNdistances;
    NNdistances *= (-sigma*sigma);
    Mat* transforms = new Mat[n];
    
    //fit all the local affine transforms:
    for(int i=0;i<n;i++)
    {
        KNNlabels     = NNlabels.ptr<short>(i);
        KNNdistances  = NNdistances.ptr<float>(i);
        hal::exp(KNNdistances,KNNdistances,k);
        
        weightedLeastSquaresAffineFit(KNNlabels,KNNdistances,k,&matches.front(),transforms[i]);
    }

    //construct the final piecewise-affine interpolation:
    short* label_row;
    float* tr;
    for(int i=0;i<h;i++)
    {
        label_row = labels.ptr<short>(i);
        Point2f* dst_row = dst_dense_flow.ptr<Point2f>(i); 
        for(int j=0;j<w;j++)
        {
            tr = transforms[label_row[j]].ptr<float>(0);
            dst_row[j] = Point2f(tr[0]*j+tr[1]*i+tr[2],tr[3]*j+tr[4]*i+tr[5]) - Point2f((float)j,(float)i);
        }
    }
    delete[] transforms;
}

void generateHypothesis(short* labels, int count, RNG& rng, bitset<SHRT_MAX>& is_used, SparseMatch* matches, Mat& dst)
{
    int idx;
    Point2f src_points[3];
    Point2f dst_points[3];
    is_used.reset();

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

void verifyHypothesis(short* labels, float* weights, int count, SparseMatch* matches, float eps, Mat& hypothesis_transform, Mat& old_transform, float& old_weighted_num_inliers)
{
    float* tr = hypothesis_transform.ptr<float>(0);
    Point2f a,b;
    float weighted_num_inliers = 0;
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

void EdgeAwareInterpolatorImpl::ransacInterpolation(Mat& labels, Mat& NNlabels, Mat& NNdistances, vector<SparseMatch>& matches, Mat& dst_dense_flow, int num_iter)
{
    int n = matches.size();
    int k = NNlabels.cols;
    float sigma = 0.05f;
    float inlier_eps = 2.0f;
    bitset<SHRT_MAX> is_used;
    short* KNNlabels;
    float* KNNdistances;
    NNdistances *= (-sigma*sigma);

    Mat* transforms = new Mat[n];
    float* weighted_inlier_nums = new float[n];
    memset(weighted_inlier_nums,0,n*sizeof(float));
    Mat hypothesis_transform;
    RNG rng(0);

    short* inlier_labels = new short[k];
    float* inlier_distances = new float[k];
    float* tr;
    int num_inliers;
    Point2f a,b;

    int start_i,end_i,inc;

    for(int it=0;it<num_iter;it++)
    {
        if(it%2==0)
        {
            start_i = 0;
            end_i = n;
            inc = 1;
        }
        else
        {
            start_i = n-1;
            end_i = -1;
            inc = -1;
        }

        for(int i=start_i;i!=end_i;i+=inc)
        {
            if(g[i].empty())
                continue;

            KNNlabels = NNlabels.ptr<short>(i);
            KNNdistances = NNdistances.ptr<float>(i);
            if(it==0)
                hal::exp(KNNdistances,KNNdistances,k);

            generateHypothesis(KNNlabels,k,rng,is_used,&matches.front(),hypothesis_transform);
            verifyHypothesis(KNNlabels,KNNdistances,k,&matches.front(),inlier_eps,hypothesis_transform,transforms[i],weighted_inlier_nums[i]);

            //propagate hypotheses from neighbors:
            node* neighbors = &g[i].front();
            for(int j=0;j<g[i].size();j++)
            {
                if((inc*neighbors[j].label)<(inc*i)) //already processed this neighbor
                    verifyHypothesis(KNNlabels,KNNdistances,k,&matches.front(),inlier_eps,transforms[neighbors[j].label],transforms[i],weighted_inlier_nums[i]);
            }

            if(it==num_iter-1)
            {
                // determine inliers and compute a least squares fit:
                tr = transforms[i].ptr<float>(0);
                num_inliers = 0;
                
                for(int j=0;j<k;j++)
                {
                    a = matches[KNNlabels[j]].reference_image_pos;
                    b = matches[KNNlabels[j]].target_image_pos;
                    if(abs(tr[0]*a.x + tr[1]*a.y + tr[2] - b.x) +
                       abs(tr[3]*a.x + tr[4]*a.y + tr[5] - b.y) < inlier_eps)
                    {
                        inlier_labels[num_inliers] = KNNlabels[j];
                        inlier_distances[num_inliers] = KNNdistances[j];
                        num_inliers++;
                    }
                }

                weightedLeastSquaresAffineFit(inlier_labels,inlier_distances,num_inliers,&matches.front(),transforms[i]);
            }
        }
    }

    //construct the final piecewise-affine interpolation:
    short* label_row;
    for(int i=0;i<h;i++)
    {
        label_row = labels.ptr<short>(i);
        Point2f* dst_row = dst_dense_flow.ptr<Point2f>(i); 
        for(int j=0;j<w;j++)
        {
            tr = transforms[label_row[j]].ptr<float>(0);
            dst_row[j] = Point2f(tr[0]*j+tr[1]*i+tr[2],tr[3]*j+tr[4]*i+tr[5]) - Point2f((float)j,(float)i);
        }
    }

    delete[] transforms;
    delete[] weighted_inlier_nums;
    delete[] inlier_labels;
    delete[] inlier_distances;
}

CV_EXPORTS_W
Ptr<EdgeAwareInterpolator> createEdgeAwareInterpolator(bool useSED)
{
    return Ptr<EdgeAwareInterpolator>(EdgeAwareInterpolatorImpl::create(useSED));
}

}
}