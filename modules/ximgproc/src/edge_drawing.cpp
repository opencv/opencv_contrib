// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "precomp.hpp"
#include "edge_drawing_common.hpp"

using namespace std;

namespace cv
{
namespace ximgproc
{

struct ComputeGradientBody : ParallelLoopBody
{
void operator() (const Range& range) const CV_OVERRIDE;

Mat_<uchar> src;
mutable Mat_<ushort> gradImage;
mutable Mat_<uchar> dirImage;
int gradThresh;
int op;
bool SumFlag;
int* grads;
bool PFmode;
};

void ComputeGradientBody::operator() (const Range& range) const
{
    const int last_col = src.cols - 1;
    int gx = 0;
    int gy = 0;
    int sum;

    for (int y = range.start; y < range.end; ++y)
    {
        const uchar* srcPrevRow = src[y - 1];
        const uchar* srcCurRow = src[y];
        const uchar* srcNextRow = src[y + 1];

        ushort* gradRow = gradImage[y];
        uchar* dirRow = dirImage[y];

        for (int x = 1; x < last_col; ++x)
        {
            int com1 = srcNextRow[x + 1] - srcPrevRow[x - 1];
            int com2 = srcPrevRow[x + 1] - srcNextRow[x - 1];

            switch (op)
            {
            case EdgeDrawing::PREWITT:
                gx = abs(com1 + com2 + srcCurRow[x + 1] - srcCurRow[x - 1]);
                gy = abs(com1 - com2 + srcNextRow[x] - srcPrevRow[x]);
                break;
            case EdgeDrawing::SOBEL:
                gx = abs(com1 + com2 + 2 * (srcCurRow[x + 1] - srcCurRow[x - 1]));
                gy = abs(com1 - com2 + 2 * (srcNextRow[x] - srcPrevRow[x]));
                break;
            case EdgeDrawing::SCHARR:
                gx = abs(3 * (com1 + com2) + 10 * (srcCurRow[x + 1] - srcCurRow[x - 1]));
                gy = abs(3 * (com1 - com2) + 10 * (srcNextRow[x] - srcPrevRow[x]));
                break;
            case EdgeDrawing::LSD:
                // com1 and com2 differs from previous operators, because LSD has 2x2 kernel
                com1 = srcNextRow[x + 1] - srcCurRow[x];
                com2 = srcCurRow[x + 1] - srcNextRow[x];

                gx = abs(com1 + com2);
                gy = abs(com1 - com2);
                break;
            }

            if (SumFlag)
                sum = gx + gy;
            else
                sum = (int)sqrt((double)gx * gx + gy * gy);

            gradRow[x] = (ushort)sum;

            if (PFmode)
                grads[sum]++;

            if (sum >= gradThresh)
            {
                if (gx >= gy)
                    dirRow[x] = EDGE_VERTICAL;
                else
                    dirRow[x] = EDGE_HORIZONTAL;
            }
        }
    }
}

class EdgeDrawingImpl : public EdgeDrawing
{
public:

    enum EllipseFittingMethods
    {
        BOOKSTEIN = 0,
        FPF = 1
    };

    EdgeDrawingImpl();
    ~EdgeDrawingImpl();
    void detectEdges(InputArray src) CV_OVERRIDE;
    void getEdgeImage(OutputArray dst) CV_OVERRIDE;
    void getGradientImage(OutputArray dst) CV_OVERRIDE;

    vector<vector<Point> > getSegments() CV_OVERRIDE;
    vector<int> getSegmentIndicesOfLines() const CV_OVERRIDE;
    void detectLines(OutputArray lines) CV_OVERRIDE;
    void detectEllipses(OutputArray ellipses) CV_OVERRIDE;

    virtual String getDefaultName() const CV_OVERRIDE;
    virtual void read(const FileNode& fn) CV_OVERRIDE;
    virtual void write(FileStorage& fs) const CV_OVERRIDE;

protected:
    int width;        // width of source image
    int height;       // height of source image
    uchar *srcImg;
    vector<vector<Point> > segmentPoints;
    vector<int> segmentIndicesOfLines;
    Mat smoothImage;
    uchar *edgeImg;   // pointer to edge image data
    uchar *smoothImg; // pointer to smoothed image data
    int segmentNos;
    Mat srcImage;

    double divForTestSegment;
    double* dH;
    int* grads;
    int np;

private:
    void ComputeGradient();
    void ComputeAnchorPoints();
    void JoinAnchorPointsUsingSortedAnchors();
    int* sortAnchorsByGradValue1();

    static int LongestChain(Chain *chains, int root);
    static int RetrieveChainNos(Chain *chains, int root, int chainNos[]);

    int anchorNos;
    vector<Point> anchorPoints;
    vector<Point> edgePoints;

    Mat edgeImage;
    Mat gradImage;
    Mat dirImage;
    uchar *dirImg;    // pointer to direction image data
    ushort *gradImg;   // pointer to gradient image data

    int op;           // edge detection operator
    int gradThresh;   // gradient threshold
    int anchorThresh; // anchor point threshold

    std::vector<EDLineSegment> lines;
    int linesNo;
    int min_line_len;
    double line_error;
    double max_distance_between_two_lines;
    double max_error;
    double precision;
    NFALUT* nfa;

    int ComputeMinLineLength();
    void SplitSegment2Lines(double* x, double* y, int noPixels, int segmentNo);
    void JoinCollinearLines();

    void ValidateLineSegments();
    bool ValidateLineSegmentRect(int* x, int* y, EDLineSegment* ls);
    bool TryToJoinTwoLineSegments(EDLineSegment* ls1, EDLineSegment* ls2, int changeIndex);

    static double ComputeMinDistance(double x1, double y1, double a, double b, int invert);
    static void ComputeClosestPoint(double x1, double y1, double a, double b, int invert, double& xOut, double& yOut);
    static void LineFit(double* x, double* y, int count, double& a, double& b, int invert);
    static void LineFit(double* x, double* y, int count, double& a, double& b, double& e, int& invert);
    static double ComputeMinDistanceBetweenTwoLines(EDLineSegment* ls1, EDLineSegment* ls2, int* pwhich);
    static void UpdateLineParameters(EDLineSegment* ls);
    static void EnumerateRectPoints(double sx, double sy, double ex, double ey, int ptsx[], int ptsy[], int* pNoPoints);

    void TestSegment(int i, int index1, int index2);
    void ExtractNewSegments();
    double NFA(double prob, int len);

    int noEllipses;
    int noCircles;
    std::vector<mCircle> Circles;
    std::vector<mEllipse> Ellipses;

    Circle* circles1;
    Circle* circles2;
    Circle* circles3;
    int noCircles1;
    int noCircles2;
    int noCircles3;

    EDArcs* edarcs1;
    EDArcs* edarcs2;
    EDArcs* edarcs3;
    EDArcs* edarcs4;

    int* segmentStartLines;
    BufferManager* bm;
    Info* info;

    void GenerateCandidateCircles();
    void DetectArcs();
    void ValidateCircles(bool validate);
    void JoinCircles();
    void JoinArcs1();
    void JoinArcs2();
    void JoinArcs3();

    // circle utility functions
    static void addCircle(Circle* circles, int& noCircles, double xc, double yc, double r, double circleFitError, double* x, double* y, int noPixels);
    static void addCircle(Circle* circles, int& noCircles, double xc, double yc, double r, double circleFitError, EllipseEquation* pEq, double ellipseFitError, double* x, double* y, int noPixels);
    static void sortCircles(Circle* circles, int noCircles);
    static bool CircleFit(double* x, double* y, int N, double* pxc, double* pyc, double* pr, double* pe);
    static void ComputeCirclePoints(double xc, double yc, double r, double* px, double* py, int* noPoints);

    // ellipse utility functions
    static bool EllipseFit(double* x, double* y, int noPoints, EllipseEquation* pResult, int mode = FPF);
    static double** AllocateMatrix(int noRows, int noColumns);
    static void A_TperB(double** A_, double** B_, double** _res, int _righA, int _colA, int _righB, int _colB);
    static void choldc(double** a, int n, double** l);
    static int inverse(double** TB, double** InvB, int N);
    static void DeallocateMatrix(double** m, int noRows);
    static void AperB_T(double** A_, double** B_, double** _res, int _righA, int _colA, int _righB, int _colB);
    static void AperB(double** A_, double** B_, double** _res, int _righA, int _colA, int _righB, int _colB);
    static void jacobi(double** a, int n, double d[], double** v);
    static void ROTATE(double** a, int i, int j, int k, int l, double tau, double s);
    static double computeEllipsePerimeter(EllipseEquation* eq);
    static double ComputeEllipseError(EllipseEquation* eq, double* px, double* py, int noPoints);
    static double ComputeEllipseCenterAndAxisLengths(EllipseEquation* eq, double* pxc, double* pyc, double* pmajorAxisLength, double* pminorAxisLength);
    static void ComputeEllipsePoints(double* pvec, double* px, double* py, int noPoints);

    // arc utility functions
    static void joinLastTwoArcs(MyArc* arcs, int& noArcs);
    static void addArc(MyArc* arcs, int& noArchs, double xc, double yc, double r, double circleFitError, // Circular arc
        double sTheta, double eTheta, int turn, int segmentNo,
        int sx, int sy, int ex, int ey,
        double* x, double* y, int noPixels, double overlapRatio = 0.0);
    static void addArc(MyArc* arcs, int& noArchs, double xc, double yc, double r, double circleFitError, // Elliptic arc
        double sTheta, double eTheta, int turn, int segmentNo,
        EllipseEquation* pEq, double ellipseFitError,
        int sx, int sy, int ex, int ey,
        double* x, double* y, int noPixels, double overlapRatio = 0.0);

    static void ComputeStartAndEndAngles(double xc, double yc, double r,
        double* x, double* y, int len,
        double* psTheta, double* peTheta);

    static void sortArc(MyArc* arcs, int noArcs);
};

Ptr<EdgeDrawing> createEdgeDrawing()
{
    return makePtr<EdgeDrawingImpl>();
}

EdgeDrawing::Params::Params()
{
    PFmode = false;
    EdgeDetectionOperator = PREWITT;
    GradientThresholdValue = 20;
    AnchorThresholdValue = 0;
    ScanInterval = 1;
    MinPathLength = 10;
    Sigma = 1.0;
    SumFlag = true;
    NFAValidation = true;
    MinLineLength = -1;
    MaxDistanceBetweenTwoLines = 6.0;
    LineFitErrorThreshold = 1.0;
    MaxErrorThreshold = 1.3;
}

void EdgeDrawing::setParams(const EdgeDrawing::Params& parameters)
{
    params = parameters;
}

void EdgeDrawing::Params::read(const cv::FileNode& fn)
{
    PFmode = (int)fn["PFmode"] != 0 ? true : false;
    EdgeDetectionOperator = fn["EdgeDetectionOperator"];
    GradientThresholdValue = fn["GradientThresholdValue"];
    AnchorThresholdValue = fn["AnchorThresholdValue"];
    ScanInterval = fn["ScanInterval"];
    MinPathLength = fn["MinPathLength"];
    Sigma = fn["Sigma"];
    SumFlag = (int)fn["SumFlag"] != 0 ? true : false;
    NFAValidation = (int)fn["NFAValidation"] != 0 ? true : false;
    MinLineLength = fn["MinLineLength"];
    MaxDistanceBetweenTwoLines = fn["MaxDistanceBetweenTwoLines"];
    LineFitErrorThreshold = fn["LineFitErrorThreshold"];
    MaxErrorThreshold = fn["MaxErrorThreshold"];
}

void EdgeDrawing::Params::write(cv::FileStorage& fs) const
{
    fs << "PFmode" << PFmode;
    fs << "EdgeDetectionOperator" << EdgeDetectionOperator;
    fs << "GradientThresholdValue" << GradientThresholdValue;
    fs << "AnchorThresholdValue" << AnchorThresholdValue;
    fs << "ScanInterval" << ScanInterval;
    fs << "MinPathLength" << MinPathLength;
    fs << "Sigma" << Sigma;
    fs << "SumFlag" << SumFlag;
    fs << "NFAValidation" << NFAValidation;
    fs << "MinLineLength" << MinLineLength;
    fs << "MaxDistanceBetweenTwoLines" << MaxDistanceBetweenTwoLines;
    fs << "LineFitErrorThreshold" << LineFitErrorThreshold;
    fs << "MaxErrorThreshold" << MaxErrorThreshold;
}

String EdgeDrawingImpl::getDefaultName() const
{
    return String("EdgeDrawing");
}

void EdgeDrawingImpl::read(const cv::FileNode& fn)
{
    params.read(fn);
}

void EdgeDrawingImpl::write(cv::FileStorage& fs) const
{
    writeFormat(fs);
    params.write(fs);
}

EdgeDrawingImpl::EdgeDrawingImpl()
{
    params = EdgeDrawing::Params();
    nfa = new NFALUT(1, 1/2, 1, 1);
    dH = new double[MAX_GRAD_VALUE];
    grads = new int[MAX_GRAD_VALUE];
}

EdgeDrawingImpl::~EdgeDrawingImpl()
{
    delete nfa;
    delete[] dH;
    delete[] grads;
}

void EdgeDrawingImpl::detectEdges(InputArray src)
{
    CV_Assert(!src.empty() && src.type() == CV_8UC1);
    gradThresh = params.GradientThresholdValue;
    anchorThresh = params.AnchorThresholdValue;
    op = params.EdgeDetectionOperator;

    // Check parameters for sanity
    if (op < 0 || op > 3)
        op = 0;

    if (gradThresh < 1)
        gradThresh = 1;

    if (anchorThresh < 0)
        anchorThresh = 0;

    segmentNos = 0;
    anchorNos = 0;
    anchorPoints.clear();
    lines.clear();
    segmentPoints.clear();
    segmentPoints.push_back(vector<Point>()); // create empty vector of points for segments
    srcImage = src.getMat();
    srcImg = srcImage.data;
    height = srcImage.rows;
    width = srcImage.cols;

    edgeImage = Mat(height, width, CV_8UC1, Scalar(0)); // initialize edge Image
    gradImage = Mat(height, width, CV_16UC1); // gradImage contains short values
    dirImage = Mat(height, width, CV_8UC1);

    if (params.Sigma < 1.0)
        smoothImage = srcImage;
    else if (params.Sigma == 1.0)
        GaussianBlur(srcImage, smoothImage, Size(5, 5), params.Sigma);
    else
        GaussianBlur(srcImage, smoothImage, Size(), params.Sigma); // calculate kernel from sigma

    // Assign Pointers from Mat's data
    smoothImg = smoothImage.data;
    gradImg = (ushort*)gradImage.data;
    edgeImg = edgeImage.data;
    dirImg = dirImage.data;

    if (params.PFmode)
    {
        memset(dH, 0, sizeof(double) * MAX_GRAD_VALUE);
        memset(grads, 0, sizeof(int) * MAX_GRAD_VALUE);
    }

    ComputeGradient();                    // COMPUTE GRADIENT & EDGE DIRECTION MAPS
    ComputeAnchorPoints();                // COMPUTE ANCHORS
    JoinAnchorPointsUsingSortedAnchors(); // JOIN ANCHORS

    if (params.PFmode)
    {
        // Compute probability function H
        int size = (width - 2) * (height - 2);

        for (int i = MAX_GRAD_VALUE - 1; i > 0; i--)
            grads[i - 1] += grads[i];

        for (int i = 0; i < MAX_GRAD_VALUE; i++)
            dH[i] = (double)grads[i] / ((double)size);

        divForTestSegment = 2.25; // Some magic number :-)
        memset(edgeImg, 0, width * height); // clear edge image
        np = 0;
        for (int i = 0; i < segmentNos; i++)
        {
            int len = (int)segmentPoints[i].size();
            np += (len * (len - 1)) / 2;
        }

        // Validate segments
        for (int i = 0; i < segmentNos; i++)
            TestSegment(i, 0, (int)segmentPoints[i].size() - 1);

        ExtractNewSegments();
    }
}

void EdgeDrawingImpl::getEdgeImage(OutputArray _dst)
{
    if (!edgeImage.empty())
        edgeImage.copyTo(_dst);
}

void EdgeDrawingImpl::getGradientImage(OutputArray _dst)
{
    if (!gradImage.empty())
        gradImage.copyTo(_dst);
}

std::vector<std::vector<Point> > EdgeDrawingImpl::getSegments()
{
    return segmentPoints;
}

std::vector<int> EdgeDrawingImpl::getSegmentIndicesOfLines() const
{
    return segmentIndicesOfLines;
}

void EdgeDrawingImpl::ComputeGradient()
{
    for (int j = 0; j < width; j++)
    {
        gradImg[j] = gradImg[(height - 1) * width + j] = (ushort)gradThresh - 1;
    }

    for (int i = 1; i < height - 1; i++)
    {
        gradImg[i * width] = gradImg[(i + 1) * width - 1] = (ushort)gradThresh - 1;
    }

    ComputeGradientBody body;
    body.src = smoothImage;
    body.gradImage = gradImage;
    body.dirImage = dirImage;
    body.gradThresh = gradThresh;
    body.SumFlag = params.SumFlag;
    body.op = op;
    body.grads = grads;
    body.PFmode = params.PFmode;

    parallel_for_(Range(1, smoothImage.rows - 1), body);
}

void EdgeDrawingImpl::ComputeAnchorPoints()
{
    for (int i = 2; i < height - 2; i++)
    {
        int start = 2;
        int inc = 1;
        if (i % params.ScanInterval != 0)
        {
            start = params.ScanInterval;
            inc = params.ScanInterval;
        }

        for (int j = start; j < width - 2; j += inc)
        {
            if (gradImg[i * width + j] < gradThresh)
                continue;

            if (dirImg[i * width + j] == EDGE_VERTICAL)
            {
                // vertical edge
                int diff1 = gradImg[i * width + j] - gradImg[i * width + j - 1];
                int diff2 = gradImg[i * width + j] - gradImg[i * width + j + 1];
                if (diff1 >= anchorThresh && diff2 >= anchorThresh)
                {
                    edgeImg[i * width + j] = ANCHOR_PIXEL;
                    anchorPoints.push_back(Point(j, i));
                }
            }
            else
            {
                // horizontal edge
                int diff1 = gradImg[i * width + j] - gradImg[(i - 1) * width + j];
                int diff2 = gradImg[i * width + j] - gradImg[(i + 1) * width + j];
                if (diff1 >= anchorThresh && diff2 >= anchorThresh)
                {
                    edgeImg[i * width + j] = ANCHOR_PIXEL;
                    anchorPoints.push_back(Point(j, i));
                }
            }
        }
    }

    anchorNos = (int)anchorPoints.size(); // get the total number of anchor points
}

void EdgeDrawingImpl::JoinAnchorPointsUsingSortedAnchors()
{
    int* chainNos = new int[(width + height) * 8];

    Point* pixels = new Point[width * height];
    StackNode* stack = new StackNode[width * height];
    Chain* chains = new Chain[width * height];

    // sort the anchor points by their gradient value in decreasing order
    int* pAnchors = sortAnchorsByGradValue1();

    // Now join the anchors starting with the anchor having the greatest gradient value

    for (int k0 = anchorNos - 1; k0 >= 0; k0--)
    {
        int pixelOffset = pAnchors[k0];

        int i = pixelOffset / width;
        int j = pixelOffset % width;

        if (edgeImg[i * width + j] != ANCHOR_PIXEL)
            continue;

        chains[0].len = 0;
        chains[0].parent = -1;
        chains[0].dir = 0;
        chains[0].children[0] = chains[0].children[1] = -1;
        chains[0].pixels = NULL;

        int noChains = 1;
        int len = 0;
        int duplicatePixelCount = 0;
        int top = -1;  // top of the stack

        if (dirImg[i * width + j] == EDGE_VERTICAL)
        {
            stack[++top].r = i;
            stack[top].c = j;
            stack[top].dir = DOWN;
            stack[top].parent = 0;

            stack[++top].r = i;
            stack[top].c = j;
            stack[top].dir = UP;
            stack[top].parent = 0;
        }
        else
        {
            stack[++top].r = i;
            stack[top].c = j;
            stack[top].dir = RIGHT;
            stack[top].parent = 0;

            stack[++top].r = i;
            stack[top].c = j;
            stack[top].dir = LEFT;
            stack[top].parent = 0;
        }

        // While the stack is not empty
StartOfWhile:
        while (top >= 0)
        {
            int r = stack[top].r;
            int c = stack[top].c;
            int dir = stack[top].dir;
            int parent = stack[top].parent;
            top--;

            if (edgeImg[r * width + c] != EDGE_PIXEL)
                duplicatePixelCount++;

            chains[noChains].dir = dir;   // traversal direction
            chains[noChains].parent = parent;
            chains[noChains].children[0] = chains[noChains].children[1] = -1;

            int chainLen = 0;

            chains[noChains].pixels = &pixels[len];

            pixels[len].y = r;
            pixels[len].x = c;
            len++;
            chainLen++;

            if (dir == LEFT)
            {
                while (dirImg[r * width + c] == EDGE_HORIZONTAL)
                {
                    edgeImg[r * width + c] = EDGE_PIXEL;

                    // The edge is horizontal. Look LEFT
                    //
                    //   A
                    //   B x
                    //   C
                    //
                    // cleanup up & down pixels
                    if (edgeImg[(r - 1) * width + c] == ANCHOR_PIXEL)
                        edgeImg[(r - 1) * width + c] = 0;
                    if (edgeImg[(r + 1) * width + c] == ANCHOR_PIXEL)
                        edgeImg[(r + 1) * width + c] = 0;

                    // Look if there is an edge pixel in the neighbors
                    if (edgeImg[r * width + c - 1] >= ANCHOR_PIXEL)
                    {
                        c--;
                    }
                    else if (edgeImg[(r - 1) * width + c - 1] >= ANCHOR_PIXEL)
                    {
                        r--;
                        c--;
                    }
                    else if (edgeImg[(r + 1) * width + c - 1] >= ANCHOR_PIXEL)
                    {
                        r++;
                        c--;
                    }
                    else
                    {
                        // else -- follow max. pixel to the LEFT
                        int A = gradImg[(r - 1) * width + c - 1];
                        int B = gradImg[r * width + c - 1];
                        int C = gradImg[(r + 1) * width + c - 1];

                        if (A > B)
                        {
                            if (A > C)
                                r--;
                            else
                                r++;
                        }
                        else  if (C > B)
                            r++;
                        c--;
                    }

                    if (edgeImg[r * width + c] == EDGE_PIXEL || gradImg[r * width + c] < gradThresh)
                    {
                        if (chainLen > 0)
                        {
                            chains[noChains].len = chainLen;
                            chains[parent].children[0] = noChains;
                            noChains++;
                        }
                        goto StartOfWhile;
                    }

                    pixels[len].y = r;
                    pixels[len].x = c;
                    len++;
                    chainLen++;
                }

                stack[++top].r = r;
                stack[top].c = c;
                stack[top].dir = DOWN;
                stack[top].parent = noChains;

                stack[++top].r = r;
                stack[top].c = c;
                stack[top].dir = UP;
                stack[top].parent = noChains;

                len--;
                chainLen--;

                chains[noChains].len = chainLen;
                chains[parent].children[0] = noChains;
                noChains++;
            }
            else if (dir == RIGHT)
            {
                while (dirImg[r * width + c] == EDGE_HORIZONTAL)
                {
                    edgeImg[r * width + c] = EDGE_PIXEL;

                    // The edge is horizontal. Look RIGHT
                    //
                    //     A
                    //   x B
                    //     C
                    //
                    // cleanup up&down pixels
                    if (edgeImg[(r + 1) * width + c] == ANCHOR_PIXEL)
                        edgeImg[(r + 1) * width + c] = 0;
                    if (edgeImg[(r - 1) * width + c] == ANCHOR_PIXEL)
                        edgeImg[(r - 1) * width + c] = 0;

                    // Look if there is an edge pixel in the neighbors
                    if (edgeImg[r * width + c + 1] >= ANCHOR_PIXEL)
                    {
                        c++;
                    }
                    else if (edgeImg[(r + 1) * width + c + 1] >= ANCHOR_PIXEL)
                    {
                        r++;
                        c++;
                    }
                    else if (edgeImg[(r - 1) * width + c + 1] >= ANCHOR_PIXEL)
                    {
                        r--;
                        c++;
                    }
                    else
                    {
                        // else -- follow max. pixel to the RIGHT
                        int A = gradImg[(r - 1) * width + c + 1];
                        int B = gradImg[r * width + c + 1];
                        int C = gradImg[(r + 1) * width + c + 1];

                        if (A > B)
                        {
                            if (A > C)
                                r--;       // A
                            else
                                r++;       // C
                        }
                        else if (C > B)
                            r++;  // C
                        c++;
                    }

                    if (edgeImg[r * width + c] == EDGE_PIXEL || gradImg[r * width + c] < gradThresh)
                    {
                        if (chainLen > 0)
                        {
                            chains[noChains].len = chainLen;
                            chains[parent].children[1] = noChains;
                            noChains++;
                        }
                        goto StartOfWhile;
                    }

                    pixels[len].y = r;
                    pixels[len].x = c;
                    len++;
                    chainLen++;
                }

                stack[++top].r = r;
                stack[top].c = c;
                stack[top].dir = DOWN;  // Go down
                stack[top].parent = noChains;

                stack[++top].r = r;
                stack[top].c = c;
                stack[top].dir = UP;   // Go up
                stack[top].parent = noChains;

                len--;
                chainLen--;

                chains[noChains].len = chainLen;
                chains[parent].children[1] = noChains;
                noChains++;

            }
            else if (dir == UP)
            {
                while (dirImg[r * width + c] == EDGE_VERTICAL)
                {
                    edgeImg[r * width + c] = EDGE_PIXEL;

                    // The edge is vertical. Look UP
                    //
                    //   A B C
                    //     x
                    //
                    // Cleanup left & right pixels
                    if (edgeImg[r * width + c - 1] == ANCHOR_PIXEL)
                        edgeImg[r * width + c - 1] = 0;
                    if (edgeImg[r * width + c + 1] == ANCHOR_PIXEL)
                        edgeImg[r * width + c + 1] = 0;

                    // Look if there is an edge pixel in the neighbors
                    if (edgeImg[(r - 1) * width + c] >= ANCHOR_PIXEL)
                    {
                        r--;
                    }
                    else if (edgeImg[(r - 1) * width + c - 1] >= ANCHOR_PIXEL)
                    {
                        r--;
                        c--;
                    }
                    else if (edgeImg[(r - 1) * width + c + 1] >= ANCHOR_PIXEL)
                    {
                        r--;
                        c++;
                    }
                    else
                    {
                        // else -- follow the max. pixel UP
                        int A = gradImg[(r - 1) * width + c - 1];
                        int B = gradImg[(r - 1) * width + c];
                        int C = gradImg[(r - 1) * width + c + 1];

                        if (A > B)
                        {
                            if (A > C)
                                c--;
                            else
                                c++;
                        }
                        else if (C > B)
                            c++;
                        r--;
                    }

                    if (edgeImg[r * width + c] == EDGE_PIXEL || gradImg[r * width + c] < gradThresh)
                    {
                        if (chainLen > 0)
                        {
                            chains[noChains].len = chainLen;
                            chains[parent].children[0] = noChains;
                            noChains++;
                        }
                        goto StartOfWhile;
                    }

                    pixels[len].y = r;
                    pixels[len].x = c;

                    len++;
                    chainLen++;
                }

                stack[++top].r = r;
                stack[top].c = c;
                stack[top].dir = RIGHT;
                stack[top].parent = noChains;

                stack[++top].r = r;
                stack[top].c = c;
                stack[top].dir = LEFT;
                stack[top].parent = noChains;

                len--;
                chainLen--;

                chains[noChains].len = chainLen;
                chains[parent].children[0] = noChains;
                noChains++;
            }
            else   // dir == DOWN
            {
                while (dirImg[r * width + c] == EDGE_VERTICAL)
                {
                    edgeImg[r * width + c] = EDGE_PIXEL;

                    // The edge is vertical
                    //
                    //     x
                    //   A B C
                    //
                    // cleanup side pixels
                    if (edgeImg[r * width + c + 1] == ANCHOR_PIXEL)
                        edgeImg[r * width + c + 1] = 0;
                    if (edgeImg[r * width + c - 1] == ANCHOR_PIXEL)
                        edgeImg[r * width + c - 1] = 0;

                    // Look if there is an edge pixel in the neighbors
                    if (edgeImg[(r + 1) * width + c] >= ANCHOR_PIXEL)
                    {
                        r++;
                    }
                    else if (edgeImg[(r + 1) * width + c + 1] >= ANCHOR_PIXEL)
                    {
                        r++;
                        c++;
                    }
                    else if (edgeImg[(r + 1) * width + c - 1] >= ANCHOR_PIXEL)
                    {
                        r++;
                        c--;
                    }
                    else
                    {
                        // else -- follow the max. pixel DOWN
                        int A = gradImg[(r + 1) * width + c - 1];
                        int B = gradImg[(r + 1) * width + c];
                        int C = gradImg[(r + 1) * width + c + 1];

                        if (A > B)
                        {
                            if (A > C)
                                c--;       // A
                            else
                                c++;       // C
                        }
                        else if (C > B)
                            c++;  // C
                        r++;
                    }

                    if (edgeImg[r * width + c] == EDGE_PIXEL || gradImg[r * width + c] < gradThresh)
                    {
                        if (chainLen > 0)
                        {
                            chains[noChains].len = chainLen;
                            chains[parent].children[1] = noChains;
                            noChains++;
                        }
                        goto StartOfWhile;
                    }

                    pixels[len].y = r;
                    pixels[len].x = c;

                    len++;
                    chainLen++;
                }

                stack[++top].r = r;
                stack[top].c = c;
                stack[top].dir = RIGHT;
                stack[top].parent = noChains;

                stack[++top].r = r;
                stack[top].c = c;
                stack[top].dir = LEFT;
                stack[top].parent = noChains;

                len--;
                chainLen--;

                chains[noChains].len = chainLen;
                chains[parent].children[1] = noChains;
                noChains++;
            }
        }

        if (len - duplicatePixelCount < params.MinPathLength)
        {
            for (int k1 = 0; k1 < len; k1++)
            {
                edgeImg[pixels[k1].y * width + pixels[k1].x] = 0;
                edgeImg[pixels[k1].y * width + pixels[k1].x] = 0;
            }
        }
        else
        {
            int noSegmentPixels = 0;
            int totalLen = LongestChain(chains, chains[0].children[1]);

            if (totalLen > 0)
            {
                // Retrieve the chainNos
                int count = RetrieveChainNos(chains, chains[0].children[1], chainNos);

                // Copy these pixels in the reverse order
                for (int k2 = count - 1; k2 >= 0; k2--)
                {
                    int chainNo = chainNos[k2];

                    /* See if we can erase some pixels from the last chain. This is for cleanup */

                    int fr = chains[chainNo].pixels[chains[chainNo].len - 1].y;
                    int fc = chains[chainNo].pixels[chains[chainNo].len - 1].x;

                    int index = noSegmentPixels - 2;
                    while (index >= 0)
                    {
                        int dr = abs(fr - segmentPoints[segmentNos][index].y);
                        int dc = abs(fc - segmentPoints[segmentNos][index].x);

                        if (dr <= 1 && dc <= 1)
                        {
                            // neighbors. Erase last pixel
                            segmentPoints[segmentNos].pop_back();
                            noSegmentPixels--;
                            index--;
                        }
                        else
                            break;
                    }

                    if (chains[chainNo].len > 1 && noSegmentPixels > 0)
                    {
                        fr = chains[chainNo].pixels[chains[chainNo].len - 2].y;
                        fc = chains[chainNo].pixels[chains[chainNo].len - 2].x;

                        int dr = abs(fr - segmentPoints[segmentNos][noSegmentPixels - 1].y);
                        int dc = abs(fc - segmentPoints[segmentNos][noSegmentPixels - 1].x);

                        if (dr <= 1 && dc <= 1)
                            chains[chainNo].len--;
                    }

                    for (int l = chains[chainNo].len - 1; l >= 0; l--)
                    {
                        segmentPoints[segmentNos].push_back(chains[chainNo].pixels[l]);
                        noSegmentPixels++;
                    }

                    chains[chainNo].len = 0;  // Mark as copied
                }
            }

            totalLen = LongestChain(chains, chains[0].children[0]);
            if (totalLen > 1)
            {
                // Retrieve the chainNos
                int count = RetrieveChainNos(chains, chains[0].children[0], chainNos);

                // Copy these chains in the forward direction. Skip the first pixel of the first chain
                // due to repetition with the last pixel of the previous chain
                int lastChainNo = chainNos[0];
                chains[lastChainNo].pixels++;
                chains[lastChainNo].len--;

                for (int k3 = 0; k3 < count; k3++)
                {
                    int chainNo = chainNos[k3];

                    /* See if we can erase some pixels from the last chain. This is for cleanup */
                    int fr = chains[chainNo].pixels[0].y;
                    int fc = chains[chainNo].pixels[0].x;

                    int index = noSegmentPixels - 2;
                    while (index >= 0)
                    {
                        int dr = abs(fr - segmentPoints[segmentNos][index].y);
                        int dc = abs(fc - segmentPoints[segmentNos][index].x);

                        if (dr <= 1 && dc <= 1)
                        {
                            // neighbors. Erase last pixel
                            segmentPoints[segmentNos].pop_back();
                            noSegmentPixels--;
                            index--;
                        }
                        else
                            break;
                    }

                    int startIndex = 0;
                    int chainLen = chains[chainNo].len;
                    if (chainLen > 1 && noSegmentPixels > 0)
                    {
                        fr = chains[chainNo].pixels[1].y;
                        fc = chains[chainNo].pixels[1].x;

                        int dr = abs(fr - segmentPoints[segmentNos][noSegmentPixels - 1].y);
                        int dc = abs(fc - segmentPoints[segmentNos][noSegmentPixels - 1].x);

                        if (dr <= 1 && dc <= 1)
                        {
                            startIndex = 1;
                        }
                    }

                    /* Start a new chain & copy pixels from the new chain */
                    for (int l = startIndex; l < chains[chainNo].len; l++)
                    {
                        segmentPoints[segmentNos].push_back(chains[chainNo].pixels[l]);
                        noSegmentPixels++;
                    }

                    chains[chainNo].len = 0;  // Mark as copied
                }
            }

            // See if the first pixel can be cleaned up
            int fr = segmentPoints[segmentNos][1].y;
            int fc = segmentPoints[segmentNos][1].x;

            int dr = abs(fr - segmentPoints[segmentNos][noSegmentPixels - 1].y);
            int dc = abs(fc - segmentPoints[segmentNos][noSegmentPixels - 1].x);

            if (dr <= 1 && dc <= 1)
            {
                segmentPoints[segmentNos].erase(segmentPoints[segmentNos].begin());
                noSegmentPixels--;
            }

            segmentNos++;
            segmentPoints.push_back(vector<Point>()); // create empty vector of points for segments

            // Copy the rest of the long chains here
            for (int k4 = 2; k4 < noChains; k4++)
            {
                if (chains[k4].len < 2)
                    continue;

                totalLen = LongestChain(chains, k4);

                if (totalLen >= 10)
                {
                    // Retrieve the chainNos
                    int count = RetrieveChainNos(chains, k4, chainNos);

                    // Copy the pixels
                    noSegmentPixels = 0;
                    for (int k5 = 0; k5 < count; k5++)
                    {
                        int chainNo = chainNos[k5];

                        /* See if we can erase some pixels from the last chain. This is for cleanup */
                        fr = chains[chainNo].pixels[0].y;
                        fc = chains[chainNo].pixels[0].x;

                        int index = noSegmentPixels - 2;
                        while (index >= 0)
                        {
                            dr = abs(fr - segmentPoints[segmentNos][index].y);
                            dc = abs(fc - segmentPoints[segmentNos][index].x);

                            if (dr <= 1 && dc <= 1)
                            {
                                // neighbors. Erase last pixel
                                segmentPoints[segmentNos].pop_back();
                                noSegmentPixels--;
                                index--;
                            }
                            else
                                break;
                        }

                        int startIndex = 0;
                        int chainLen = chains[chainNo].len;
                        if (chainLen > 1 && noSegmentPixels > 0)
                        {
                            fr = chains[chainNo].pixels[1].y;
                            fc = chains[chainNo].pixels[1].x;

                            dr = abs(fr - segmentPoints[segmentNos][noSegmentPixels - 1].y);
                            dc = abs(fc - segmentPoints[segmentNos][noSegmentPixels - 1].x);

                            if (dr <= 1 && dc <= 1)
                            {
                                startIndex = 1;
                            }
                        }

                        /* Start a new chain & copy pixels from the new chain */
                        for (int l = startIndex; l < chains[chainNo].len; l++)
                        {
                            segmentPoints[segmentNos].push_back(chains[chainNo].pixels[l]);
                            noSegmentPixels++;
                        }

                        chains[chainNo].len = 0;  // Mark as copied
                    }
                    segmentPoints.push_back(vector<Point>()); // create empty vector of points for segments
                    segmentNos++;
                }
            }
        }
    }

    // pop back last segment from vector
    // because of one preallocation in the beginning, it will always empty
    segmentPoints.pop_back();

    // Clean up
    delete[] pAnchors;
    delete[] chains;
    delete[] stack;
    delete[] chainNos;
    delete[] pixels;
}

int* EdgeDrawingImpl::sortAnchorsByGradValue1()
{
    int SIZE = 128 * 256;
    int* C = new int[SIZE];
    memset(C, 0, sizeof(int) * SIZE);

    // Count the number of grad values
    for (int i = 1; i < height - 1; i++)
    {
        for (int j = 1; j < width - 1; j++)
        {
            if (edgeImg[i * width + j] != ANCHOR_PIXEL)
                continue;

            int grad = gradImg[i * width + j];
            C[grad]++;
        }
    }

    // Compute indices
    for (int i = 1; i < SIZE; i++)
        C[i] += C[i - 1];

    int noAnchors = C[SIZE - 1];
    int* A = new int[noAnchors];

    for (int i = 1; i < height - 1; i++)
    {
        for (int j = 1; j < width - 1; j++)
        {
            if (edgeImg[i * width + j] != ANCHOR_PIXEL)
                continue;

            int grad = gradImg[i * width + j];
            int index = --C[grad];
            A[index] = i * width + j;    // anchor's offset
        }
    }

    delete[] C;
    return A;
}

int EdgeDrawingImpl::LongestChain(Chain* chains, int root)
{
    if (root == -1 || chains[root].len == 0)
        return 0;

    int len0 = 0;
    if (chains[root].children[0] != -1)
        len0 = LongestChain(chains, chains[root].children[0]);

    int len1 = 0;
    if (chains[root].children[1] != -1)
        len1 = LongestChain(chains, chains[root].children[1]);

    int max = 0;

    if (len0 >= len1)
    {
        max = len0;
        chains[root].children[1] = -1;
    }
    else
    {
        max = len1;
        chains[root].children[0] = -1;
    }

    return chains[root].len + max;
}

int EdgeDrawingImpl::RetrieveChainNos(Chain* chains, int root, int chainNos[])
{
    int count = 0;

    while (root != -1)
    {
        chainNos[count] = root;
        count++;

        if (chains[root].children[0] != -1)
            root = chains[root].children[0];
        else
            root = chains[root].children[1];
    }

    return count;
}

void EdgeDrawingImpl::detectLines(OutputArray _lines)
{
    std::vector<Vec4f> linePoints;
    if (segmentPoints.size() < 1)
    {
        Mat(linePoints).copyTo(_lines);
        return;
    }

    min_line_len = params.MinLineLength;
    line_error = params.LineFitErrorThreshold;
    max_distance_between_two_lines = params.MaxDistanceBetweenTwoLines;
    max_error = params.MaxErrorThreshold;

    if (min_line_len == -1) // If no initial value given, compute it
        min_line_len = ComputeMinLineLength();

    if (min_line_len < 9) // avoids small line segments in the result. Might be deleted!
        min_line_len = 9;

    // Temporary buffers used during line fitting
    double* x = new double[(width + height) * 8];
    double* y = new double[(width + height) * 8];

    lines.clear();
    linesNo = 0;

    // Use the whole segment
    for (size_t segmentNumber = 0; segmentNumber < segmentPoints.size(); segmentNumber++)
    {
        std::vector<Point> segment = segmentPoints[segmentNumber];
        for (int k = 0; k < (int)segment.size(); k++)
        {
            x[k] = segment[k].x;
            y[k] = segment[k].y;
        }
        SplitSegment2Lines(x, y, (int)segment.size(), (int)segmentNumber);
    }

    JoinCollinearLines();

    if (params.NFAValidation)
        ValidateLineSegments();

    // Delete redundant space from lines
    // Pop them back
    int size = (int)lines.size();
    for (int i = 1; i <= size - linesNo; i++)
        lines.pop_back();

    segmentIndicesOfLines.clear();
    for (int i = 0; i < linesNo; i++)
    {
        Vec4f line((float)lines[i].sx, (float)lines[i].sy, (float)lines[i].ex, (float)lines[i].ey);
        linePoints.push_back(line);
        segmentIndicesOfLines.push_back(lines[i].segmentNo);
    }
    Mat(linePoints).copyTo(_lines);

    delete[] x;
    delete[] y;
}

// Computes the minimum line length using the NFA formula given width & height values
int EdgeDrawingImpl::ComputeMinLineLength()
{
    // The reason we are dividing the theoretical minimum line length by 2 is because
    // we now test short line segments by a line support region rectangle having width=2.
    // This means that within a line support region rectangle for a line segment of length "l"
    // there are "2*l" many pixels. Thus, a line segment of length "l" has a chance of getting
    // validated by NFA.

    double logNT = 2.0 * (log10((double)width) + log10((double)height));
    return (int)round((-logNT / log10(0.125)) * 0.5);
}

//-----------------------------------------------------------------
// Given a full segment of pixels, splits the chain to lines
// This code is used when we use the whole segment of pixels
//
void EdgeDrawingImpl::SplitSegment2Lines(double* x, double* y, int noPixels, int segmentNo)
{
    // First pixel of the line segment within the segment of points
    int firstPixelIndex = 0;

    while (noPixels >= min_line_len)
    {
        // Start by fitting a line to MIN_LINE_LEN pixels
        bool valid = false;
        double lastA(0), lastB(0), error(0);
        int lastInvert(0);

        while (noPixels >= min_line_len)
        {
            LineFit(x, y, min_line_len, lastA, lastB, error, lastInvert);
            if (error <= 0.5)
            {
                valid = true;
                break;
            }

            noPixels -= 1;   // Go slowly
            x += 1;
            y += 1;
            firstPixelIndex += 1;
        }

        if (valid == false)
            return;

        // Now try to extend this line
        int index = min_line_len;
        int len = min_line_len;

        while (index < noPixels)
        {
            int startIndex = index;
            int lastGoodIndex = index - 1;
            int goodPixelCount = 0;
            int badPixelCount = 0;

            while (index < noPixels)
            {
                double d = ComputeMinDistance(x[index], y[index], lastA, lastB, lastInvert);

                if (d <= line_error)
                {
                    lastGoodIndex = index;
                    goodPixelCount++;
                    badPixelCount = 0;
                }
                else
                {
                    badPixelCount++;
                    if (badPixelCount >= 5)
                        break;
                }
                index++;
            }

            if (goodPixelCount >= 2)
            {
                len += lastGoodIndex - startIndex + 1;
                LineFit(x, y, len, lastA, lastB, lastInvert);  // faster LineFit
                index = lastGoodIndex + 1;
            }

            if (goodPixelCount < 2 || index >= noPixels)
            {
                // End of a line segment. Compute the end points
                double sx, sy, ex, ey;

                index = 0;
                while (ComputeMinDistance(x[index], y[index], lastA, lastB, lastInvert) > line_error)
                    index++;
                ComputeClosestPoint(x[index], y[index], lastA, lastB, lastInvert, sx, sy);
                int noSkippedPixels = index;

                index = lastGoodIndex;
                while (ComputeMinDistance(x[index], y[index], lastA, lastB, lastInvert) > line_error)
                    index--;
                ComputeClosestPoint(x[index], y[index], lastA, lastB, lastInvert, ex, ey);

                if ((sx == ex) & (sy == ey))
                    break;

                // Add the line segment to lines
                lines.push_back(EDLineSegment(lastA, lastB, lastInvert, sx, sy, ex, ey, segmentNo, firstPixelIndex + noSkippedPixels, index - noSkippedPixels + 1));
                linesNo++;
                len = index + 1;

                break;
            }
        }

        noPixels -= len;
        x += len;
        y += len;
        firstPixelIndex += len;
    }
}

//------------------------------------------------------------------
// Goes over the original line segments and combines collinear lines that belong to the same segment
//
void EdgeDrawingImpl::JoinCollinearLines()
{
    int lastLineIndex = -1;   //Index of the last line in the joined lines
    int i = 0;
    while (i < linesNo)
    {
        int segmentNo = lines[i].segmentNo;

        lastLineIndex++;
        if (lastLineIndex != i)
            lines[lastLineIndex] = lines[i];

        int firstLineIndex = lastLineIndex;  // Index of the first line in this segment

        int count = 1;
        for (int j = i + 1; j < linesNo; j++)
        {
            if (lines[j].segmentNo != segmentNo)
                break;

            // Try to combine this line with the previous line in this segment
            if (TryToJoinTwoLineSegments(&lines[lastLineIndex], &lines[j],
                                         lastLineIndex) == false)
            {
                lastLineIndex++;
                if (lastLineIndex != j)
                    lines[lastLineIndex] = lines[j];
            }
            count++;
        }

        // Try to join the first & last line of this segment
        if (firstLineIndex != lastLineIndex)
        {
            if (TryToJoinTwoLineSegments(&lines[firstLineIndex], &lines[lastLineIndex],
                                         firstLineIndex))
            {
                lastLineIndex--;
            }
        }
        i += count;
    }
    linesNo = lastLineIndex + 1;
}

void EdgeDrawingImpl::ValidateLineSegments()
{
#define PRECISION_ANGLE 22.5
    precision = (PRECISION_ANGLE / 180) * CV_PI;
#undef PRECISION_ANGLE

    if (nfa->LUTSize == 1)
    {
        int lutSize = (width + height) / 8;
        double prob = 1.0 / 8;  // probability of alignment
        nfa = new NFALUT(lutSize, prob, width, height);
    }

    int* x = new int[(width + height) * 4];
    int* y = new int[(width + height) * 4];

    int noValidLines = 0;

    for (int i = 0; i < linesNo; i++)
    {
        EDLineSegment* ls = &lines[i];

        // Compute Line's angle
        double lineAngle;

        if (ls->invert == 0)
        {
            // y = a + bx
            lineAngle = atan(ls->b);
        }
        else
        {
            // x = a + by
            lineAngle = atan(1.0 / ls->b);
        }

        if (lineAngle < 0)
            lineAngle += CV_PI;

        Point* pixels = &(segmentPoints[ls->segmentNo][0]);
        int noPixels = ls->len;

        bool valid = false;

        // Accept very long lines without testing. They are almost never invalidated.
        if (ls->len >= 80)
        {
            valid = true;
            // Validate short line segments by a line support region rectangle having width=2
        }
        else if (ls->len <= 25)
        {
            valid = ValidateLineSegmentRect(x, y, ls);
        }
        else
        {
            // Longer line segments are first validated by a line support region rectangle having width=1 (for speed)
            // If the line segment is still invalid, then a line support region rectangle having width=2 is tried
            // If the line segment fails both tests, it is discarded
            int aligned = 0;
            int count = 0;
            for (int j = 0; j < noPixels; j++)
            {
                int r = pixels[j].x;
                int c = pixels[j].y;

                if (r <= 0 || r >= height - 1 || c <= 0 || c >= width - 1)
                    continue;

                count++;

                // compute gx & gy using the simple [-1 -1 -1]
                //                                  [ 1  1  1]  filter in both directions
                // Faster method below
                // A B C
                // D x E
                // F G H
                // gx = (C-A) + (E-D) + (H-F)
                // gy = (F-A) + (G-B) + (H-C)
                //
                // To make this faster:
                // com1 = (H-A)
                // com2 = (C-F)
                // Then: gx = com1 + com2 + (E-D) = (H-A) + (C-F) + (E-D) = (C-A) + (E-D) + (H-F)
                //       gy = com2 - com1 + (G-B) = (H-A) - (C-F) + (G-B) = (F-A) + (G-B) + (H-C)
                //
                int com1 = srcImg[(r + 1) * width + c + 1] - srcImg[(r - 1) * width + c - 1];
                int com2 = srcImg[(r - 1) * width + c + 1] - srcImg[(r + 1) * width + c - 1];

                int gx = com1 + com2 + srcImg[r * width + c + 1] - srcImg[r * width + c - 1];
                int gy = com1 - com2 + srcImg[(r + 1) * width + c] - srcImg[(r - 1) * width + c];

                double pixelAngle = nfa->myAtan2((double)gx, (double)-gy);
                double diff = fabs(lineAngle - pixelAngle);

                if (diff <= precision || diff >= CV_PI - precision)
                    aligned++;
            }

            // Check validation by NFA computation (fast due to LUT)
            valid = nfa->checkValidationByNFA(count, aligned) || ValidateLineSegmentRect(x, y, ls);
        }

        if (valid)
        {
            if (i != noValidLines)
                lines[noValidLines] = lines[i];
            noValidLines++;
        }
    }

    linesNo = noValidLines;

    delete[] x;
    delete[] y;
}

bool EdgeDrawingImpl::ValidateLineSegmentRect(int* x, int* y, EDLineSegment* ls)
{
    // Compute Line's angle
    double lineAngle;

    if (ls->invert == 0)
    {
        // y = a + bx
        lineAngle = atan(ls->b);
    }
    else
    {
        // x = a + by
        lineAngle = atan(1.0 / ls->b);
    }

    if (lineAngle < 0)
        lineAngle += CV_PI;

    int noPoints = 0;

    // Enumerate all pixels that fall within the bounding rectangle
    EnumerateRectPoints(ls->sx, ls->sy, ls->ex, ls->ey, x, y, &noPoints);

    int count = 0;
    int aligned = 0;

    for (int i = 0; i < noPoints; i++)
    {
        int r = y[i];
        int c = x[i];

        if (r <= 0 || r >= height - 1 || c <= 0 || c >= width - 1)
            continue;

        count++;

        // compute gx & gy using the simple [-1 -1 -1]
        //                                  [ 1  1  1]  filter in both directions
        // Faster method below
        // A B C
        // D x E
        // F G H
        // gx = (C-A) + (E-D) + (H-F)
        // gy = (F-A) + (G-B) + (H-C)
        //
        // To make this faster:
        // com1 = (H-A)
        // com2 = (C-F)
        // Then: gx = com1 + com2 + (E-D) = (H-A) + (C-F) + (E-D) = (C-A) + (E-D) + (H-F)
        //       gy = com2 - com1 + (G-B) = (H-A) - (C-F) + (G-B) = (F-A) + (G-B) + (H-C)
        //
        int com1 = srcImg[(r + 1) * width + c + 1] - srcImg[(r - 1) * width + c - 1];
        int com2 = srcImg[(r - 1) * width + c + 1] - srcImg[(r + 1) * width + c - 1];

        int gx = com1 + com2 + srcImg[r * width + c + 1] - srcImg[r * width + c - 1];
        int gy = com1 - com2 + srcImg[(r + 1) * width + c] - srcImg[(r - 1) * width + c];
        double pixelAngle = nfa->myAtan2((double)gx, (double)-gy);

        double diff = fabs(lineAngle - pixelAngle);

        if (diff <= precision || diff >= CV_PI - precision)
            aligned++;
    }
    return nfa->checkValidationByNFA(count, aligned);
}

double EdgeDrawingImpl::ComputeMinDistance(double x1, double y1, double a, double b, int invert)
{
    double x2, y2;

    if (invert == 0)
    {
        if (b == 0)
        {
            x2 = x1;
            y2 = a;
        }
        else
        {
            // Let the line passing through (x1, y1) that is perpendicular to a+bx be c+dx
            double d = -1.0 / (b);
            double c = y1 - d * x1;

            x2 = (a - c) / (d - b);
            y2 = a + b * x2;
        }
    }
    else
    {
        /// invert = 1
        if (b == 0)
        {
            x2 = a;
            y2 = y1;
        }
        else
        {
            // Let the line passing through (x1, y1) that is perpendicular to a+by be c+dy
            double d = -1.0 / (b);
            double c = x1 - d * y1;

            y2 = (a - c) / (d - b);
            x2 = a + b * y2;
        }
    }

    return sqrt((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2));
}

//---------------------------------------------------------------------------------
// Given a point (x1, y1) and a line equation y=a+bx (invert=0) OR x=a+by (invert=1)
// Computes the (x2, y2) on the line that is closest to (x1, y1)
//
void EdgeDrawingImpl::ComputeClosestPoint(double x1, double y1, double a, double b, int invert, double& xOut, double& yOut)
{
    double x2, y2;

    if (invert == 0)
    {
        if (b == 0)
        {
            x2 = x1;
            y2 = a;
        }
        else
        {
            // Let the line passing through (x1, y1) that is perpendicular to a+bx be c+dx
            double d = -1.0 / (b);
            double c = y1 - d * x1;

            x2 = (a - c) / (d - b);
            y2 = a + b * x2;
        }
    }
    else
    {
        /// invert = 1
        if (b == 0)
        {
            x2 = a;
            y2 = y1;
        }
        else
        {
            // Let the line passing through (x1, y1) that is perpendicular to a+by be c+dy
            double d = -1.0 / (b);
            double c = x1 - d * y1;

            y2 = (a - c) / (d - b);
            x2 = a + b * y2;
        }
    }

    xOut = x2;
    yOut = y2;
}

//-----------------------------------------------------------------------------------
// Fits a line of the form y=a+bx (invert == 0) OR x=a+by (invert == 1)
// Assumes that the direction of the line is known by a previous computation
//
void EdgeDrawingImpl::LineFit(double* x, double* y, int count, double& a, double& b, int invert)
{
    if (count < 2)
        return;

    double S = count, Sx = 0.0, Sy = 0.0, Sxx = 0.0, Sxy = 0.0;
    for (int i = 0; i < count; i++)
    {
        Sx += x[i];
        Sy += y[i];
    }

    if (invert)
    {
        // Vertical line. Swap x & y, Sx & Sy
        double* t = x;
        x = y;
        y = t;

        double d = Sx;
        Sx = Sy;
        Sy = d;
    }

    // Now compute Sxx & Sxy
    for (int i = 0; i < count; i++)
    {
        Sxx += x[i] * x[i];
        Sxy += x[i] * y[i];
    }

    double D = S * Sxx - Sx * Sx;
    a = (Sxx * Sy - Sx * Sxy) / D;
    b = (S * Sxy - Sx * Sy) / D;
}

//-----------------------------------------------------------------------------------
// Fits a line of the form y=a+bx (invert == 0) OR x=a+by (invert == 1)
//
void EdgeDrawingImpl::LineFit(double* x, double* y, int count, double& a, double& b, double& e, int& invert)
{
    if (count < 2)
        return;

    double S = count, Sx = 0.0, Sy = 0.0, Sxx = 0.0, Sxy = 0.0;
    for (int i = 0; i < count; i++)
    {
        Sx += x[i];
        Sy += y[i];
    }

    double mx = Sx / count;
    double my = Sy / count;

    double dx = 0.0;
    double dy = 0.0;

    for (int i = 0; i < count; i++)
    {
        dx += (x[i] - mx) * (x[i] - mx);
        dy += (y[i] - my) * (y[i] - my);
    }

    if (dx < dy)
    {
        // Vertical line. Swap x & y, Sx & Sy
        invert = 1;
        double* t = x;
        x = y;
        y = t;

        double d = Sx;
        Sx = Sy;
        Sy = d;
    }
    else
    {
        invert = 0;
    }

    // Now compute Sxx & Sxy
    for (int i = 0; i < count; i++)
    {
        Sxx += x[i] * x[i];
        Sxy += x[i] * y[i];
    }

    double D = S * Sxx - Sx * Sx;
    a = (Sxx * Sy - Sx * Sxy) / D;
    b = (S * Sxy - Sx * Sy) / D;

    if (b == 0.0)
    {
        // Vertical or horizontal line
        double error = 0.0;
        for (int i = 0; i < count; i++)
        {
            error += fabs((a)-y[i]);
        }
        e = error / count;
    }
    else
    {
        double error = 0.0;
        for (int i = 0; i < count; i++)
        {
            // Let the line passing through (x[i], y[i]) that is perpendicular to a+bx be c+dx
            double d = -1.0 / (b);
            double c = y[i] - d * x[i];
            double x2 = ((a)-c) / (d - (b));
            double y2 = (a)+(b)*x2;

            double dist = (x[i] - x2) * (x[i] - x2) + (y[i] - y2) * (y[i] - y2);
            error += dist;
        }
        e = sqrt(error / count);
    }
}

//-----------------------------------------------------------------
// Checks if the given line segments are collinear & joins them if they are
// In case of a join, ls1 is updated. ls2 is NOT changed
// Returns true if join is successful, false otherwise
//
bool EdgeDrawingImpl::TryToJoinTwoLineSegments(EDLineSegment* ls1, EDLineSegment* ls2, int changeIndex)
{
    int which;
    double dist = ComputeMinDistanceBetweenTwoLines(ls1, ls2, &which);
    if (dist > max_distance_between_two_lines)
        return false;

    // Compute line lengths. Use the longer one as the ground truth
    double dx = ls1->sx - ls1->ex;
    double dy = ls1->sy - ls1->ey;
    double prevLen = sqrt(dx * dx + dy * dy);

    dx = ls2->sx - ls2->ex;
    dy = ls2->sy - ls2->ey;
    double nextLen = sqrt(dx * dx + dy * dy);

    // Use the longer line as the ground truth
    EDLineSegment* shorter = ls1;
    EDLineSegment* longer = ls2;

    if (prevLen > nextLen)
    {
        shorter = ls2;
        longer = ls1;
    }

    // Just use 3 points to check for collinearity
    dist = ComputeMinDistance(shorter->sx, shorter->sy, longer->a, longer->b, longer->invert);
    dist += ComputeMinDistance((shorter->sx + shorter->ex) / 2.0, (shorter->sy + shorter->ey) / 2.0, longer->a, longer->b, longer->invert);
    dist += ComputeMinDistance(shorter->ex, shorter->ey, longer->a, longer->b, longer->invert);

    dist /= 3.0;

    if (dist > max_error)
        return false;

    /// 4 cases: 1:(s1, s2), 2:(s1, e2), 3:(e1, s2), 4:(e1, e2)

    /// case 1: (s1, s2)
    dx = fabs(ls1->sx - ls2->sx);
    dy = fabs(ls1->sy - ls2->sy);
    double d = dx + dy;
    double max = d;
    which = 1;

    /// case 2: (s1, e2)
    dx = fabs(ls1->sx - ls2->ex);
    dy = fabs(ls1->sy - ls2->ey);
    d = dx + dy;
    if (d > max)
    {
        max = d;
        which = 2;
    }

    /// case 3: (e1, s2)
    dx = fabs(ls1->ex - ls2->sx);
    dy = fabs(ls1->ey - ls2->sy);
    d = dx + dy;
    if (d > max)
    {
        max = d;
        which = 3;
    }

    /// case 4: (e1, e2)
    dx = fabs(ls1->ex - ls2->ex);
    dy = fabs(ls1->ey - ls2->ey);
    d = dx + dy;
    if (d > max)
    {
        max = d;
        which = 4;
    }

    if (which == 1)
    {
        // (s1, s2)
        ls1->ex = ls2->sx;
        ls1->ey = ls2->sy;
    }
    else if (which == 2)
    {
        // (s1, e2)
        ls1->ex = ls2->ex;
        ls1->ey = ls2->ey;
    }
    else if (which == 3)
    {
        // (e1, s2)
        ls1->sx = ls2->sx;
        ls1->sy = ls2->sy;
    }
    else
    {
        // (e1, e2)
        ls1->sx = ls1->ex;
        ls1->sy = ls1->ey;

        ls1->ex = ls2->ex;
        ls1->ey = ls2->ey;
    }

    // Update the first line's parameters
    if (ls1->firstPixelIndex + ls1->len + 5 >= ls2->firstPixelIndex)
        ls1->len += ls2->len;
    else if (ls2->len > ls1->len)
    {
        ls1->firstPixelIndex = ls2->firstPixelIndex;
        ls1->len = ls2->len;
    }

    UpdateLineParameters(ls1);
    lines[changeIndex] = *ls1;

    return true;
}

//-------------------------------------------------------------------------------
// Computes the minimum distance between the end points of two lines
//
double EdgeDrawingImpl::ComputeMinDistanceBetweenTwoLines(EDLineSegment* ls1, EDLineSegment* ls2, int* pwhich)
{
    double dx = ls1->sx - ls2->sx;
    double dy = ls1->sy - ls2->sy;
    double d = sqrt(dx * dx + dy * dy);
    double min = d;
    int which = SOUTH_SOUTH;

    dx = ls1->sx - ls2->ex;
    dy = ls1->sy - ls2->ey;
    d = sqrt(dx * dx + dy * dy);
    if (d < min)
    {
        min = d;
        which = SOUTH_EAST;
    }

    dx = ls1->ex - ls2->sx;
    dy = ls1->ey - ls2->sy;
    d = sqrt(dx * dx + dy * dy);
    if (d < min)
    {
        min = d;
        which = EAST_SOUTH;
    }

    dx = ls1->ex - ls2->ex;
    dy = ls1->ey - ls2->ey;
    d = sqrt(dx * dx + dy * dy);
    if (d < min)
    {
        min = d;
        which = EAST_EAST;
    }

    if (pwhich)
        *pwhich = which;
    return min;
}

//-----------------------------------------------------------------------------------
// Uses the two end points (sx, sy)----(ex, ey) of the line segment
// and computes the line that passes through these points (a, b, invert)
//
void EdgeDrawingImpl::UpdateLineParameters(EDLineSegment* ls)
{
    double dx = ls->ex - ls->sx;
    double dy = ls->ey - ls->sy;

    if (fabs(dx) >= fabs(dy))
    {
        /// Line will be of the form y = a + bx
        ls->invert = 0;
        if (fabs(dy) < 1e-3)
        {
            ls->b = 0;
            ls->a = (ls->sy + ls->ey) / 2;
        }
        else
        {
            ls->b = dy / dx;
            ls->a = ls->sy - (ls->b) * ls->sx;
        }
    }
    else
    {
        /// Line will be of the form x = a + by
        ls->invert = 1;
        if (fabs(dx) < 1e-3)
        {
            ls->b = 0;
            ls->a = (ls->sx + ls->ex) / 2;
        }
        else
        {
            ls->b = dx / dy;
            ls->a = ls->sx - (ls->b) * ls->sy;
        }
    }
}

void EdgeDrawingImpl::EnumerateRectPoints(double sx, double sy, double ex, double ey, int ptsx[], int ptsy[], int* pNoPoints)
{
    double vxTmp[4], vyTmp[4];
    double vx[4], vy[4];
    int n, offset;

    double x1 = sx;
    double y1 = sy;
    double x2 = ex;
    double y2 = ey;
    double width = 2;

    double dx = x2 - x1;
    double dy = y2 - y1;
    double vLen = sqrt(dx * dx + dy * dy);

    // make unit vector
    dx = dx / vLen;
    dy = dy / vLen;

    /* build list of rectangle corners ordered
    in a circular way around the rectangle */
    vxTmp[0] = x1 - dy * width / 2.0;
    vyTmp[0] = y1 + dx * width / 2.0;
    vxTmp[1] = x2 - dy * width / 2.0;
    vyTmp[1] = y2 + dx * width / 2.0;
    vxTmp[2] = x2 + dy * width / 2.0;
    vyTmp[2] = y2 - dx * width / 2.0;
    vxTmp[3] = x1 + dy * width / 2.0;
    vyTmp[3] = y1 - dx * width / 2.0;

    /* compute rotation of index of corners needed so that the first
    point has the smaller x.

    if one side is vertical, thus two corners have the same smaller x
    value, the one with the largest y value is selected as the first.
    */
    if (x1 < x2 && y1 <= y2)
        offset = 0;
    else if (x1 >= x2 && y1 < y2)
        offset = 1;
    else if (x1 > x2&& y1 >= y2)
        offset = 2;
    else
        offset = 3;

    /* apply rotation of index. */
    for (n = 0; n < 4; n++)
    {
        vx[n] = vxTmp[(offset + n) % 4];
        vy[n] = vyTmp[(offset + n) % 4];
    }

    /* Set a initial condition.

    The values are set to values that will cause 'ri_inc' (that will
    be called immediately) to initialize correctly the first 'column'
    and compute the limits 'ys' and 'ye'.

    'y' is set to the integer value of vy[0], the starting corner.

    'ys' and 'ye' are set to very small values, so 'ri_inc' will
    notice that it needs to start a new 'column'.

    The smaller integer coordinate inside of the rectangle is
    'ceil(vx[0])'. The current 'x' value is set to that value minus
    one, so 'ri_inc' (that will increase x by one) will advance to
    the first 'column'.
    */
    int x = (int)ceil(vx[0]) - 1;
    int y = (int)ceil(vy[0]);
    double ys = -DBL_MAX, ye = -DBL_MAX;

    int noPoints = 0;
    while (1)
    {
        /* if not at end of exploration,
        increase y value for next pixel in the 'column' */
        y++;

        /* if the end of the current 'column' is reached,
        and it is not the end of exploration,
        advance to the next 'column' */
        while (y > ye&& x <= vx[2])
        {
            /* increase x, next 'column' */
            x++;

            /* if end of exploration, return */
            if (x > vx[2])
                break;

            /* update lower y limit (start) for the new 'column'.

            We need to interpolate the y value that corresponds to the
            lower side of the rectangle. The first thing is to decide if
            the corresponding side is

            vx[0],vy[0] to vx[3],vy[3] or
            vx[3],vy[3] to vx[2],vy[2]

            Then, the side is interpolated for the x value of the
            'column'. But, if the side is vertical (as it could happen if
            the rectangle is vertical and we are dealing with the first
            or last 'columns') then we pick the lower value of the side
            by using 'inter_low'.
            */
            if ((double)x < vx[3])
            {
                /* interpolation */
                if (fabs(vx[0] - vx[3]) <= 0.01)
                {
                    if (vy[0] < vy[3])
                        ys = vy[0];
                    else if (vy[0] > vy[3])
                        ys = vy[3];
                    else
                        ys = vy[0] + (x - vx[0]) * (vy[3] - vy[0]) / (vx[3] - vx[0]);
                }
                else
                    ys = vy[0] + (x - vx[0]) * (vy[3] - vy[0]) / (vx[3] - vx[0]);
            }
            else
            {
                /* interpolation */
                if (fabs(vx[3] - vx[2]) <= 0.01)
                {
                    if (vy[3] < vy[2])
                        ys = vy[3];
                    else if (vy[3] > vy[2])
                        ys = vy[2];
                    else
                        ys = vy[3] + (x - vx[3]) * (y2 - vy[3]) / (vx[2] - vx[3]);
                }
                else
                    ys = vy[3] + (x - vx[3]) * (vy[2] - vy[3]) / (vx[2] - vx[3]);
            }

            /* update upper y limit (end) for the new 'column'.

            We need to interpolate the y value that corresponds to the
            upper side of the rectangle. The first thing is to decide if
            the corresponding side is

            vx[0],vy[0] to vx[1],vy[1] or
            vx[1],vy[1] to vx[2],vy[2]

            Then, the side is interpolated for the x value of the
            'column'. But, if the side is vertical (as it could happen if
            the rectangle is vertical and we are dealing with the first
            or last 'columns') then we pick the lower value of the side
            by using 'inter_low'.
            */
            if ((double)x < vx[1])
            {
                /* interpolation */
                if (fabs(vx[0] - vx[1]) <= 0.01)
                {
                    if (vy[0] < vy[1])
                        ye = vy[1];
                    else if (vy[0] > vy[1])
                        ye = vy[0];
                    else
                        ye = vy[0] + (x - vx[0]) * (vy[1] - vy[0]) / (vx[1] - vx[0]);
                }
                else
                    ye = vy[0] + (x - vx[0]) * (vy[1] - vy[0]) / (vx[1] - vx[0]);
            }
            else
            {
                /* interpolation */
                if (fabs(vx[1] - vx[2]) <= 0.01)
                {
                    if (vy[1] < vy[2])
                        ye = vy[2];
                    else if (vy[1] > vy[2])
                        ye = vy[1];
                    else
                        ye = vy[1] + (x - vx[1]) * (vy[2] - vy[1]) / (vx[2] - vx[1]);
                }
                else
                    ye = vy[1] + (x - vx[1]) * (vy[2] - vy[1]) / (vx[2] - vx[1]);
            }
            /* new y */
            y = (int)ceil(ys);
        }

        // Are we done?
        if (x > vx[2])
            break;

        ptsx[noPoints] = x;
        ptsy[noPoints] = y;
        noPoints++;
    }
    *pNoPoints = noPoints;
}

/*--------------------------------------EDPF----------------------------------------*/

//----------------------------------------------------------------------------------
// Resursive validation using half of the pixels as suggested by DMM algorithm
// We take pixels at Nyquist distance, i.e., 2 (as suggested by DMM)
//
void EdgeDrawingImpl::TestSegment(int i, int index1, int index2)
{
    int chainLen = index2 - index1 + 1;
    if (chainLen < params.MinPathLength)
        return;

    // Test from index1 to index2. If OK, then we are done. Otherwise, split into two and
    // recursively test the left & right halves

    // First find the min. gradient along the segment
    int minGrad = 1 << 30;
    int minGradIndex = 0;

    for (int k = index1; k <= index2; k++) {
        int r = segmentPoints[i][k].y;
        int c = segmentPoints[i][k].x;
        if (gradImg[r * width + c] < minGrad) { minGrad = gradImg[r * width + c]; minGradIndex = k; }
    }

    double nfa0 = NFA(dH[minGrad], (int)(chainLen / divForTestSegment));

    if (nfa0 <= 1.0) {
        for (int k = index1; k <= index2; k++) {
            int r = segmentPoints[i][k].y;
            int c = segmentPoints[i][k].x;

            edgeImg[r * width + c] = 255;
        }
        return;
    }

    // Split into two halves. We divide at the point where the gradient is the minimum
    int end = minGradIndex - 1;
    while (end > index1) {
        int r = segmentPoints[i][end].y;
        int c = segmentPoints[i][end].x;

        if (gradImg[r * width + c] <= minGrad) end--;
        else break;
    }

    int start = minGradIndex + 1;
    while (start < index2) {
        int r = segmentPoints[i][start].y;
        int c = segmentPoints[i][start].x;

        if (gradImg[r * width + c] <= minGrad) start++;
        else break;
    }

    TestSegment(i, index1, end);
    TestSegment(i, start, index2);
}

//----------------------------------------------------------------------------------------------
// After the validation of the edge segments, extracts the valid ones
// In other words, updates the valid segments' pixel arrays and their lengths
void EdgeDrawingImpl::ExtractNewSegments()
{
    vector< vector<Point> > validSegments;
    int noSegments = 0;

    for (int i = 0; i < segmentNos; i++) {
        int start = 0;
        while (start < (int)segmentPoints[i].size()) {

            while (start < (int)segmentPoints[i].size()) {
                int r = segmentPoints[i][start].y;
                int c = segmentPoints[i][start].x;

                if (edgeImg[r * width + c]) break;
                start++;
            }

            int end = start + 1;
            while (end < (int)segmentPoints[i].size()) {
                int r = segmentPoints[i][end].y;
                int c = segmentPoints[i][end].x;

                if (edgeImg[r * width + c] == 0) break;
                end++;
            }

            int len = end - start;
            if (len >= 10) {
                // A new segment. Accepted only only long enough (whatever that means)
                //segments[noSegments].pixels = &map->segments[i].pixels[start];
                //segments[noSegments].noPixels = len;
                validSegments.push_back(vector<Point>());
                vector<Point> subVec(&segmentPoints[i][start], &segmentPoints[i][end - 1]);
                validSegments[noSegments] = subVec;
                noSegments++;
            }
            start = end + 1;
        }
    }

    segmentPoints = validSegments;
    segmentNos = noSegments;
}

double EdgeDrawingImpl::NFA(double prob, int len)
{
    double nfa0 = np;
    for (int i = 0; i<len && nfa0 > 1.0; i++)
        nfa0 *= prob;

    return nfa0;
}

/*---------------------------------EDCircle--------------------------------------------*/

void EdgeDrawingImpl::detectEllipses(OutputArray ellipses)
{
    vector<Vec6d> _ellipses;
    if (segmentPoints.size() < 1)
    {
        Mat(_ellipses).copyTo(ellipses);
        return;
    }

    min_line_len = 6;
    line_error = params.LineFitErrorThreshold;
    Circles.clear();
    Ellipses.clear();
    lines.clear();
    // Arcs & circles to be detected
    // If the end-points of the segment is very close to each other,
    // then directly fit a circle/ellipse instread of line fitting
    noCircles1 = 0;
    circles1 = new Circle[(width + height) * 8];

    int bufferSize = 0;
    for (int i = 0; i < (int)segmentPoints.size(); i++)
        bufferSize += (int)segmentPoints[i].size();

    // Compute the starting line number for each segment
    segmentStartLines = new int[segmentNos + 1];

    bm = new BufferManager(bufferSize * 8);

#define CIRCLE_MIN_LINE_LEN 6

    for (int i = 0; i < segmentNos; i++)
    {
        // Make note of the starting line number for this segment
        segmentStartLines[i] = (int)lines.size();

        int noPixels = (int)segmentPoints[i].size();

        if (noPixels < 2 * CIRCLE_MIN_LINE_LEN)
            continue;

        double* x = bm->getX();
        double* y = bm->getY();

        for (int j = 0; j < noPixels; j++)
        {
            x[j] = segmentPoints[i][j].x;
            y[j] = segmentPoints[i][j].y;
        }

        // If the segment is reasonably long, then see if the segment traverses the boundary of a closed shape
        if (noPixels >= 4 * CIRCLE_MIN_LINE_LEN)
        {
            // If the end-points of the segment is close to each other, then assume a circular/elliptic structure
            double dx = x[0] - x[noPixels - 1];
            double dy = y[0] - y[noPixels - 1];
            double d = sqrt(dx * dx + dy * dy);
            double r = noPixels / CV_2PI;      // Assume a complete circle

            double maxDistanceBetweenEndPoints = std::max(3.0, r / 4.0);

            // If almost closed loop, then try to fit a circle/ellipse
            if (d <= maxDistanceBetweenEndPoints)
            {
                double xc, yc, circleFitError = 1e10;

                CircleFit(x, y, noPixels, &xc, &yc, &r, &circleFitError);

                EllipseEquation eq;
                double ellipseFitError = 1e10;

                if (circleFitError > LONG_ARC_ERROR)
                {
                    // Try fitting an ellipse
                    if (EllipseFit(x, y, noPixels, &eq))
                        ellipseFitError = ComputeEllipseError(&eq, x, y, noPixels);
                }

                if (circleFitError <= LONG_ARC_ERROR)
                {
                    addCircle(circles1, noCircles1, xc, yc, r, circleFitError, x, y, noPixels);
                    bm->move(noPixels);
                    continue;
                }
                else if (ellipseFitError <= ELLIPSE_ERROR)
                {
                    double major, minor;
                    ComputeEllipseCenterAndAxisLengths(&eq, &xc, &yc, &major, &minor);

                    // Assume major is longer. Otherwise, swap
                    if (minor > major)
                    {
                        double tmp = major;
                        major = minor;
                        minor = tmp;
                    }

                    if (major < 8 * minor)
                    {
                        addCircle(circles1, noCircles1, xc, yc, r, circleFitError, &eq, ellipseFitError, x, y, noPixels);
                        bm->move(noPixels);
                    }
                    continue;
                }
            }
        }
        // Otherwise, split to lines
        SplitSegment2Lines(x, y, noPixels, i);
    }

    min_line_len = params.MinLineLength;
    segmentStartLines[segmentNos] = (int)lines.size();

    // ------------------------------- DETECT ARCS ---------------------------------

    info = new Info[lines.size()];

    // Compute the angle information for each line segment
    for (int i = 0; i < segmentNos; i++)
    {
        for (int j = segmentStartLines[i]; j < segmentStartLines[i + 1]; j++)
        {
            EDLineSegment* l1 = &lines[j];
            EDLineSegment* l2;

            if (j == segmentStartLines[i + 1] - 1)
                l2 = &lines[segmentStartLines[i]];
            else
                l2 = &lines[j + 1];

            // If the end points of the lines are far from each other, then stop at this line
            double dx = l1->ex - l2->sx;
            double dy = l1->ey - l2->sy;
            double d = sqrt(dx * dx + dy * dy);
            if (d >= 15)
            {
                info[j].angle = 10;
                info[j].sign = 2;
                info[j].taken = false;
                continue;
            }

            // Compute the angle between the lines & their turn direction
            double v1x = l1->ex - l1->sx;
            double v1y = l1->ey - l1->sy;
            double v1Len = sqrt(v1x * v1x + v1y * v1y);

            double v2x = l2->ex - l2->sx;
            double v2y = l2->ey - l2->sy;
            double v2Len = sqrt(v2x * v2x + v2y * v2y);

            double dotProduct = (v1x * v2x + v1y * v2y) / (v1Len * v2Len);
            if (dotProduct > 1.0)
                dotProduct = 1.0;
            else if (dotProduct < -1.0)
                dotProduct = -1.0;

            info[j].angle = acos(dotProduct);
            info[j].sign = (v1x * v2y - v2x * v1y) >= 0 ? 1 : -1;  // compute cross product
            info[j].taken = false;
        }
    }

    // This is how much space we will allocate for circles buffers
    int maxNoOfCircles = (int)lines.size() / 3 + noCircles1 * 2 + 2;

    edarcs1 = new EDArcs(maxNoOfCircles);
    DetectArcs();    // Detect all arcs

    // Try to join arcs that are almost perfectly circular.
    // Use the distance between the arc end-points as a metric in choosing in choosing arcs to join
    edarcs2 = new EDArcs(maxNoOfCircles);
    JoinArcs1();

    // Try to join arcs that belong to the same segment
    edarcs3 = new EDArcs(maxNoOfCircles);
    JoinArcs2();

    // Try to combine arcs that belong to different segments
    edarcs4 = new EDArcs(maxNoOfCircles);     // The remaining arcs
    JoinArcs3();

    // Finally, go over the arcs & circles, and generate candidate circles
    GenerateCandidateCircles();

    //----------------------------- VALIDATE CIRCLES --------------------------
    noCircles2 = 0;
    circles2 = new Circle[maxNoOfCircles];
    GaussianBlur(srcImage, smoothImage, Size(), 0.50); // calculate kernel from sigma;

    ValidateCircles(params.NFAValidation);

    //----------------------------- JOIN CIRCLES --------------------------
    noCircles3 = 0;
    circles3 = new Circle[maxNoOfCircles];
    JoinCircles();

    noCircles = 0;
    noEllipses = 0;
    for (int i = 0; i < noCircles3; i++)
    {
        if (circles3[i].isEllipse)
            noEllipses++;
        else
            noCircles++;
    }

    for (int i = 0; i < noCircles3; i++)
    {
        if (circles3[i].isEllipse)
        {
            EllipseEquation eq = circles3[i].eq;
            double xc;
            double yc;
            double a;
            double b;
            double theta = ComputeEllipseCenterAndAxisLengths(&eq, &xc, &yc, &a, &b);
            Ellipses.push_back(mEllipse(Point2d(xc, yc), Size2d(a, b), theta));
            _ellipses.push_back(Vec6d(xc, yc, 0, a, b, theta * 180 / CV_PI));
        }
        else
        {
            double r = circles3[i].r;
            double xc = circles3[i].xc;
            double yc = circles3[i].yc;

            Circles.push_back(mCircle(Point2d(xc, yc), r));
            _ellipses.push_back(Vec6d(xc, yc, r, 0, 0, 0));
        }
    }

    Mat(_ellipses).copyTo(ellipses);

    delete edarcs1;
    delete edarcs2;
    delete edarcs3;
    delete edarcs4;

    delete[] circles1;
    delete[] circles2;
    delete[] circles3;

    delete bm;
    delete[] segmentStartLines;
    delete[] info;
}

void EdgeDrawingImpl::GenerateCandidateCircles()
{
    // Now, go over the circular arcs & add them to circles1
    MyArc* arcs = edarcs4->arcs;
    for (int i = 0; i < edarcs4->noArcs; i++)
    {
        if (arcs[i].isEllipse)
        {
            // Ellipse
            if (arcs[i].coverRatio >= CANDIDATE_ELLIPSE_RATIO && arcs[i].ellipseFitError <= ELLIPSE_ERROR)
            {
                addCircle(circles1, noCircles1, arcs[i].xc, arcs[i].yc, arcs[i].r, arcs[i].circleFitError, &arcs[i].eq, arcs[i].ellipseFitError,
                    arcs[i].x, arcs[i].y, arcs[i].noPixels);
            }
            else
            {
                double coverRatio = MAX(ArcLength(arcs[i].sTheta, arcs[i].eTheta) / CV_2PI, arcs[i].coverRatio);
                if ((coverRatio >= FULL_CIRCLE_RATIO && arcs[i].circleFitError <= LONG_ARC_ERROR) ||
                    (coverRatio >= HALF_CIRCLE_RATIO && arcs[i].circleFitError <= HALF_ARC_ERROR) ||
                    (coverRatio >= CANDIDATE_CIRCLE_RATIO2 && arcs[i].circleFitError <= SHORT_ARC_ERROR))
                {
                    addCircle(circles1, noCircles1, arcs[i].xc, arcs[i].yc, arcs[i].r, arcs[i].circleFitError, arcs[i].x, arcs[i].y, arcs[i].noPixels);
                }
            }
        }
        else
        {
            // If a very short arc, ignore
            if (arcs[i].coverRatio < CANDIDATE_CIRCLE_RATIO1)
                continue;

            // If the arc is long enough and the circleFitError is small enough, assume a circle
            if ((arcs[i].coverRatio >= FULL_CIRCLE_RATIO && arcs[i].circleFitError <= LONG_ARC_ERROR) ||
                (arcs[i].coverRatio >= HALF_CIRCLE_RATIO && arcs[i].circleFitError <= HALF_ARC_ERROR) ||
                (arcs[i].coverRatio >= CANDIDATE_CIRCLE_RATIO2 && arcs[i].circleFitError <= SHORT_ARC_ERROR))
            {

                addCircle(circles1, noCircles1, arcs[i].xc, arcs[i].yc, arcs[i].r, arcs[i].circleFitError, arcs[i].x, arcs[i].y, arcs[i].noPixels);

                continue;
            }

            if (arcs[i].coverRatio < CANDIDATE_CIRCLE_RATIO2)
                continue;

            // Circle is not possible. Try an ellipse
            EllipseEquation eq;
            double ellipseFitError = 1e10;
            double coverRatio(0);

            int noPixels = arcs[i].noPixels;
            if (EllipseFit(arcs[i].x, arcs[i].y, noPixels, &eq))
            {
                ellipseFitError = ComputeEllipseError(&eq, arcs[i].x, arcs[i].y, noPixels);
                coverRatio = noPixels / computeEllipsePerimeter(&eq);
            }

            if (arcs[i].coverRatio > coverRatio)
                coverRatio = arcs[i].coverRatio;

            if (coverRatio >= CANDIDATE_ELLIPSE_RATIO && ellipseFitError <= ELLIPSE_ERROR)
            {
                addCircle(circles1, noCircles1, arcs[i].xc, arcs[i].yc, arcs[i].r, arcs[i].circleFitError, &eq, ellipseFitError, arcs[i].x, arcs[i].y, arcs[i].noPixels);
            }
        }
    }
}

void EdgeDrawingImpl::DetectArcs()
{
    double maxLineLengthThreshold = MAX(width, height) / 5;

    double MIN_ANGLE = CV_PI / 30;  // 6 degrees
    double MAX_ANGLE = CV_PI / 3;   // 60 degrees

    for (int iter = 1; iter <= 2; iter++)
    {
        if (iter == 2)
            MAX_ANGLE = CV_PI / 1.9;  // 95 degrees

        for (int curSegmentNo = 0; curSegmentNo < segmentNos; curSegmentNo++)
        {
            int firstLine = segmentStartLines[curSegmentNo];
            int stopLine = segmentStartLines[curSegmentNo + 1];

            // We need at least 2 line segments
            if (stopLine - firstLine <= 1)
                continue;

            // Process the info for the lines of this segment
            while (firstLine < stopLine - 1)
            {
                // If the line is already taken during the previous step, continue
                if (info[firstLine].taken)
                {
                    firstLine++;
                    continue;
                }

                // very long lines cannot be part of an arc
                if (lines[firstLine].len >= maxLineLengthThreshold)
                {
                    firstLine++;
                    continue;
                }

                // Skip lines that cannot be part of an arc
                if (info[firstLine].angle < MIN_ANGLE || info[firstLine].angle > MAX_ANGLE)
                {
                    firstLine++;
                    continue;
                }

                // Find a group of lines (at least 3) with the same sign & angle < MAX_ANGLE degrees
                int lastLine = firstLine + 1;
                while (lastLine < stopLine - 1)
                {
                    if (info[lastLine].taken)
                        break;
                    if (info[lastLine].sign != info[firstLine].sign)
                        break;

                    if (lines[lastLine].len >= maxLineLengthThreshold)
                        break;                                // very long lines cannot be part of an arc
                    if (info[lastLine].angle < MIN_ANGLE)
                        break;
                    if (info[lastLine].angle > MAX_ANGLE)
                        break;

                    lastLine++;
                }

                bool specialCase = false;
                int wrapCase = -1;  // 1: wrap the first two lines with the last line, 2: wrap the last two lines with the first line

                if (lastLine - firstLine == 1)
                {
                    // Just 2 lines. If long enough, then try to combine. Angle between 15 & 45 degrees. Min. length = 40
                    int totalLineLength = lines[firstLine].len + lines[firstLine + 1].len;
                    int shorterLen = lines[firstLine].len;
                    int longerLen = lines[firstLine + 1].len;

                    if (lines[firstLine + 1].len < shorterLen)
                    {
                        shorterLen = lines[firstLine + 1].len;
                        longerLen = lines[firstLine].len;
                    }

                    if (info[firstLine].angle >= CV_PI / 12 && info[firstLine].angle <= CV_PI / 4 && totalLineLength >= 40 && shorterLen * 2 >= longerLen)
                    {
                        specialCase = true;
                    }

                    // If the two lines do not make up for arc generation, then try to wrap the lines to the first OR last line.
                    // There are two wrapper cases:
                    if (specialCase == false)
                    {
                        // Case 1: Combine the first two lines with the last line of the segment
                        if (firstLine == segmentStartLines[curSegmentNo] && info[stopLine - 1].angle >= MIN_ANGLE && info[stopLine - 1].angle <= MAX_ANGLE)
                        {
                            wrapCase = 1;
                            specialCase = true;
                        }

                        // Case 2: Combine the last two lines with the first line of the segment
                        else if (lastLine == stopLine - 1 && info[lastLine].angle >= MIN_ANGLE && info[lastLine].angle <= MAX_ANGLE)
                        {
                            wrapCase = 2;
                            specialCase = true;
                        }
                    }

                    // If still not enough for arc generation, then skip
                    if (specialCase == false)
                    {
                        firstLine = lastLine;
                        continue;
                    }
                }

                // Copy the pixels of this segment to an array
                int noPixels = 0;
                double* x = bm->getX();
                double* y = bm->getY();

                // wrapCase 1: Combine the first two lines with the last line of the segment
                if (wrapCase == 1)
                {
                    int index = lines[stopLine - 1].firstPixelIndex;

                    for (int n = 0; n < lines[stopLine - 1].len; n++)
                    {
                        x[noPixels] = segmentPoints[curSegmentNo][index + n].x;
                        y[noPixels] = segmentPoints[curSegmentNo][index + n].y;
                        noPixels++;
                    }
                }

                for (int m = firstLine; m <= lastLine; m++)
                {
                    int index = lines[m].firstPixelIndex;

                    for (int n = 0; n < lines[m].len; n++)
                    {
                        x[noPixels] = segmentPoints[curSegmentNo][index + n].x;
                        y[noPixels] = segmentPoints[curSegmentNo][index + n].y;
                        noPixels++;
                    }
                }

                // wrapCase 2: Combine the last two lines with the first line of the segment
                if (wrapCase == 2)
                {
                    int index = lines[segmentStartLines[curSegmentNo]].firstPixelIndex;

                    for (int n = 0; n < lines[segmentStartLines[curSegmentNo]].len; n++)
                    {
                        x[noPixels] = segmentPoints[curSegmentNo][index + n].x;
                        y[noPixels] = segmentPoints[curSegmentNo][index + n].y;
                        noPixels++;
                    }
                }

                // Move buffer pointers
                bm->move(noPixels);

                // Try to fit a circle to the entire arc of lines
                double xc = -1, yc = -1, radius = -1, circleFitError = -1;
                CircleFit(x, y, noPixels, &xc, &yc, &radius, &circleFitError);

                double coverage = noPixels / (CV_2PI * radius);

                // In the case of the special case, the arc must cover at least 22.5 degrees
                if (specialCase && coverage < 1.0 / 16)
                {
                    info[firstLine].taken = true;
                    firstLine = lastLine;
                    continue;
                }

                // If only 3 lines, use the SHORT_ARC_ERROR
                double MYERROR = SHORT_ARC_ERROR;
                if (lastLine - firstLine >= 3)
                    MYERROR = LONG_ARC_ERROR;
                if (circleFitError <= MYERROR)
                {
                    // Add this to the list of arcs
                    if (wrapCase == 1)
                    {
                        x += lines[stopLine - 1].len;
                        y += lines[stopLine - 1].len;
                        noPixels -= lines[stopLine - 1].len;
                    }
                    else if (wrapCase == 2)
                    {
                        noPixels -= lines[segmentStartLines[curSegmentNo]].len;
                    }

                    if ((coverage >= FULL_CIRCLE_RATIO && circleFitError <= LONG_ARC_ERROR))
                    {
                        addCircle(circles1, noCircles1, xc, yc, radius, circleFitError, x, y, noPixels);
                    }
                    else
                    {
                        double sTheta, eTheta;
                        ComputeStartAndEndAngles(xc, yc, radius, x, y, noPixels, &sTheta, &eTheta);

                        addArc(edarcs1->arcs, edarcs1->noArcs, xc, yc, radius, circleFitError, sTheta, eTheta, info[firstLine].sign, curSegmentNo,
                            (int)x[0], (int)y[0], (int)x[noPixels - 1], (int)y[noPixels - 1], x, y, noPixels);
                    }

                    for (int m = firstLine; m < lastLine; m++)
                        info[m].taken = true;
                    firstLine = lastLine;
                    continue;
                }

                // Check if this is an almost closed loop (i.e, if 60% of the circle is present). If so, try to fit an ellipse to the entire arc of lines
                double dx = x[0] - x[noPixels - 1];
                double dy = y[0] - y[noPixels - 1];
                double distanceBetweenEndPoints = sqrt(dx * dx + dy * dy);

                bool isAlmostClosedLoop = (distanceBetweenEndPoints <= 1.72 * radius && coverage >= FULL_CIRCLE_RATIO);
                if (isAlmostClosedLoop || (iter == 1 && coverage >= 0.25))    // an arc covering at least 90 degrees
                {
                    EllipseEquation eq;
                    double ellipseFitError = 1e10;

                    bool valid = EllipseFit(x, y, noPixels, &eq);
                    if (valid)
                        ellipseFitError = ComputeEllipseError(&eq, x, y, noPixels);

                    MYERROR = ELLIPSE_ERROR;
                    if (isAlmostClosedLoop == false)
                        MYERROR = 0.75;

                    if (ellipseFitError <= MYERROR)
                    {
                        // Add this to the list of arcs
                        if (wrapCase == 1)
                        {
                            x += lines[stopLine - 1].len;
                            y += lines[stopLine - 1].len;
                            noPixels -= lines[stopLine - 1].len;
                        }
                        else if (wrapCase == 2)
                        {
                            noPixels -= lines[segmentStartLines[curSegmentNo]].len;
                        }

                        if (isAlmostClosedLoop)
                        {
                            addCircle(circles1, noCircles1, xc, yc, radius, circleFitError, &eq, ellipseFitError, x, y, noPixels);  // Add an ellipse for validation
                        }
                        else
                        {
                            double sTheta, eTheta;
                            ComputeStartAndEndAngles(xc, yc, radius, x, y, noPixels, &sTheta, &eTheta);

                            addArc(edarcs1->arcs, edarcs1->noArcs, xc, yc, radius, circleFitError, sTheta, eTheta, info[firstLine].sign, curSegmentNo, &eq, ellipseFitError,
                                (int)x[0], (int)y[0], (int)x[noPixels - 1], (int)y[noPixels - 1], x, y, noPixels);
                        }

                        for (int m = firstLine; m < lastLine; m++)
                            info[m].taken = true;
                        firstLine = lastLine;
                        continue;
                    }
                }

                if (specialCase)
                {
                    info[firstLine].taken = true;
                    firstLine = lastLine;
                    continue;
                }

                // Continue until we finish all lines that belong to arc of lines
                while (firstLine <= lastLine - 2)
                {
                    // Fit an initial arc and extend it
                    int curLine = firstLine + 2;

                    // Fit a circle to the pixels of these lines and see if the error is less than a threshold
                    double XC(0), YC(0), R(0), Error = 1e10;
                    bool found = false;

                    noPixels = 0;
                    while (curLine <= lastLine)
                    {
                        noPixels = 0;
                        for (int m = firstLine; m <= curLine; m++)
                            noPixels += lines[m].len;

                        // Fit circle
                        CircleFit(x, y, noPixels, &XC, &YC, &R, &Error);
                        if (Error <= SHORT_ARC_ERROR)
                        {
                            found = true;    // found if the error is smaller than the threshold
                            break;
                        }

                        // Not found. Move to the next set of lines
                        x += lines[firstLine].len;
                        y += lines[firstLine].len;

                        firstLine++;
                        curLine++;
                    }

                    // If no initial arc found, then we are done with this arc of lines
                    if (!found)
                        break;

                    // If we found an initial arc, then extend it
                    for (int m = curLine - 2; m <= curLine; m++)
                        info[m].taken = true;
                    curLine++;

                    while (curLine <= lastLine)
                    {
                        int noPixelsSave = noPixels;

                        noPixels += lines[curLine].len;

                        double r, error;
                        CircleFit(x, y, noPixels, &xc, &yc, &r, &error);
                        if (error > LONG_ARC_ERROR)
                        {
                            noPixels = noPixelsSave;    // Adding this line made the error big. So, we do not use this line
                            break;
                        }

                        // OK. Longer arc
                        XC = xc;
                        YC = yc;
                        R = r;
                        Error = error;

                        info[curLine].taken = true;
                        curLine++;
                    }

                    coverage = noPixels / (CV_2PI * radius);
                    if ((coverage >= FULL_CIRCLE_RATIO && circleFitError <= LONG_ARC_ERROR))
                    {
                        addCircle(circles1, noCircles1, XC, YC, R, Error, x, y, noPixels);
                    }
                    else
                    {
                        // Add this to the list of arcs
                        double sTheta, eTheta;
                        ComputeStartAndEndAngles(XC, YC, R, x, y, noPixels, &sTheta, &eTheta);

                        addArc(edarcs1->arcs, edarcs1->noArcs, XC, YC, R, Error, sTheta, eTheta, info[firstLine].sign, curSegmentNo,
                            (int)x[0], (int)y[0], (int)x[noPixels - 1], (int)y[noPixels - 1], x, y, noPixels);
                    }

                    x += noPixels;
                    y += noPixels;

                    firstLine = curLine;
                }
                firstLine = lastLine;
            }
        }
    }
}

// Go over all circles & ellipses and validate them
// The idea here is to look at all pixels of a circle/ellipse
// rather than only the pixels of the lines making up the circle/ellipse
void EdgeDrawingImpl::ValidateCircles(bool validate)
{
    precision = CV_PI / 16;  // Alignment precision

    int points_buffer_size = 8 * (width + height);
    double *px = new double[points_buffer_size];
    double *py = new double[points_buffer_size];

    if (nfa->LUTSize == 1 && params.NFAValidation)
    {
        int lutSize = (width + height) / 8;
        double prob = 1.0 / 8;  // probability of alignment
        nfa = new NFALUT(lutSize, prob, width, height); // create look up table
    }

    // Validate circles & ellipses
    bool validateAgain;
    int count = 0;
    for (int i = 0; i < noCircles1; )
    {
        Circle* circle = &circles1[i];
        double xc = circle->xc;
        double yc = circle->yc;
        double radius = circle->r;

        // Skip potential invalid circles (sometimes these kinds of candidates get generated!)
        if (radius > MAX(width, height))
        {
            i++;
            continue;
        }

        validateAgain = false;

        int noPoints = (int)(computeEllipsePerimeter(&circle->eq));

        if (noPoints > points_buffer_size)
        {
            i++;
            continue;
        }

        if (circle->isEllipse)
        {
            ComputeEllipsePoints(circle->eq.coeff, px, py, noPoints);
        }
        else
        {
            ComputeCirclePoints(xc, yc, radius, px, py, &noPoints);
        }

        int pr = -1;  // previous row
        int pc = -1;  // previous column

        int tr = -100;
        int tc = -100;

        int noPeripheryPixels = 0;
        int aligned = 0;
        for (int j = 0; j < noPoints; j++)
        {
            int r = (int)(py[j] + 0.5);
            int c = (int)(px[j] + 0.5);

            if (r == pr && c == pc)
                continue;
            noPeripheryPixels++;

            if (r <= 0 || r >= height - 1)
                continue;
            if (c <= 0 || c >= width - 1)
                continue;

            pr = r;
            pc = c;

            int dr = abs(r - tr);
            int dc = abs(c - tc);
            if (dr + dc >= 2)
            {
                tr = r;
                tc = c;
            }

            //
            // See if there is an edge pixel within 1 pixel vicinity
            //
            if (edgeImg[r * width + c] != 255)
            {
                //   y-cy=-x-cx    y-cy=x-cx
                //         \       /
                //          \ IV. /
                //           \   /
                //            \ /
                //     III.    +   I. quadrant
                //            / \
                //           /   \
                //          / II. \
                //         /       \
                //
                // (x, y)-->(x-cx, y-cy)
                //

                int x = c;
                int y = r;

                int diff1 = (int)(y - yc - x + xc);
                int diff2 = (int)(y - yc + x - xc);

                if (diff1 < 0)
                {
                    if (diff2 > 0)
                    {
                        // I. quadrant
                        c = x - 1;
                        if (c >= 1 && edgeImg[r * width + c] == 255)
                            goto out;
                        c = x + 1;
                        if (c < width - 1 && edgeImg[r * width + c] == 255)
                            goto out;

                        c = x - 2;
                        if (c >= 2 && edgeImg[r * width + c] == 255)
                            goto out;
                        c = x + 2;
                        if (c < width - 2 && edgeImg[r * width + c] == 255)
                            goto out;
                    }
                    else
                    {
                        // IV. quadrant
                        r = y - 1;
                        if (r >= 1 && edgeImg[r * width + c] == 255)
                            goto out;
                        r = y + 1;
                        if (r < height - 1 && edgeImg[r * width + c] == 255)
                            goto out;

                        r = y - 2;
                        if (r >= 2 && edgeImg[r * width + c] == 255)
                            goto out;
                        r = y + 2;
                        if (r < height - 2 && edgeImg[r * width + c] == 255)
                            goto out;
                    }
                }
                else
                {
                    if (diff2 > 0)
                    {
                        // II. quadrant
                        r = y - 1;
                        if (r >= 1 && edgeImg[r * width + c] == 255)
                            goto out;
                        r = y + 1;
                        if (r < height - 1 && edgeImg[r * width + c] == 255)
                            goto out;

                        r = y - 2;
                        if (r >= 2 && edgeImg[r * width + c] == 255)
                            goto out;
                        r = y + 2;
                        if (r < height - 2 && edgeImg[r * width + c] == 255)
                            goto out;
                    }
                    else
                    {
                        // III. quadrant
                        c = x - 1;
                        if (c >= 1 && edgeImg[r * width + c] == 255)
                            goto out;
                        c = x + 1;
                        if (c < width - 1 && edgeImg[r * width + c] == 255)
                            goto out;

                        c = x - 2;
                        if (c >= 2 && edgeImg[r * width + c] == 255)
                            goto out;
                        c = x + 2;
                        if (c < width - 2 && edgeImg[r * width + c] == 255)
                            goto out;
                    }
                }

                r = pr;
                c = pc;
                continue;  // Ignore non-edge pixels.
                           // This produces less false positives, but occationally misses on some valid circles
            }
        out:
            // compute gx & gy
            int com1 = smoothImg[(r + 1) * width + c + 1] - smoothImg[(r - 1) * width + c - 1];
            int com2 = smoothImg[(r - 1) * width + c + 1] - smoothImg[(r + 1) * width + c - 1];

            int gx = com1 + com2 + smoothImg[r * width + c + 1] - smoothImg[r * width + c - 1];
            int gy = com1 - com2 + smoothImg[(r + 1) * width + c] - smoothImg[(r - 1) * width + c];
            double pixelAngle = nfa->myAtan2((double)gx, (double)-gy);

            double derivX, derivY;
            if (circle->isEllipse)
            {
                // Ellipse
                derivX = 2 * circle->eq.A() * c + circle->eq.B() * r + circle->eq.D();
                derivY = circle->eq.B() * c + 2 * circle->eq.C() * r + circle->eq.E();
            }
            else
            {
                // circle
                derivX = c - xc;
                derivY = r - yc;
            }

            double idealPixelAngle = nfa->myAtan2(derivX, -derivY);
            double diff = fabs(pixelAngle - idealPixelAngle);
            if (diff <= precision || diff >= CV_PI - precision)
                aligned++;
        }

        bool isValid = !validate || nfa->checkValidationByNFA(noPeripheryPixels, aligned);

        if (isValid)
        {
            circles2[count++] = circles1[i];
        }
        else if (circle->isEllipse == false && circle->coverRatio >= CANDIDATE_ELLIPSE_RATIO)
        {
            // Fit an ellipse to this circle, and try to revalidate
            double ellipseFitError = 1e10;
            EllipseEquation eq;

            if (EllipseFit(circle->x, circle->y, circle->noPixels, &eq))
            {
                ellipseFitError = ComputeEllipseError(&eq, circle->x, circle->y, circle->noPixels);
            }

            if (ellipseFitError <= ELLIPSE_ERROR)
            {
                circle->isEllipse = true;
                circle->ellipseFitError = ellipseFitError;
                circle->eq = eq;

                validateAgain = true;
            }
        }

        if (validateAgain == false)
            i++;
    }

    noCircles2 = count;

    delete[] px;
    delete[] py;
}

void EdgeDrawingImpl::JoinCircles()
{
    // Sort the circles wrt their radius
    sortCircles(circles2, noCircles2);

    noCircles = noCircles2;
    Circle* circles = circles2;

    vector<bool> taken;
    vector<int> candidateCircles;
    int noCandidateCircles;

    for (int i = 0; i < noCircles; i++)
    {
        taken.push_back(false);
        candidateCircles.push_back(0);

        if (circles[i].isEllipse)
        {
            ComputeEllipseCenterAndAxisLengths(&circles[i].eq, &circles[i].xc, &circles[i].yc, &circles[i].majorAxisLength, &circles[i].minorAxisLength);
        }
    }

    for (int i = 0; i < noCircles; i++)
    {
        if (taken[i])
            continue;

        // Current arc
        double majorAxisLength, minorAxisLength;

        if (circles[i].isEllipse)
        {
            majorAxisLength = circles[i].majorAxisLength;
            minorAxisLength = circles[i].minorAxisLength;
        }
        else
        {
            majorAxisLength = circles[i].r;
            minorAxisLength = circles[i].r;
        }

        // Find other circles to join with
        noCandidateCircles = 0;

        for (int j = i + 1; j < noCircles; j++)
        {
            if (taken[j])
                continue;

#define JOINED_SHORT_ARC_ERROR_THRESHOLD  2
#define AXIS_LENGTH_DIFF_THRESHOLD     6 //(JOINED_SHORT_ARC_ERROR_THRESHOLD*2+1)
#define CENTER_DISTANCE_THRESHOLD      12 //(AXIS_LENGTH_DIFF_THRESHOLD*2)

            double dx = circles[i].xc - circles[j].xc;
            double dy = circles[i].yc - circles[j].yc;
            double centerDistance = sqrt(dx * dx + dy * dy);
            if (centerDistance > CENTER_DISTANCE_THRESHOLD)
                continue;

            double diff1, diff2;
            if (circles[j].isEllipse)
            {
                diff1 = fabs(majorAxisLength - circles[j].majorAxisLength);
                diff2 = fabs(minorAxisLength - circles[j].minorAxisLength);
            }
            else
            {
                diff1 = fabs(majorAxisLength - circles[j].r);
                diff2 = fabs(minorAxisLength - circles[j].r);
            }

            if (diff1 > AXIS_LENGTH_DIFF_THRESHOLD)
                continue;
            if (diff2 > AXIS_LENGTH_DIFF_THRESHOLD)
                continue;

            // Add to candidates
            candidateCircles[noCandidateCircles] = j;
            noCandidateCircles++;
        }

        // Try to join the current arc with the candidate arc (if there is one)
        double XC = circles[i].xc;
        double YC = circles[i].yc;
        double R = circles[i].r;

        double CircleFitError = circles[i].circleFitError;
        bool CircleFitValid = false;

        EllipseEquation Eq;
        double EllipseFitError(0);
        bool EllipseFitValid = false;

        if (noCandidateCircles > 0)
        {
            int noPixels = circles[i].noPixels;
            double* x = bm->getX();
            double* y = bm->getY();
            memcpy(x, circles[i].x, noPixels * sizeof(double));
            memcpy(y, circles[i].y, noPixels * sizeof(double));

            for (int j = 0; j < noCandidateCircles; j++)
            {
                int CandidateArcNo = candidateCircles[j];

                int noPixelsSave = noPixels;
                memcpy(x + noPixels, circles[CandidateArcNo].x, circles[CandidateArcNo].noPixels * sizeof(double));
                memcpy(y + noPixels, circles[CandidateArcNo].y, circles[CandidateArcNo].noPixels * sizeof(double));
                noPixels += circles[CandidateArcNo].noPixels;

                bool circleFitOK = false;
                if (EllipseFitValid == false && circles[i].isEllipse == false && circles[CandidateArcNo].isEllipse == false)
                {
                    double xc, yc, r, error = 1e10;
                    CircleFit(x, y, noPixels, &xc, &yc, &r, &error);

                    if (error <= JOINED_SHORT_ARC_ERROR_THRESHOLD)
                    {
                        taken[CandidateArcNo] = true;

                        XC = xc;
                        YC = yc;
                        R = r;
                        CircleFitError = error;

                        circleFitOK = true;
                        CircleFitValid = true;
                    }
                }

                bool ellipseFitOK = false;
                if (circleFitOK == false)
                {
                    // Try to fit an ellipse
                    double error = 1e10;
                    EllipseEquation eq;
                    if (EllipseFit(x, y, noPixels, &eq))
                    {
                        error = ComputeEllipseError(&eq, x, y, noPixels);
                    }

                    if (error <= JOINED_SHORT_ARC_ERROR_THRESHOLD)
                    {
                        taken[CandidateArcNo] = true;

                        Eq = eq;
                        EllipseFitError = error;

                        ellipseFitOK = true;
                        EllipseFitValid = true;
                        CircleFitValid = false;
                    }
                }

                if (circleFitOK == false && ellipseFitOK == false)
                {
                    noPixels = noPixelsSave;
                }
            }
        }

        // Add the new circle/ellipse to circles2
        if (CircleFitValid)
        {
            addCircle(circles3, noCircles3, XC, YC, R, CircleFitError, NULL, NULL, 0);
        }
        else if (EllipseFitValid)
        {
            addCircle(circles3, noCircles3, XC, YC, R, CircleFitError, &Eq, EllipseFitError, NULL, NULL, 0);
        }
        else
        {
            circles3[noCircles3] = circles[i];
            noCircles3++;
        }
    }
}

void EdgeDrawingImpl::JoinArcs1()
{
    AngleSet angles;

    // Sort the arcs with respect to their length so that longer arcs are at the beginning
    sortArc(edarcs1->arcs, edarcs1->noArcs);

    int noArcs = edarcs1->noArcs;
    MyArc* arcs = edarcs1->arcs;

    bool* taken = new bool[noArcs];
    for (int i = 0; i < noArcs; i++)
        taken[i] = false;

    struct CandidateArc
    {
        int arcNo;
        int which;    // 1: (SX, SY)-(sx, sy), 2: (SX, SY)-(ex, ey), 3: (EX, EY)-(sx, sy), 4: (EX, EY)-(ex, ey)
        double dist;  // min distance between the end points
    };

    CandidateArc* candidateArcs = new CandidateArc[noArcs];
    int noCandidateArcs;

    for (int i = 0; i < noArcs; i++)
    {
        if (taken[i])
            continue;
        if (arcs[i].isEllipse)
        {
            edarcs2->arcs[edarcs2->noArcs++] = arcs[i];
            continue;
        }

        // Current arc
        bool CircleEqValid = false;
        double XC = arcs[i].xc;
        double YC = arcs[i].yc;
        double R = arcs[i].r;
        double CircleFitError = arcs[i].circleFitError;
        int Turn = arcs[i].turn;
        int NoPixels = arcs[i].noPixels;

        int SX = arcs[i].sx;
        int SY = arcs[i].sy;
        int EX = arcs[i].ex;
        int EY = arcs[i].ey;

        // Take the pixels making up this arc
        int noPixels = arcs[i].noPixels;

        double* x = bm->getX();
        double* y = bm->getY();
        memcpy(x, arcs[i].x, noPixels * sizeof(double));
        memcpy(y, arcs[i].y, noPixels * sizeof(double));

        angles.clear();
        angles.set(arcs[i].sTheta, arcs[i].eTheta);

        while (1)
        {
            bool extendedArc = false;

            // Find other arcs to join with
            noCandidateArcs = 0;

            for (int j = i + 1; j < noArcs; j++)
            {
                if (taken[j])
                    continue;
                if (arcs[j].isEllipse)
                    continue;

                double minR = MIN(R, arcs[j].r);
                double radiusDiffThreshold = minR * 0.25;

                double diff = fabs(R - arcs[j].r);
                if (diff > radiusDiffThreshold)
                    continue;

                // If 50% of the current arc overlaps with the existing arc, then ignore this arc
                if (angles.overlap(arcs[j].sTheta, arcs[j].eTheta) >= 0.50)
                    continue;

                // Compute the distances
                // 1: (SX, SY)-(sx, sy)
                double dx = SX - arcs[j].sx;
                double dy = SY - arcs[j].sy;
                double d = sqrt(dx * dx + dy * dy);
                int which = 1;

                // 2: (SX, SY)-(ex, ey)
                dx = SX - arcs[j].ex;
                dy = SY - arcs[j].ey;
                double d2 = sqrt(dx * dx + dy * dy);

                if (d2 < d)
                {
                    d = d2;
                    which = 2;
                }

                // 3: (EX, EY)-(sx, sy)
                dx = EX - arcs[j].sx;
                dy = EY - arcs[j].sy;
                d2 = sqrt(dx * dx + dy * dy);

                if (d2 < d)
                {
                    d = d2;
                    which = 3;
                }

                // 4: (EX, EY)-(ex, ey)
                dx = EX - arcs[j].ex;
                dy = EY - arcs[j].ey;
                d2 = sqrt(dx * dx + dy * dy);

                if (d2 < d)
                {
                    d = d2;
                    which = 4;
                }

                // Endpoints must be very close
                double maxDistanceBetweenEndpoints = minR * 1.75; //1.5;
                if (d > maxDistanceBetweenEndpoints)
                    continue;

                // This is to give precedence to better matching arc
                d += diff;

                // They have to turn in the same direction
                if (which == 2 || which == 3)
                {
                    if (Turn != arcs[j].turn)
                        continue;
                }
                else
                {
                    if (Turn == arcs[j].turn)
                        continue;
                }

                // Add to candidate arcs in sorted order. User insertion sort
                int index = noCandidateArcs - 1;

                while (index >= 0)
                {
                    if (candidateArcs[index].dist < d)
                        break;

                    candidateArcs[index + 1] = candidateArcs[index];
                    index--;
                }

                // Add the new candidate arc to the candidate list
                index++;
                candidateArcs[index].arcNo = j;
                candidateArcs[index].which = which;
                candidateArcs[index].dist = d;
                noCandidateArcs++;
            }

            // Try to join the current arc with the candidate arc (if there is one)
            if (noCandidateArcs > 0)
            {
                for (int j = 0; j < noCandidateArcs; j++)
                {
                    int CandidateArcNo = candidateArcs[j].arcNo;
                    int Which = candidateArcs[j].which;

                    int noPixelsSave = noPixels;
                    memcpy(x + noPixels, arcs[CandidateArcNo].x, arcs[CandidateArcNo].noPixels * sizeof(double));
                    memcpy(y + noPixels, arcs[CandidateArcNo].y, arcs[CandidateArcNo].noPixels * sizeof(double));
                    noPixels += arcs[CandidateArcNo].noPixels;

                    double xc, yc, r, circleFitError;
                    CircleFit(x, y, noPixels, &xc, &yc, &r, &circleFitError);

                    if (circleFitError > LONG_ARC_ERROR)
                    {
                        // No match. Continue with the next candidate
                        noPixels = noPixelsSave;
                    }
                    else
                    {
                        // Match. Take it
                        extendedArc = true;
                        CircleEqValid = true;
                        XC = xc;
                        YC = yc;
                        R = r;
                        CircleFitError = circleFitError;
                        NoPixels = noPixels;

                        taken[CandidateArcNo] = true;
                        taken[i] = true;

                        angles.set(arcs[CandidateArcNo].sTheta, arcs[CandidateArcNo].eTheta);

                        // Update the end points of the new arc
                        switch (Which)
                        {
                            // (SX, SY)-(sy, sy)
                        case 1:
                            SX = EX, SY = EY;
                            EX = arcs[CandidateArcNo].ex;
                            EY = arcs[CandidateArcNo].ey;
                            if (Turn == 1)
                                Turn = -1;
                            else
                                Turn = 1; // reverse the turn direction
                            break;

                            // (SX, SY)-(ex, ey)
                        case 2:
                            SX = EX, SY = EY;
                            EX = arcs[CandidateArcNo].sx;
                            EY = arcs[CandidateArcNo].sy;
                            if (Turn == 1)
                                Turn = -1;
                            else
                                Turn = 1; // reverse the turn direction
                            break;

                            // (EX, EY)-(sx, sy)
                        case 3:
                            EX = arcs[CandidateArcNo].ex;
                            EY = arcs[CandidateArcNo].ey;
                            break;

                            // (EX, EY)-(ex, ey)
                        case 4:
                            EX = arcs[CandidateArcNo].sx;
                            EY = arcs[CandidateArcNo].sy;
                            break;
                        } //end-switch

                        break; // Do not look at the other candidates
                    }
                }
            }

            if (extendedArc == false)
                break;
        }

        if (CircleEqValid == false)
        {
            // Add to arcs
            edarcs2->arcs[edarcs2->noArcs++] = arcs[i];
        }
        else
        {
            // Add the current OR the extended arc to the new arcs
            double sTheta, eTheta;
            angles.computeStartEndTheta(sTheta, eTheta);

            double coverage = ArcLength(sTheta, eTheta) / CV_2PI;
            if ((coverage >= FULL_CIRCLE_RATIO && CircleFitError <= LONG_ARC_ERROR))
                addCircle(circles1, noCircles1, XC, YC, R, CircleFitError, x, y, NoPixels);
            else
                addArc(edarcs2->arcs, edarcs2->noArcs, XC, YC, R, CircleFitError, sTheta, eTheta, Turn, arcs[i].segmentNo, SX, SY, EX, EY, x, y, NoPixels, angles.overlapRatio());

            bm->move(NoPixels);
        }
    }

    delete[] taken;
    delete[] candidateArcs;
}

void EdgeDrawingImpl::JoinArcs2()
{
    AngleSet angles;

    // Sort the arcs with respect to their length so that longer arcs are at the beginning
    sortArc(edarcs2->arcs, edarcs2->noArcs);

    int noArcs = edarcs2->noArcs;
    MyArc* arcs = edarcs2->arcs;

    bool* taken = new bool[noArcs];
    for (int i = 0; i < noArcs; i++)
        taken[i] = false;

    struct CandidateArc
    {
        int arcNo;
        int which;    // 1: (SX, SY)-(sx, sy), 2: (SX, SY)-(ex, ey), 3: (EX, EY)-(sx, sy), 4: (EX, EY)-(ex, ey)
        double dist;  // min distance between the end points
    };

    CandidateArc* candidateArcs = new CandidateArc[noArcs];
    int noCandidateArcs;

    for (int i = 0; i < noArcs; i++)
    {
        if (taken[i])
            continue;

        // Current arc
        bool EllipseEqValid = false;
        EllipseEquation Eq;
        double EllipseFitError(0);

        double R = arcs[i].r;
        int Turn = arcs[i].turn;
        int NoPixels = arcs[i].noPixels;

        int SX = arcs[i].sx;
        int SY = arcs[i].sy;
        int EX = arcs[i].ex;
        int EY = arcs[i].ey;

        // Take the pixels making up this arc
        int noPixels = arcs[i].noPixels;

        double* x = bm->getX();
        double* y = bm->getY();
        memcpy(x, arcs[i].x, noPixels * sizeof(double));
        memcpy(y, arcs[i].y, noPixels * sizeof(double));

        angles.clear();
        angles.set(arcs[i].sTheta, arcs[i].eTheta);

        while (1)
        {
            bool extendedArc = false;

            // Find other arcs to join with
            noCandidateArcs = 0;

            for (int j = i + 1; j < noArcs; j++)
            {
                if (taken[j])
                    continue;
                if (arcs[j].segmentNo != arcs[i].segmentNo)
                    continue;
                if (arcs[j].turn != Turn)
                    continue;

                double minR = MIN(R, arcs[j].r);
                double radiusDiffThreshold = minR * 2.5;

                double diff = fabs(R - arcs[j].r);
                if (diff > radiusDiffThreshold)
                    continue;

                // If 75% of the current arc overlaps with the existing arc, then ignore this arc
                if (angles.overlap(arcs[j].sTheta, arcs[j].eTheta) >= 0.75)
                    continue;

                // Compute the distances
                // 1: (SX, SY)-(sx, sy)
                double dx = SX - arcs[j].sx;
                double dy = SY - arcs[j].sy;
                double d = sqrt(dx * dx + dy * dy);
                int which = 1;

                // 2: (SX, SY)-(ex, ey)
                dx = SX - arcs[j].ex;
                dy = SY - arcs[j].ey;
                double d2 = sqrt(dx * dx + dy * dy);

                if (d2 < d)
                {
                    d = d2;
                    which = 2;
                }

                // 3: (EX, EY)-(sx, sy)
                dx = EX - arcs[j].sx;
                dy = EY - arcs[j].sy;
                d2 = sqrt(dx * dx + dy * dy);

                if (d2 < d)
                {
                    d = d2;
                    which = 3;
                }

                // 4: (EX, EY)-(ex, ey)
                dx = EX - arcs[j].ex;
                dy = EY - arcs[j].ey;
                d2 = sqrt(dx * dx + dy * dy);

                if (d2 < d)
                {
                    d = d2;
                    which = 4;
                }

                // Endpoints must be very close
                double maxDistanceBetweenEndpoints = 5;
                if (d > maxDistanceBetweenEndpoints)
                    continue;

                // Add to candidate arcs in sorted order. User insertion sort
                int index = noCandidateArcs - 1;
                while (index >= 0)
                {
                    if (candidateArcs[index].dist < d)
                        break;

                    candidateArcs[index + 1] = candidateArcs[index];
                    index--;
                }

                // Add the new candidate arc to the candidate list
                index++;
                candidateArcs[index].arcNo = j;
                candidateArcs[index].which = which;
                candidateArcs[index].dist = d;
                noCandidateArcs++;
            }

            // Try to join the current arc with the candidate arc (if there is one)
            if (noCandidateArcs > 0)
            {
                for (int j = 0; j < noCandidateArcs; j++)
                {
                    int CandidateArcNo = candidateArcs[j].arcNo;
                    int Which = candidateArcs[j].which;

                    int noPixelsSave = noPixels;
                    memcpy(x + noPixels, arcs[CandidateArcNo].x, arcs[CandidateArcNo].noPixels * sizeof(double));
                    memcpy(y + noPixels, arcs[CandidateArcNo].y, arcs[CandidateArcNo].noPixels * sizeof(double));
                    noPixels += arcs[CandidateArcNo].noPixels;

                    // Directly fit an ellipse
                    EllipseEquation eq;
                    double ellipseFitError = 1e10;
                    if (EllipseFit(x, y, noPixels, &eq))
                        ellipseFitError = ComputeEllipseError(&eq, x, y, noPixels);

                    if (ellipseFitError > ELLIPSE_ERROR)
                    {
                        // No match. Continue with the next candidate
                        noPixels = noPixelsSave;
                    }
                    else
                    {
                        // Match. Take it
                        extendedArc = true;
                        EllipseEqValid = true;
                        Eq = eq;
                        EllipseFitError = ellipseFitError;
                        NoPixels = noPixels;

                        taken[CandidateArcNo] = true;
                        taken[i] = true;

                        R = (R + arcs[CandidateArcNo].r) / 2.0;

                        angles.set(arcs[CandidateArcNo].sTheta, arcs[CandidateArcNo].eTheta);

                        // Update the end points of the new arc
                        switch (Which)
                        {
                            // (SX, SY)-(sy, sy)
                        case 1:
                            SX = EX, SY = EY;
                            EX = arcs[CandidateArcNo].ex;
                            EY = arcs[CandidateArcNo].ey;
                            if (Turn == 1)
                                Turn = -1;
                            else
                                Turn = 1; // reverse the turn direction
                            break;

                            // (SX, SY)-(ex, ey)
                        case 2:
                            SX = EX, SY = EY;
                            EX = arcs[CandidateArcNo].sx;
                            EY = arcs[CandidateArcNo].sy;
                            if (Turn == 1)
                                Turn = -1;
                            else
                                Turn = 1; // reverse the turn direction
                            break;

                            // (EX, EY)-(sx, sy)
                        case 3:
                            EX = arcs[CandidateArcNo].ex;
                            EY = arcs[CandidateArcNo].ey;
                            break;

                            // (EX, EY)-(ex, ey)
                        case 4:
                            EX = arcs[CandidateArcNo].sx;
                            EY = arcs[CandidateArcNo].sy;
                            break;
                        }

                        break; // Do not look at the other candidates
                    }
                }
            }

            if (extendedArc == false)
                break;
        }

        if (EllipseEqValid == false)
        {
            // Add to arcs
            edarcs3->arcs[edarcs3->noArcs++] = arcs[i];
        }
        else
        {
            // Add the current OR the extended arc to the new arcs
            double sTheta, eTheta;
            angles.computeStartEndTheta(sTheta, eTheta);

            double XC, YC, CircleFitError;
            CircleFit(x, y, NoPixels, &XC, &YC, &R, &CircleFitError);

            double coverage = ArcLength(sTheta, eTheta) / CV_2PI;
            if ((coverage >= FULL_CIRCLE_RATIO && CircleFitError <= LONG_ARC_ERROR))
                addCircle(circles1, noCircles1, XC, YC, R, CircleFitError, x, y, NoPixels);
            else
                addArc(edarcs3->arcs, edarcs3->noArcs, XC, YC, R, CircleFitError, sTheta, eTheta, Turn, arcs[i].segmentNo, &Eq, EllipseFitError, SX, SY, EX, EY, x, y, NoPixels, angles.overlapRatio());

            // Move buffer pointers
            bm->move(NoPixels);
        }
    }

    delete[] taken;
    delete[] candidateArcs;
}

void EdgeDrawingImpl::JoinArcs3()
{
    AngleSet angles;

    // Sort the arcs with respect to their length so that longer arcs are at the beginning
    sortArc(edarcs3->arcs, edarcs3->noArcs);

    int noArcs = edarcs3->noArcs;
    MyArc* arcs = edarcs3->arcs;

    bool* taken = new bool[noArcs];
    for (int i = 0; i < noArcs; i++)
        taken[i] = false;

    struct CandidateArc
    {
        int arcNo;
        int which;    // 1: (SX, SY)-(sx, sy), 2: (SX, SY)-(ex, ey), 3: (EX, EY)-(sx, sy), 4: (EX, EY)-(ex, ey)
        double dist;  // min distance between the end points
    };

    CandidateArc* candidateArcs = new CandidateArc[noArcs];
    int noCandidateArcs;

    for (int i = 0; i < noArcs; i++)
    {
        if (taken[i])
            continue;

        // Current arc
        bool EllipseEqValid = false;
        EllipseEquation Eq;
        double EllipseFitError(0);

        double R = arcs[i].r;
        int Turn = arcs[i].turn;
        int NoPixels = arcs[i].noPixels;

        int SX = arcs[i].sx;
        int SY = arcs[i].sy;
        int EX = arcs[i].ex;
        int EY = arcs[i].ey;

        // Take the pixels making up this arc
        int noPixels = arcs[i].noPixels;

        double* x = bm->getX();
        double* y = bm->getY();
        memcpy(x, arcs[i].x, noPixels * sizeof(double));
        memcpy(y, arcs[i].y, noPixels * sizeof(double));

        angles.clear();
        angles.set(arcs[i].sTheta, arcs[i].eTheta);

        while (1)
        {
            bool extendedArc = false;

            // Find other arcs to join with
            noCandidateArcs = 0;

            for (int j = i + 1; j < noArcs; j++)
            {
                if (taken[j])
                    continue;

                /******************************************************************
                * It seems that for minimum false detections,
                * radiusDiffThreshold =  minR*0.5 & maxDistanceBetweenEndpoints = minR*0.75.
                * But these parameters results in many valid misses too!
                ******************************************************************/

                double minR = MIN(R, arcs[j].r);
                double diff = fabs(R - arcs[j].r);
                if (diff > minR)
                    continue;

                // If 50% of the current arc overlaps with the existing arc, then ignore this arc
                if (angles.overlap(arcs[j].sTheta, arcs[j].eTheta) >= 0.50)
                    continue;

                // Compute the distances
                // 1: (SX, SY)-(sx, sy)
                double dx = SX - arcs[j].sx;
                double dy = SY - arcs[j].sy;
                double d = sqrt(dx * dx + dy * dy);
                int which = 1;

                // 2: (SX, SY)-(ex, ey)
                dx = SX - arcs[j].ex;
                dy = SY - arcs[j].ey;
                double d2 = sqrt(dx * dx + dy * dy);

                if (d2 < d)
                {
                    d = d2;
                    which = 2;
                }

                // 3: (EX, EY)-(sx, sy)
                dx = EX - arcs[j].sx;
                dy = EY - arcs[j].sy;
                d2 = sqrt(dx * dx + dy * dy);

                if (d2 < d)
                {
                    d = d2;
                    which = 3;
                }

                // 4: (EX, EY)-(ex, ey)
                dx = EX - arcs[j].ex;
                dy = EY - arcs[j].ey;
                d2 = sqrt(dx * dx + dy * dy);

                if (d2 < d)
                {
                    d = d2;
                    which = 4;
                }

                // Endpoints must be very close
                if (diff <= 0.50 * minR)
                {
                    if (d > minR * 0.75)
                        continue;
                }
                else if (diff <= 0.75 * minR)
                {
                    if (d > minR * 0.50)
                        continue;
                }
                else if (diff <= 1.00 * minR)
                {
                    if (d > minR * 0.25)
                        continue;
                }
                else
                    continue;

                // This is to allow more circular arcs a precedence
                d += diff;

                // They have to turn in the same direction
                if (which == 2 || which == 3)
                {
                    if (Turn != arcs[j].turn)
                        continue;
                }
                else
                {
                    if (Turn == arcs[j].turn)
                        continue;
                }

                // Add to candidate arcs in sorted order. User insertion sort
                int index = noCandidateArcs - 1;
                while (index >= 0)
                {
                    if (candidateArcs[index].dist < d)
                        break;

                    candidateArcs[index + 1] = candidateArcs[index];
                    index--;
                }

                // Add the new candidate arc to the candidate list
                index++;
                candidateArcs[index].arcNo = j;
                candidateArcs[index].which = which;
                candidateArcs[index].dist = d;
                noCandidateArcs++;
            }

            // Try to join the current arc with the candidate arc (if there is one)
            if (noCandidateArcs > 0)
            {
                for (int j = 0; j < noCandidateArcs; j++)
                {
                    int CandidateArcNo = candidateArcs[j].arcNo;
                    int Which = candidateArcs[j].which;

                    int noPixelsSave = noPixels;
                    memcpy(x + noPixels, arcs[CandidateArcNo].x, arcs[CandidateArcNo].noPixels * sizeof(double));
                    memcpy(y + noPixels, arcs[CandidateArcNo].y, arcs[CandidateArcNo].noPixels * sizeof(double));
                    noPixels += arcs[CandidateArcNo].noPixels;

                    // Directly fit an ellipse
                    EllipseEquation eq;
                    double ellipseFitError = 1e10;
                    if (EllipseFit(x, y, noPixels, &eq))
                        ellipseFitError = ComputeEllipseError(&eq, x, y, noPixels);

                    if (ellipseFitError > ELLIPSE_ERROR)
                    {
                        // No match. Continue with the next candidate
                        noPixels = noPixelsSave;
                    }
                    else
                    {
                        // Match. Take it
                        extendedArc = true;
                        EllipseEqValid = true;
                        Eq = eq;
                        EllipseFitError = ellipseFitError;
                        NoPixels = noPixels;

                        taken[CandidateArcNo] = true;
                        taken[i] = true;

                        R = (R + arcs[CandidateArcNo].r) / 2.0;

                        angles.set(arcs[CandidateArcNo].sTheta, arcs[CandidateArcNo].eTheta);

                        // Update the end points of the new arc
                        switch (Which)
                        {
                            // (SX, SY)-(sy, sy)
                        case 1:
                            SX = EX, SY = EY;
                            EX = arcs[CandidateArcNo].ex;
                            EY = arcs[CandidateArcNo].ey;
                            if (Turn == 1)
                                Turn = -1;
                            else
                                Turn = 1; // reverse the turn direction
                            break;

                            // (SX, SY)-(ex, ey)
                        case 2:
                            SX = EX, SY = EY;
                            EX = arcs[CandidateArcNo].sx;
                            EY = arcs[CandidateArcNo].sy;
                            if (Turn == 1)
                                Turn = -1;
                            else
                                Turn = 1; // reverse the turn direction
                            break;

                            // (EX, EY)-(sx, sy)
                        case 3:
                            EX = arcs[CandidateArcNo].ex;
                            EY = arcs[CandidateArcNo].ey;
                            break;

                            // (EX, EY)-(ex, ey)
                        case 4:
                            EX = arcs[CandidateArcNo].sx;
                            EY = arcs[CandidateArcNo].sy;
                            break;
                        }
                        break; // Do not look at the other candidates
                    }
                }
            }

            if (extendedArc == false)
                break;
        }

        if (EllipseEqValid == false)
        {
            // Add to arcs
            edarcs4->arcs[edarcs4->noArcs++] = arcs[i];
        }
        else
        {
            // Add the current OR the extended arc to the new arcs
            double sTheta, eTheta;
            angles.computeStartEndTheta(sTheta, eTheta);

            double XC, YC, CircleFitError;
            CircleFit(x, y, NoPixels, &XC, &YC, &R, &CircleFitError);

            double coverage = ArcLength(sTheta, eTheta) / CV_2PI;
            if ((coverage >= FULL_CIRCLE_RATIO && CircleFitError <= LONG_ARC_ERROR))
                addCircle(circles1, noCircles1, XC, YC, R, CircleFitError, x, y, NoPixels);
            else
                addArc(edarcs4->arcs, edarcs4->noArcs, XC, YC, R, CircleFitError, sTheta, eTheta, Turn, arcs[i].segmentNo, &Eq, EllipseFitError, SX, SY, EX, EY, x, y, NoPixels, angles.overlapRatio());

            bm->move(NoPixels);
        }
    }

    delete[] taken;
    delete[] candidateArcs;
}

void EdgeDrawingImpl::addCircle(Circle* circles, int& noCircles, double xc, double yc, double r, double circleFitError, double* x, double* y, int noPixels)
{
    circles[noCircles].xc = xc;
    circles[noCircles].yc = yc;
    circles[noCircles].r = r;
    circles[noCircles].circleFitError = circleFitError;
    circles[noCircles].coverRatio = noPixels / CV_2PI * r;

    circles[noCircles].x = x;
    circles[noCircles].y = y;
    circles[noCircles].noPixels = noPixels;

    circles[noCircles].isEllipse = false;

    noCircles++;
}

void EdgeDrawingImpl::addCircle(Circle* circles, int& noCircles, double xc, double yc, double r, double circleFitError, EllipseEquation* pEq, double ellipseFitError, double* x, double* y, int noPixels)
{
    circles[noCircles].xc = xc;
    circles[noCircles].yc = yc;
    circles[noCircles].r = r;
    circles[noCircles].circleFitError = circleFitError;
    circles[noCircles].coverRatio = noPixels / computeEllipsePerimeter(pEq);

    circles[noCircles].x = x;
    circles[noCircles].y = y;
    circles[noCircles].noPixels = noPixels;

    circles[noCircles].eq = *pEq;
    circles[noCircles].ellipseFitError = ellipseFitError;
    circles[noCircles].isEllipse = true;

    noCircles++;
}

void EdgeDrawingImpl::sortCircles(Circle* circles, int noCircles)
{
    for (int i = 0; i < noCircles - 1; i++)
    {
        int max = i;
        for (int j = i + 1; j < noCircles; j++)
        {
            if (circles[j].r > circles[max].r)
                max = j;
        }

        if (max != i)
        {
            Circle t = circles[i];
            circles[i] = circles[max];
            circles[max] = t;
        }
    }
}


// ---------------------------------------------------------------------------
// Given an ellipse equation, computes the length of the perimeter of the ellipse
// Calculates the ellipse perimeter wrt the Ramajunan II formula
//
double EdgeDrawingImpl::computeEllipsePerimeter(EllipseEquation* eq)
{
    double mult = 1;

    double A = eq->A() * mult;
    double B = eq->B() * mult;
    double C = eq->C() * mult;
    double D = eq->D() * mult;
    double E = eq->E() * mult;
    double F = eq->F() * mult;

    double A2(0), C2(0), D2(0), E2(0), F2(0), theta(0);  //rotated coefficients
    double D3, E3, F3;                                   //ellipse form coefficients
    double cX, cY, a, b;                                 //(cX,cY) center, a & b: semimajor & semiminor axes
    double h;                                            //h = (a-b)^2 / (a+b)^2
    bool rotation = false;

    //Normalize coefficients
    B /= A;
    C /= A;
    D /= A;
    E /= A;
    F /= A;
    A /= A;

    if (B == 0) //Then not need to rotate the axes
    {
        A2 = A;
        C2 = C;
        D2 = D;
        E2 = E;
        F2 = F;
    }

    else if (B != 0) //Rotate the axes
    {
        rotation = true;

        //Determine the rotation angle (in radians)
        theta = atan(B / (A - C)) / 2;

        //Compute the coefficients wrt the new coordinate system
        A2 = 0.5 * (A * (1 + cos(2 * theta) + B * sin(2 * theta) + C * (1 - cos(2 * theta))));

        C2 = 0.5 * (A * (1 - cos(2 * theta) - B * sin(2 * theta) + C * (1 + cos(2 * theta))));

        D2 = D * cos(theta) + E * sin(theta);

        E2 = -D * sin(theta) + E * cos(theta);

        F2 = F;
    }

    //Transform the conic equation into the ellipse form
    D3 = D2 / A2; //normalize x term's coef

    E3 = E2 / C2; //normalize y term's coef

    cX = -(D3 / 2);   //center X
    cY = -(E3 / 2);   //center Y

    F3 = A2 * pow(cX, 2.0) + C2 * pow(cY, 2.0) - F2;

    //semimajor axis
    a = sqrt(F3 / A2);
    //semiminor axis
    b = sqrt(F3 / C2);

    //Center coordinates have to be re-transformed if rotation is applied!
    if (rotation)
    {
        double tmpX = cX, tmpY = cY;
        cX = tmpX * cos(theta) - tmpY * sin(theta);
        cY = tmpX * sin(theta) + tmpY * cos(theta);
    }

    //Perimeter Computation(s)
    h = pow((a - b), 2.0) / pow((a + b), 2.0);

    //Ramajunan II
    double P2 = CV_PI * (a + b) * (1 + 3 * h / (10 + sqrt(4 - 3 * h)));

    return P2;
}

double EdgeDrawingImpl::ComputeEllipseError(EllipseEquation* eq, double* px, double* py, int noPoints)
{
    double error = 0;

    double A = eq->A();
    double B = eq->B();
    double C = eq->C();
    double D = eq->D();
    double E = eq->E();
    double F = eq->F();

    double xc, yc, major, minor;
    ComputeEllipseCenterAndAxisLengths(eq, &xc, &yc, &major, &minor);

    for (int i = 0; i < noPoints; i++)
    {
        double dx = px[i] - xc;
        double dy = py[i] - yc;

        double min;
        double xs;

        if (fabs(dx) > fabs(dy))
        {
            // The line equation is of the form: y = mx+n
            double m = dy / dx;
            double n = yc - m * xc;

            // a*x^2 + b*x + c
            double a = A + B * m + C * m * m;
            double b = B * n + 2 * C * m * n + D + E * m;
            double c = C * n * n + E * n + F;
            double det = b * b - 4 * a * c;
            if (det < 0)
                det = 0;
            double x1 = -(b + sqrt(det)) / (2 * a);
            double x2 = -(b - sqrt(det)) / (2 * a);

            double y1 = m * x1 + n;
            double y2 = m * x2 + n;

            dx = px[i] - x1;
            dy = py[i] - y1;
            double d1 = dx * dx + dy * dy;

            dx = px[i] - x2;
            dy = py[i] - y2;
            double d2 = dx * dx + dy * dy;

            if (d1 < d2)
            {
                min = d1;
                xs = x1;
            }
            else
            {
                min = d2;
                xs = x2;
            }
        }
        else
        {
            // The line equation is of the form: x = my+n
            double m = dx / dy;
            double n = xc - m * yc;

            // a*y^2 + b*y + c
            double a = A * m * m + B * m + C;
            double b = 2 * A * m * n + B * n + D * m + E;
            double c = A * n * n + D * n + F;
            double det = b * b - 4 * a * c;
            if (det < 0)
                det = 0;
            double y1 = -(b + sqrt(det)) / (2 * a);
            double y2 = -(b - sqrt(det)) / (2 * a);

            double x1 = m * y1 + n;
            double x2 = m * y2 + n;

            dx = px[i] - x1;
            dy = py[i] - y1;
            double d1 = dx * dx + dy * dy;

            dx = px[i] - x2;
            dy = py[i] - y2;
            double d2 = dx * dx + dy * dy;

            if (d1 < d2)
            {
                min = d1;
                xs = x1;
            }
            else
            {
                min = d2;
                xs = x2;
            }
        }

        // Refine the search in the vicinity of (xs, ys)
        double delta = 0.5;
        double x = xs;
        while (1)
        {
            x += delta;

            double a = C;
            double b = B * x + E;
            double c = A * x * x + D * x + F;
            double det = b * b - 4 * a * c;
            if (det < 0)
                det = 0;

            double y1 = -(b + sqrt(det)) / (2 * a);
            double y2 = -(b - sqrt(det)) / (2 * a);

            dx = px[i] - x;
            dy = py[i] - y1;
            double d1 = dx * dx + dy * dy;

            dy = py[i] - y2;
            double d2 = dx * dx + dy * dy;

            if (d1 <= min)
            {
                min = d1;
            }
            else if (d2 <= min)
            {
                min = d2;
            }
            else
                break;
        }

        x = xs;
        while (1)
        {
            x -= delta;

            double a = C;
            double b = B * x + E;
            double c = A * x * x + D * x + F;
            double det = b * b - 4 * a * c;
            if (det < 0)
                det = 0;

            double y1 = -(b + sqrt(det)) / (2 * a);
            double y2 = -(b - sqrt(det)) / (2 * a);

            dx = px[i] - x;
            dy = py[i] - y1;
            double d1 = dx * dx + dy * dy;

            dy = py[i] - y2;
            double d2 = dx * dx + dy * dy;

            if (d1 <= min)
            {
                min = d1;
            }
            else if (d2 <= min)
            {
                min = d2;
            }
            else
                break;
        }
        error += min;
    }

    error = sqrt(error / noPoints);

    return error;
}

// also returns rotate angle theta
double EdgeDrawingImpl::ComputeEllipseCenterAndAxisLengths(EllipseEquation* eq, double* pxc, double* pyc, double* pmajorAxisLength, double* pminorAxisLength)
{
    double mult = 1;

    double A = eq->A() * mult;
    double B = eq->B() * mult;
    double C = eq->C() * mult;
    double D = eq->D() * mult;
    double E = eq->E() * mult;
    double F = eq->F() * mult;

    double A2(0), C2(0), D2(0), E2(0), F2(0), theta(0);  //rotated coefficients
    double D3, E3, F3;                                   //ellipse form coefficients
    double cX, cY, a, b;                                 //(cX,cY) center, a & b: semimajor & semiminor axes
    bool rotation = false;

    //Normalize coefficients
    B /= A;
    C /= A;
    D /= A;
    E /= A;
    F /= A;
    A /= A;

    if (B == 0) //Then not need to rotate the axes
    {
        A2 = A;
        C2 = C;
        D2 = D;
        E2 = E;
        F2 = F;
    }
    else if (B != 0) //Rotate the axes
    {
        rotation = true;

        //Determine the rotation angle (in radians)
        theta = atan(B / (A - C)) / 2;

        //Compute the coefficients wrt the new coordinate system
        A2 = 0.5 * (A * (1 + cos(2 * theta) + B * sin(2 * theta) + C * (1 - cos(2 * theta))));

        C2 = 0.5 * (A * (1 - cos(2 * theta) - B * sin(2 * theta) + C * (1 + cos(2 * theta))));

        D2 = D * cos(theta) + E * sin(theta);

        E2 = -D * sin(theta) + E * cos(theta);

        F2 = F;
    }

    //Transform the conic equation into the ellipse form
    D3 = D2 / A2; //normalize x term's coef

    E3 = E2 / C2; //normalize y term's coef

    cX = -(D3 / 2);   //center X
    cY = -(E3 / 2);   //center Y

    F3 = A2 * pow(cX, 2.0) + C2 * pow(cY, 2.0) - F2;

    //semimajor axis
    a = sqrt(F3 / A2);
    //semiminor axis
    b = sqrt(F3 / C2);

    //Center coordinates have to be re-transformed if rotation is applied!
    if (rotation)
    {
        double tmpX = cX, tmpY = cY;
        cX = tmpX * cos(theta) - tmpY * sin(theta);
        cY = tmpX * sin(theta) + tmpY * cos(theta);
    }

    *pxc = cX;
    *pyc = cY;

    *pmajorAxisLength = a;
    *pminorAxisLength = b;

    return theta;
}

// ---------------------------------------------------------------------------
// Given an ellipse equation, computes "noPoints" many consecutive points
// on the ellipse periferi. These points can be used to draw the ellipse
//
void EdgeDrawingImpl::ComputeEllipsePoints(double* pvec, double* px, double* py, int noPoints)
{
    int npts = noPoints / 2;

    double** u = AllocateMatrix(3, npts + 1);
    double** Aiu = AllocateMatrix(3, npts + 1);
    double** L = AllocateMatrix(3, npts + 1);
    double** B = AllocateMatrix(3, npts + 1);
    double** Xpos = AllocateMatrix(3, npts + 1);
    double** Xneg = AllocateMatrix(3, npts + 1);
    double** ss1 = AllocateMatrix(3, npts + 1);
    double** ss2 = AllocateMatrix(3, npts + 1);
    double* lambda = new double[npts + 1];
    double** uAiu = AllocateMatrix(3, npts + 1);
    double** A = AllocateMatrix(3, 3);
    double** Ai = AllocateMatrix(3, 3);
    double** Aib = AllocateMatrix(3, 2);
    double** b = AllocateMatrix(3, 2);
    double** r1 = AllocateMatrix(2, 2);
    double Ao, Ax, Ay, Axx, Ayy, Axy;
    double theta;
    int i;
    int j;
    double kk;

    memset(lambda, 0, sizeof(double) * (npts + 1));

    Ao = pvec[6];
    Ax = pvec[4];
    Ay = pvec[5];
    Axx = pvec[1];
    Ayy = pvec[3];
    Axy = pvec[2];

    A[1][1] = Axx;
    A[1][2] = Axy / 2;
    A[2][1] = Axy / 2;
    A[2][2] = Ayy;
    b[1][1] = Ax;
    b[2][1] = Ay;

    // Generate normals linspace
    for (i = 1, theta = 0.0; i <= npts; i++, theta += (CV_PI / npts))
    {
        u[1][i] = cos(theta);
        u[2][i] = sin(theta);
    }

    inverse(A, Ai, 2);

    AperB(Ai, b, Aib, 2, 2, 2, 1);
    A_TperB(b, Aib, r1, 2, 1, 2, 1);
    r1[1][1] = r1[1][1] - 4 * Ao;

    AperB(Ai, u, Aiu, 2, 2, 2, npts);
    for (i = 1; i <= 2; i++)
        for (j = 1; j <= npts; j++)
            uAiu[i][j] = u[i][j] * Aiu[i][j];

    for (j = 1; j <= npts; j++)
    {
        if ((kk = (r1[1][1] / (uAiu[1][j] + uAiu[2][j]))) >= 0.0)
            lambda[j] = sqrt(kk);
        else
            lambda[j] = -1.0;
    }

    // Builds up B and L
    for (j = 1; j <= npts; j++)
        L[1][j] = L[2][j] = lambda[j];
    for (j = 1; j <= npts; j++)
    {
        B[1][j] = b[1][1];
        B[2][j] = b[2][1];
    }

    for (j = 1; j <= npts; j++)
    {
        ss1[1][j] = 0.5 * (L[1][j] * u[1][j] - B[1][j]);
        ss1[2][j] = 0.5 * (L[2][j] * u[2][j] - B[2][j]);
        ss2[1][j] = 0.5 * (-L[1][j] * u[1][j] - B[1][j]);
        ss2[2][j] = 0.5 * (-L[2][j] * u[2][j] - B[2][j]);
    }

    AperB(Ai, ss1, Xpos, 2, 2, 2, npts);
    AperB(Ai, ss2, Xneg, 2, 2, 2, npts);

    for (j = 1; j <= npts; j++)
    {
        if (lambda[j] == -1.0)
        {
            px[j - 1] = -1;
            py[j - 1] = -1;
            px[j - 1 + npts] = -1;
            py[j - 1 + npts] = -1;
        }
        else
        {
            px[j - 1] = Xpos[1][j];
            py[j - 1] = Xpos[2][j];
            px[j - 1 + npts] = Xneg[1][j];
            py[j - 1 + npts] = Xneg[2][j];
        }
    }

    DeallocateMatrix(u, 3);
    DeallocateMatrix(Aiu, 3);
    DeallocateMatrix(L, 3);
    DeallocateMatrix(B, 3);
    DeallocateMatrix(Xpos, 3);
    DeallocateMatrix(Xneg, 3);
    DeallocateMatrix(ss1, 3);
    DeallocateMatrix(ss2, 3);
    delete[] lambda;
    DeallocateMatrix(uAiu, 3);
    DeallocateMatrix(A, 3);
    DeallocateMatrix(Ai, 3);
    DeallocateMatrix(Aib, 3);
    DeallocateMatrix(b, 3);
    DeallocateMatrix(r1, 2);
}


// Tries to join the last two arcs if their end-points are very close to each other
// and if they are part of the same segment. This is useful in cases where an arc on a segment
// is broken due to a noisy patch along the arc, and the long arc is broken into two or more arcs.
// This function will join such broken arcs
//
void EdgeDrawingImpl::joinLastTwoArcs(MyArc* arcs, int& noArcs)
{
    if (noArcs < 2)
        return;

    int prev = noArcs - 2;
    int last = noArcs - 1;

    if (arcs[prev].segmentNo != arcs[last].segmentNo)
        return;
    if (arcs[prev].turn != arcs[last].turn)
        return;
    if (arcs[prev].isEllipse || arcs[last].isEllipse)
        return;

    // The radius difference between the arcs must be very small
    double minR = MIN(arcs[prev].r, arcs[last].r);
    double radiusDiffThreshold = minR * 0.25;

    double diff = fabs(arcs[prev].r - arcs[last].r);
    if (diff > radiusDiffThreshold)
        return;

    // End-point distance
    double dx = arcs[prev].ex - arcs[last].sx;
    double dy = arcs[prev].ey - arcs[last].sy;
    double d = sqrt(dx * dx + dy * dy);

    double endPointDiffThreshold = 10;
    if (d > endPointDiffThreshold)
        return;

    // Try join
    int noPixels = arcs[prev].noPixels + arcs[last].noPixels;

    double xc, yc, r, circleFitError;
    CircleFit(arcs[prev].x, arcs[prev].y, noPixels, &xc, &yc, &r, &circleFitError);

    if (circleFitError <= LONG_ARC_ERROR)
    {
        arcs[prev].noPixels = noPixels;
        arcs[prev].circleFitError = circleFitError;

        arcs[prev].xc = xc;
        arcs[prev].yc = yc;
        arcs[prev].r = r;
        arcs[prev].ex = arcs[last].ex;
        arcs[prev].ey = arcs[last].ey;

        AngleSet angles;
        angles.set(arcs[prev].sTheta, arcs[prev].eTheta);
        angles.set(arcs[last].sTheta, arcs[last].eTheta);
        angles.computeStartEndTheta(arcs[prev].sTheta, arcs[prev].eTheta);

        arcs[prev].coverRatio = ArcLength(arcs[prev].sTheta, arcs[prev].eTheta) / (CV_2PI);

        noArcs--;
    }
}


//-----------------------------------------------------------------------
// Add a new arc to arcs
//
void EdgeDrawingImpl::addArc(MyArc* arcs, int& noArcs, double xc, double yc, double r, double circleFitError, double sTheta, double eTheta, int turn, int segmentNo, int sx, int sy, int ex, int ey, double* x, double* y, int noPixels, double overlapRatio)
{
    CV_UNUSED(overlapRatio);
    arcs[noArcs].xc = xc;
    arcs[noArcs].yc = yc;
    arcs[noArcs].r = r;
    arcs[noArcs].circleFitError = circleFitError;

    arcs[noArcs].sTheta = sTheta;
    arcs[noArcs].eTheta = eTheta;
    arcs[noArcs].coverRatio = ArcLength(sTheta, eTheta) / CV_2PI;

    arcs[noArcs].turn = turn;

    arcs[noArcs].segmentNo = segmentNo;

    arcs[noArcs].isEllipse = false;

    arcs[noArcs].sx = sx;
    arcs[noArcs].sy = sy;
    arcs[noArcs].ex = ex;
    arcs[noArcs].ey = ey;

    arcs[noArcs].x = x;
    arcs[noArcs].y = y;
    arcs[noArcs].noPixels = noPixels;

    noArcs++;

    // See if you can join the last two arcs
    joinLastTwoArcs(arcs, noArcs);
}

//-------------------------------------------------------------------------
// Add an elliptic arc to the list of arcs
//
void EdgeDrawingImpl::addArc(MyArc* arcs, int& noArcs, double xc, double yc, double r, double circleFitError, double sTheta, double eTheta, int turn, int segmentNo, EllipseEquation* pEq, double ellipseFitError, int sx, int sy, int ex, int ey, double* x, double* y, int noPixels, double overlapRatio)
{
    arcs[noArcs].xc = xc;
    arcs[noArcs].yc = yc;
    arcs[noArcs].r = r;
    arcs[noArcs].circleFitError = circleFitError;

    arcs[noArcs].sTheta = sTheta;
    arcs[noArcs].eTheta = eTheta;
    arcs[noArcs].coverRatio = (double)((1.0 - overlapRatio) * noPixels) / computeEllipsePerimeter(pEq);
    arcs[noArcs].turn = turn;

    arcs[noArcs].segmentNo = segmentNo;

    arcs[noArcs].isEllipse = true;
    arcs[noArcs].eq = *pEq;
    arcs[noArcs].ellipseFitError = ellipseFitError;

    arcs[noArcs].sx = sx;
    arcs[noArcs].sy = sy;
    arcs[noArcs].ex = ex;
    arcs[noArcs].ey = ey;

    arcs[noArcs].x = x;
    arcs[noArcs].y = y;
    arcs[noArcs].noPixels = noPixels;

    noArcs++;
}

//--------------------------------------------------------------
// Given a circular arc, computes the start & end angles of the arc in radians
//
void EdgeDrawingImpl::ComputeStartAndEndAngles(double xc, double yc, double r, double* x, double* y, int len, double* psTheta, double* peTheta)
{
    double sx = x[0];
    double sy = y[0];
    double ex = x[len - 1];
    double ey = y[len - 1];
    double mx = x[len / 2];
    double my = y[len / 2];

    double d = (sx - xc) / r;
    if (d > 1.0)
        d = 1.0;
    else if (d < -1.0)
        d = -1.0;
    double theta1 = acos(d);

    double sTheta;
    if (sx >= xc)
    {
        if (sy >= yc)
        {
            // I. quadrant
            sTheta = theta1;
        }
        else
        {
            // IV. quadrant
            sTheta = CV_2PI - theta1;
        }
    }
    else
    {
        if (sy >= yc)
        {
            // II. quadrant
            sTheta = theta1;
        }
        else
        {
            // III. quadrant
            sTheta = CV_2PI - theta1;
        }
    }

    d = (ex - xc) / r;
    if (d > 1.0)
        d = 1.0;
    else if (d < -1.0)
        d = -1.0;
    theta1 = acos(d);

    double eTheta;
    if (ex >= xc)
    {
        if (ey >= yc)
        {
            // I. quadrant
            eTheta = theta1;
        }
        else
        {
            // IV. quadrant
            eTheta = CV_2PI - theta1;
        }
    }
    else
    {
        if (ey >= yc)
        {
            // II. quadrant
            eTheta = theta1;
        }
        else
        {
            // III. quadrant
            eTheta = CV_2PI - theta1;
        }
    }

    // Determine whether the arc is clockwise (CW) or counter-clockwise (CCW)
    double circumference = CV_2PI * r;
    double ratio = len / circumference;

    if (ratio <= 0.25 || ratio >= 0.75)
    {
        double angle1, angle2;

        if (eTheta > sTheta)
        {
            angle1 = eTheta - sTheta;
            angle2 = CV_2PI - eTheta + sTheta;
        }
        else
        {
            angle1 = sTheta - eTheta;
            angle2 = CV_2PI - sTheta + eTheta;
        }

        angle1 = angle1 / CV_2PI;
        angle2 = angle2 / CV_2PI;

        double diff1 = fabs(ratio - angle1);
        double diff2 = fabs(ratio - angle2);

        if (diff1 < diff2)
        {
            // angle1 is correct
            if (eTheta > sTheta)
            {
                ;
            }
            else
            {
                double tmp = sTheta;
                sTheta = eTheta;
                eTheta = tmp;
            }
        }
        else
        {
            // angle2 is correct
            if (eTheta > sTheta)
            {
                double tmp = sTheta;
                sTheta = eTheta;
                eTheta = tmp;
            }
        }
    }
    else
    {
        double v1x = mx - sx;
        double v1y = my - sy;
        double v2x = ex - mx;
        double v2y = ey - my;

        // cross product
        double cross = v1x * v2y - v1y * v2x;
        if (cross < 0)
        {
            // swap sTheta & eTheta
            double tmp = sTheta;
            sTheta = eTheta;
            eTheta = tmp;
        }
    }

    double diff = fabs(sTheta - eTheta);
    if (diff < (CV_2PI / 120))
    {
        sTheta = 0;
        eTheta = 6.26;   // 359 degrees
    }

    // Round the start & etheta to 0 if very close to 6.28 or 0
    if (sTheta >= 6.26)
        sTheta = 0;
    if (eTheta < 1.0 / CV_2PI)
        eTheta = 6.28;  // if less than 1 degrees, then round to 6.28

    *psTheta = sTheta;
    *peTheta = eTheta;
}

void EdgeDrawingImpl::sortArc(MyArc* arcs, int noArcs)
{
    for (int i = 0; i < noArcs - 1; i++)
    {
        int max = i;
        for (int j = i + 1; j < noArcs; j++)
        {
            if (arcs[j].coverRatio > arcs[max].coverRatio)
                max = j;
        }

        if (max != i)
        {
            MyArc t = arcs[i];
            arcs[i] = arcs[max];
            arcs[max] = t;
        }
    }
}


//---------------------------------------------------------------------
// Fits a circle to a given set of points. There must be at least 2 points
// The circle equation is of the form: (x-xc)^2 + (y-yc)^2 = r^2
// Returns true if there is a fit, false in case no circles can be fit
//
bool EdgeDrawingImpl::CircleFit(double* x, double* y, int N, double* pxc, double* pyc, double* pr, double* pe)
{
    *pe = 1e20;
    if (N < 3)
        return false;

    double xAvg = 0;
    double yAvg = 0;

    for (int i = 0; i < N; i++)
    {
        xAvg += x[i];
        yAvg += y[i];
    }

    xAvg /= N;
    yAvg /= N;

    double Suu = 0;
    double Suv = 0;
    double Svv = 0;
    double Suuu = 0;
    double Suvv = 0;
    double Svvv = 0;
    double Svuu = 0;

    for (int i = 0; i < N; i++)
    {
        double u = x[i] - xAvg;
        double v = y[i] - yAvg;

        Suu += u * u;
        Suv += u * v;
        Svv += v * v;
        Suuu += u * u * u;
        Suvv += u * v * v;
        Svvv += v * v * v;
        Svuu += v * u * u;
    }

    // Now, we solve for the following linear system of equations
    // Av = b, where v = (uc, vc) is the center of the circle
    //
    // |N    Suv| |uc| = |b1|
    // |Suv  Svv| |vc| = |b2|
    //
    // where b1 = 0.5*(Suuu+Suvv) and b2 = 0.5*(Svvv+Svuu)
    //
    double detA = Suu * Svv - Suv * Suv;
    if (detA == 0)
        return false;

    double b1 = 0.5 * (Suuu + Suvv);
    double b2 = 0.5 * (Svvv + Svuu);

    double uc = (Svv * b1 - Suv * b2) / detA;
    double vc = (Suu * b2 - Suv * b1) / detA;

    double R = sqrt(uc * uc + vc * vc + (Suu + Svv) / N);

    *pxc = uc + xAvg;
    *pyc = vc + yAvg;

    // Compute mean square error
    double error = 0;
    for (int i = 0; i < N; i++)
    {
        double dx = x[i] - *pxc;
        double dy = y[i] - *pyc;
        double d = sqrt(dx * dx + dy * dy) - R;
        error += d * d;
    }

    *pr = R;
    *pe = sqrt(error / N);

    return true;
}


//------------------------------------------------------------------------------------
// Computes the points making up a circle
//
void EdgeDrawingImpl::ComputeCirclePoints(double xc, double yc, double r, double* px, double* py, int* noPoints)
{
    int len = (int)(CV_2PI * r + 0.5);
    double angleInc = CV_2PI / len;
    double angle = 0;

    int count = 0;

    while (angle < CV_2PI)
    {
        int x = (int)(cos(angle) * r + xc + 0.5);
        int y = (int)(sin(angle) * r + yc + 0.5);

        angle += angleInc;

        px[count] = x;
        py[count] = y;
        count++;
    }

    *noPoints = count;
}

bool EdgeDrawingImpl::EllipseFit(double* x, double* y, int noPoints, EllipseEquation* pResult, int mode)
{
    double** D = AllocateMatrix(noPoints + 1, 7);
    double** S = AllocateMatrix(7, 7);
    double** Const = AllocateMatrix(7, 7);
    double** temp = AllocateMatrix(7, 7);
    double** L = AllocateMatrix(7, 7);
    double** C = AllocateMatrix(7, 7);

    double** invL = AllocateMatrix(7, 7);
    double* d = new double[7];
    double** V = AllocateMatrix(7, 7);
    double** sol = AllocateMatrix(7, 7);
    double tx, ty;

    memset(d, 0, sizeof(double) * 7);

    switch (mode)
    {
    case (FPF):
        Const[1][3] = -2;
        Const[2][2] = 1;
        Const[3][1] = -2;
        break;
    case (BOOKSTEIN):
        Const[1][1] = 2;
        Const[2][2] = 1;
        Const[3][3] = 2;
    }

    if (noPoints < 6)
        return false;

    // Now first fill design matrix
    for (int i = 1; i <= noPoints; i++)
    {
        tx = x[i - 1];
        ty = y[i - 1];

        D[i][1] = tx * tx;
        D[i][2] = tx * ty;
        D[i][3] = ty * ty;
        D[i][4] = tx;
        D[i][5] = ty;
        D[i][6] = 1.0;
    }

    // Now compute scatter matrix  S
    A_TperB(D, D, S, noPoints, 6, noPoints, 6);

    choldc(S, 6, L);

    inverse(L, invL, 6);

    AperB_T(Const, invL, temp, 6, 6, 6, 6);
    AperB(invL, temp, C, 6, 6, 6, 6);

    jacobi(C, 6, d, V);

    A_TperB(invL, V, sol, 6, 6, 6, 6);

    // Now normalize them
    for (int j = 1; j <= 6; j++)  /* Scan columns */
    {
        double mod = 0.0;
        for (int i = 1; i <= 6; i++)
            mod += sol[i][j] * sol[i][j];
        for (int i = 1; i <= 6; i++)
            sol[i][j] /= sqrt(mod);
    }

    double zero = 10e-20;
    double minev = 10e+20;
    int solind = 0;
    int i;
    switch (mode)
    {
    case (BOOKSTEIN):  // smallest eigenvalue
        for (i = 1; i <= 6; i++)
            if (d[i] < minev && fabs(d[i]) > zero)
                solind = i;
        break;
    case (FPF):
        for (i = 1; i <= 6; i++)
            if (d[i] < 0 && fabs(d[i]) > zero)
                solind = i;
    }

    bool valid = true;
    if (solind == 0)
        valid = false;

    if (valid)
    {
        // Now fetch the right solution
        for (int j = 1; j <= 6; j++)
        {
            pResult->coeff[j] = sol[j][solind];
        }
    }

    DeallocateMatrix(D, noPoints + 1);
    DeallocateMatrix(S, 7);
    DeallocateMatrix(Const, 7);
    DeallocateMatrix(temp, 7);
    DeallocateMatrix(L, 7);
    DeallocateMatrix(C, 7);
    DeallocateMatrix(invL, 7);
    delete[] d;
    DeallocateMatrix(V, 7);
    DeallocateMatrix(sol, 7);

    if (valid)
    {
        int len = (int)computeEllipsePerimeter(pResult);
        if (len <= 0 || len > 50000)
            valid = false;
    }

    return valid;
}

double** EdgeDrawingImpl::AllocateMatrix(int noRows, int noColumns)
{
    double** m = new double* [noRows];

    for (int i = 0; i < noRows; i++)
    {
        m[i] = new double[noColumns];
        memset(m[i], 0, sizeof(double) * noColumns);
    }

    return m;
}

void EdgeDrawingImpl::A_TperB(double** A_, double** B_, double** _res, int _righA, int _colA, int _righB, int _colB)
{
    CV_UNUSED(_righB);
    int p, q, l;
    for (p = 1; p <= _colA; p++)
        for (q = 1; q <= _colB; q++)
        {
            _res[p][q] = 0.0;
            for (l = 1; l <= _righA; l++)
                _res[p][q] = _res[p][q] + A_[l][p] * B_[l][q];
        }
}

//-----------------------------------------------------------
// Perform the Cholesky decomposition
// Return the lower triangular L  such that L*L'=A
//
void EdgeDrawingImpl::choldc(double** a, int n, double** l)
{
    int i, j, k;
    double sum;
    double* p = new double[n + 1];
    memset(p, 0, sizeof(double) * (n + 1));

    for (i = 1; i <= n; i++)
    {
        for (j = i; j <= n; j++)
        {
            for (sum = a[i][j], k = i - 1; k >= 1; k--)
                sum -= a[i][k] * a[j][k];
            if (i == j)
            {
                if (sum <= 0.0)
                {
                }
                else
                    p[i] = sqrt(sum);
            }
            else
            {
                a[j][i] = sum / p[i];
            }
        }
    }

    for (i = 1; i <= n; i++)
    {
        for (j = i; j <= n; j++)
        {
            if (i == j)
                l[i][i] = p[i];
            else
            {
                l[j][i] = a[j][i];
                l[i][j] = 0.0;
            }
        }
    }

    delete[] p;
}

int EdgeDrawingImpl::inverse(double** TB, double** InvB, int N)
{
    int k, i, j, p, q;
    double mult;
    double D, temp;
    double maxpivot;
    int npivot;
    double** B = AllocateMatrix(N + 1, N + 2);
    double** A = AllocateMatrix(N + 1, 2 * N + 2);
    double** C = AllocateMatrix(N + 1, N + 1);
    double eps = 10e-20;

    for (k = 1; k <= N; k++)
        for (j = 1; j <= N; j++)
            B[k][j] = TB[k][j];

    for (k = 1; k <= N; k++)
    {
        for (j = 1; j <= N + 1; j++)
            A[k][j] = B[k][j];
        for (j = N + 2; j <= 2 * N + 1; j++)
            A[k][j] = (double)0;
        A[k][k - 1 + N + 2] = (double)1;
    }
    for (k = 1; k <= N; k++)
    {
        maxpivot = fabs((double)A[k][k]);
        npivot = k;
        for (i = k; i <= N; i++)
            if (maxpivot < fabs((double)A[i][k]))
            {
                maxpivot = fabs((double)A[i][k]);
                npivot = i;
            }
        if (maxpivot >= eps)
        {
            if (npivot != k)
                for (j = k; j <= 2 * N + 1; j++)
                {
                    temp = A[npivot][j];
                    A[npivot][j] = A[k][j];
                    A[k][j] = temp;
                };
            D = A[k][k];
            for (j = 2 * N + 1; j >= k; j--)
                A[k][j] = A[k][j] / D;
            for (i = 1; i <= N; i++)
            {
                if (i != k)
                {
                    mult = A[i][k];
                    for (j = 2 * N + 1; j >= k; j--)
                        A[i][j] = A[i][j] - mult * A[k][j];
                }
            }
        }
        else
        {
            DeallocateMatrix(B, N + 1);
            DeallocateMatrix(A, N + 1);
            DeallocateMatrix(C, N + 1);

            return (-1);
        }
    }

    for (k = 1, p = 1; k <= N; k++, p++)
        for (j = N + 2, q = 1; j <= 2 * N + 1; j++, q++)
            InvB[p][q] = A[k][j];

    DeallocateMatrix(B, N + 1);
    DeallocateMatrix(A, N + 1);
    DeallocateMatrix(C, N + 1);

    return (0);
}

void EdgeDrawingImpl::DeallocateMatrix(double** m, int noRows)
{
    for (int i = 0; i < noRows; i++)
        delete[] m[i];
    delete[] m;
}

void EdgeDrawingImpl::AperB_T(double** A_, double** B_, double** _res, int _righA, int _colA, int _righB, int _colB)
{
    CV_UNUSED(_righB);
    int p, q, l;
    for (p = 1; p <= _colA; p++)
        for (q = 1; q <= _colB; q++)
        {
            _res[p][q] = 0.0;
            for (l = 1; l <= _righA; l++)
                _res[p][q] = _res[p][q] + A_[p][l] * B_[q][l];
        }
}

void EdgeDrawingImpl::AperB(double** A_, double** B_, double** _res, int _righA, int _colA, int _righB, int _colB)
{
    CV_UNUSED(_righB);
    int p, q, l;
    for (p = 1; p <= _righA; p++)
        for (q = 1; q <= _colB; q++)
        {
            _res[p][q] = 0.0;
            for (l = 1; l <= _colA; l++)
                _res[p][q] = _res[p][q] + A_[p][l] * B_[l][q];
        }
}

void EdgeDrawingImpl::jacobi(double** a, int n, double d[], double** v)
{
    int j, iq, ip, i;
    double tresh, theta, tau, t, sm, s, h, g, c;

    double* b = new double[n + 1];
    double* z = new double[n + 1];
    memset(b, 0, sizeof(double) * (n + 1));
    memset(z, 0, sizeof(double) * (n + 1));

    for (ip = 1; ip <= n; ip++)
    {
        for (iq = 1; iq <= n; iq++)
            v[ip][iq] = 0.0;
        v[ip][ip] = 1.0;
    }
    for (ip = 1; ip <= n; ip++)
    {
        b[ip] = d[ip] = a[ip][ip];
        z[ip] = 0.0;
    }
    for (i = 1; i <= 50; i++)
    {
        sm = 0.0;
        for (ip = 1; ip <= n - 1; ip++)
        {
            for (iq = ip + 1; iq <= n; iq++)
                sm += fabs(a[ip][iq]);
        }
        if (sm == 0.0)
        {
            delete[] b;
            delete[] z;
            return;
        }
        if (i < 4)
            tresh = 0.2 * sm / (n * n);
        else
            tresh = 0.0;
        for (ip = 1; ip <= n - 1; ip++)
        {
            for (iq = ip + 1; iq <= n; iq++)
            {
                g = 100.0 * fabs(a[ip][iq]);

                if (i > 4 && g == 0.0)
                    a[ip][iq] = 0.0;
                else if (fabs(a[ip][iq]) > tresh)
                {
                    h = d[iq] - d[ip];
                    if (g == 0.0)
                        t = (a[ip][iq]) / h;
                    else
                    {
                        theta = 0.5 * h / (a[ip][iq]);
                        t = 1.0 / (fabs(theta) + sqrt(1.0 + theta * theta));
                        if (theta < 0.0)
                            t = -t;
                    }
                    c = 1.0 / sqrt(1 + t * t);
                    s = t * c;
                    tau = s / (1.0 + c);
                    h = t * a[ip][iq];
                    z[ip] -= h;
                    z[iq] += h;
                    d[ip] -= h;
                    d[iq] += h;
                    a[ip][iq] = 0.0;
                    for (j = 1; j <= ip - 1; j++)
                    {
                        ROTATE(a, j, ip, j, iq, tau, s);
                    }
                    for (j = ip + 1; j <= iq - 1; j++)
                    {
                        ROTATE(a, ip, j, j, iq, tau, s);
                    }
                    for (j = iq + 1; j <= n; j++)
                    {
                        ROTATE(a, ip, j, iq, j, tau, s);
                    }
                    for (j = 1; j <= n; j++)
                    {
                        ROTATE(v, j, ip, j, iq, tau, s);
                    }
                }
            }
        }

        for (ip = 1; ip <= n; ip++)
        {
            b[ip] += z[ip];
            d[ip] = b[ip];
            z[ip] = 0.0;
        }
    }
    delete[] b;
    delete[] z;
}

void EdgeDrawingImpl::ROTATE(double** a, int i, int j, int k, int l, double tau, double s)
{
    double g, h;
    g = a[i][j];
    h = a[k][l];
    a[i][j] = g - s * (h + g * tau);
    a[k][l] = h + s * (g - h * tau);
}

} // namespace cv
} // namespace ximgproc
