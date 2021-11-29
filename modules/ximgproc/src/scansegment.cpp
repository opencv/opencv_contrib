////////////////////////////////////////////////////////////////////////////////////////
//
// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2021, Dr Seng Cheong Loke (lokesengcheong@gmail.com)
//
//

#include "precomp.hpp"

#include <numeric>
#include <atomic>

namespace cv {
namespace ximgproc {

ScanSegment::~ScanSegment()
{
    // nothing
}

class ScanSegmentImpl CV_FINAL : public ScanSegment
{
#define UNKNOWN 0
#define BORDER -1
#define UNCLASSIFIED -2
#define NONE -3

public:

    ScanSegmentImpl(int image_width, int image_height, int num_superpixels, int slices, bool merge_small);

    virtual ~ScanSegmentImpl();

    virtual int getNumberOfSuperpixels() CV_OVERRIDE { return clusterCount; }

    virtual void iterate(InputArray img) CV_OVERRIDE;

    virtual void getLabels(OutputArray labels_out) CV_OVERRIDE;

    virtual void getLabelContourMask(OutputArray image, bool thick_line = false) CV_OVERRIDE;

private:
    static const int neighbourCount = 8;    // number of pixel neighbours
    static const int smallClustersDiv = 10000;  // divide total pixels by this to give smallClusters
    const float tolerance100 = 10.0f;       // colour tolerance for image size of 100x100px

    int processthreads;                     // concurrent threads for parallel processing
    int width, height;                      // image size
    int superpixels;                        // number of superpixels
    bool merge;                             // merge small superpixels
    int indexSize;                          // size of label mat vector
    int clusterSize;                        // max size of clusters
    int clusterCount;                       // number of superpixels from the most recent iterate
    float adjTolerance;                     // adjusted colour tolerance

    int horzDiv, vertDiv;                   // number of horizontal and vertical segments
    float horzLength, vertLength;           // length of each segment
    int effectivethreads;                   // effective number of concurrent threads
    int smallClusters;                      // clusters below this pixel count are considered small for merging

    cv::AutoBuffer<cv::Rect> seedRects;     // autobuffer of seed rectangles
    cv::AutoBuffer<cv::Rect> seedRectsExt;  // autobuffer of extended seed rectangles
    cv::AutoBuffer<cv::Rect> offsetRects;   // autobuffer of offset rectangles
    cv::Point neighbourLoc[8] = { cv::Point(-1, -1), cv::Point(0, -1), cv::Point(1, -1), cv::Point(-1, 0), cv::Point(1, 0), cv::Point(-1, 1), cv::Point(0, 1), cv::Point(1, 1) };                // neighbour locations

    std::vector<int> indexNeighbourVec;     // indices for parallel processing
    std::vector<std::pair<int, int>> indexProcessVec;

    cv::AutoBuffer<int> labelsBuffer;       // label autobuffer
    cv::AutoBuffer<int> clusterBuffer;      // cluster autobuffer
    cv::AutoBuffer<uchar> pixelBuffer;      // pixel autobuffer
    std::vector<cv::AutoBuffer<int>> offsetVec; // vector of offset autobuffers
    cv::Vec3b* labBuffer;                   // lab buffer
    int neighbourLocBuffer[neighbourCount]; // neighbour locations

    std::atomic<int> clusterIndex, clusterID;  // atomic indices

    cv::Mat src, labelsMat;                 // mats

    struct WSNode
    {
        int next;
        int mask_ofs;
        int img_ofs;
    };

    // Queue for WSNodes
    struct WSQueue
    {
        WSQueue() { first = last = 0; }
        int first, last;
    };

    void OP1(int v);
    void OP2(std::pair<int, int> const& p);
    void OP3(int v);
    void OP4(std::pair<int, int> const& p);
    void expandCluster(int* offsetBuffer, const cv::Point& point);
    void calculateCluster(int* offsetBuffer, int* offsetEnd, int pointIndex, int currentClusterID);
    static int allocWSNodes(std::vector<WSNode>& storage);
    static void watershedEx(const cv::Mat& src, cv::Mat& dst);
};

CV_EXPORTS Ptr<ScanSegment> createScanSegment(int image_width, int image_height, int num_superpixels, int slices, bool merge_small)
{
    return makePtr<ScanSegmentImpl>(image_width, image_height, num_superpixels, slices, merge_small);
}

ScanSegmentImpl::ScanSegmentImpl(int image_width, int image_height, int num_superpixels, int slices, bool merge_small)
{
    // set the number of process threads
    processthreads = (slices > 0) ? slices : cv::getNumThreads();

    width = image_width;
    height = image_height;
    superpixels = num_superpixels;
    merge = merge_small;
    indexSize = height * width;
    clusterSize = cvRound(1.1f * (float)(width * height) / (float)superpixels);
    clusterCount = 0;
    labelsMat = cv::Mat(height, width, CV_32SC1);

    // divide bounds area into uniformly distributed rectangular segments
    int shortCount = cvFloor(sqrtf((float)processthreads));
    int longCount = processthreads / shortCount;
    horzDiv = width > height ? longCount : shortCount;
    vertDiv = width > height ? shortCount : longCount;
    horzLength = (float)width / (float)horzDiv;
    vertLength = (float)height / (float)vertDiv;
    effectivethreads = horzDiv * vertDiv;
    smallClusters = 0;

    // get array of seed rects
    seedRects = cv::AutoBuffer<cv::Rect>(horzDiv * vertDiv);
    seedRectsExt = cv::AutoBuffer<cv::Rect>(horzDiv * vertDiv);
    offsetRects = cv::AutoBuffer<cv::Rect>(horzDiv * vertDiv);
    for (int y = 0; y < vertDiv; y++) {
        for (int x = 0; x < horzDiv; x++) {
            int xStart = cvFloor((float)x * horzLength);
            int yStart = cvFloor((float)y * vertLength);
            cv::Rect seedRect = cv::Rect(xStart, yStart, (int)(x == horzDiv - 1 ? width - xStart : horzLength), (int)(y == vertDiv - 1 ? height - yStart : vertLength));

            int bnd_l = seedRect.x;
            int bnd_t = seedRect.y;
            int bnd_r = seedRect.x + seedRect.width - 1;
            int bnd_b = seedRect.y + seedRect.height - 1;
            if (bnd_l > 0) {
                bnd_l -= 1;
            }
            if (bnd_t > 0) {
                bnd_t -= 1;
            }
            if (bnd_r < width - 1) {
                bnd_r += 1;
            }
            if (bnd_b < height - 1) {
                bnd_b += 1;
            }

            seedRects.data()[(y * horzDiv) + x] = seedRect;
            seedRectsExt.data()[(y * horzDiv) + x] = cv::Rect(bnd_l, bnd_t, bnd_r - bnd_l + 1, bnd_b - bnd_t + 1);
            offsetRects.data()[(y * horzDiv) + x] = cv::Rect(seedRect.x - bnd_l, seedRect.y - bnd_t, seedRect.width, seedRect.height);
        }
    }

    // get adjusted tolerance = (100 / average length (horz/vert)) x sqrt(3) [ie. euclidean lab colour distance sqrt(l2 + a2 + b2)] x tolerance100
    adjTolerance = (200.0f / (width + height)) * sqrtf(3) * tolerance100;
    adjTolerance = adjTolerance * adjTolerance;

    // create neighbour vector
    indexNeighbourVec = std::vector<int>(effectivethreads);
    std::iota(indexNeighbourVec.begin(), indexNeighbourVec.end(), 0);

    // create process vector
    indexProcessVec = std::vector<std::pair<int, int>>(processthreads);
    int processDiv = indexSize / processthreads;
    int processCurrent = 0;
    for (int i = 0; i < processthreads - 1; i++) {
        indexProcessVec[i] = std::make_pair(processCurrent, processCurrent + processDiv);
        processCurrent += processDiv;
    }
    indexProcessVec[processthreads - 1] = std::make_pair(processCurrent, indexSize);

    // create buffers and initialise
    labelsBuffer = cv::AutoBuffer<int>(indexSize);
    clusterBuffer = cv::AutoBuffer<int>(indexSize);
    pixelBuffer = cv::AutoBuffer<uchar>(indexSize);
    offsetVec = std::vector<cv::AutoBuffer<int>>(effectivethreads);
    int offsetSize = (clusterSize + 1) * sizeof(int);
    for (int i = 0; i < effectivethreads; i++) {
        offsetVec[i] = cv::AutoBuffer<int>(offsetSize);
    }
    for (int i = 0; i < neighbourCount; i++) {
        neighbourLocBuffer[i] = (neighbourLoc[i].y * width) + neighbourLoc[i].x;
    }
}

ScanSegmentImpl::~ScanSegmentImpl()
{
    // clean up
    if (!src.empty()) {
        src.release();
    }
    if (!labelsMat.empty()) {
        labelsMat.release();
    }
}

void ScanSegmentImpl::iterate(InputArray img)
{
    if (img.isMat())
    {
        // get Mat
        src = img.getMat();

        // image should be valid
        CV_Assert(!src.empty());
    }
    else if (img.isMatVector())
    {
        std::vector<cv::Mat> vec;

        // get vector Mat
        img.getMatVector(vec);

        // array should be valid
        CV_Assert(!vec.empty());

        // merge into Mat
        cv::merge(vec, src);
    }
    else
        CV_Error(Error::StsInternal, "Invalid InputArray.");

    int depth = src.depth();

    CV_Assert(src.size().width == width && src.size().height == height);
    CV_Assert(depth == CV_8U);
    CV_Assert(src.channels() == 3);

    clusterCount = 0;
    clusterIndex.store(0);
    clusterID.store(1);

    smallClusters = indexSize / smallClustersDiv;

    // set labels to NONE
    labelsMat.setTo(NONE);

    // set labels buffer to UNCLASSIFIED
    std::fill(labelsBuffer.data(), labelsBuffer.data() + indexSize, UNCLASSIFIED);

    // apply light blur
    cv::medianBlur(src, src, 3);

    // start at the center of the rect, then run through the remainder
    labBuffer = reinterpret_cast<cv::Vec3b*>(src.data);
    cv::parallel_for_(Range(0, (int)indexNeighbourVec.size()), [&](const Range& range) {
        for (int i = range.start; i < range.end; i++) {
            OP1(i);
        }
    });

    if (merge) {
        // get cutoff size for clusters
        std::vector<std::pair<int, int>> countVec;
        int clusterIndexSize = clusterIndex.load();
        countVec.reserve(clusterIndexSize / 2);
        for (int i = 1; i < clusterIndexSize; i += 2) {
            int count = clusterBuffer.data()[i];
            if (count >= smallClusters) {
                int currentID = clusterBuffer.data()[i - 1];
                countVec.push_back(std::make_pair(currentID, count));
            }
        }

        // sort descending
        std::sort(countVec.begin(), countVec.end(), [](const std::pair<int, int>& left, const std::pair<int, int>& right) {
            return left.second > right.second;
        });

        int countSize = (int)countVec.size();
        int cutoff = MAX(smallClusters, countVec[MIN(countSize - 1, superpixels - 1)].second);
        clusterCount = (int)std::count_if(countVec.begin(), countVec.end(), [&cutoff](std::pair<int, int> p) {return p.second > cutoff; });

        // change labels to 1 -> clusterCount, 0 = UNKNOWN, reuse clusterbuffer
        std::fill_n(clusterBuffer.data(), indexSize, UNKNOWN);
        int countLimit = cutoff == -1 ? (int)countVec.size() : clusterCount;
        for (int i = 0; i < countLimit; i++) {
            clusterBuffer.data()[countVec[i].first] = i + 1;
        }

        parallel_for_(Range(0, (int)indexProcessVec.size()), [&](const Range& range) {
            for (int i = range.start; i < range.end; i++) {
                OP2(indexProcessVec[i]);
            }
        });

        // make copy of labels buffer
        memcpy(labelsMat.data, labelsBuffer.data(), indexSize * sizeof(int));

        // run watershed
        cv::parallel_for_(Range(0, (int)indexNeighbourVec.size()), [&](const Range& range) {
            for (int i = range.start; i < range.end; i++) {
                OP3(i);
            }
        });

        // copy back to labels mat
        parallel_for_(Range(0, (int)indexProcessVec.size()), [&](const Range& range) {
            for (int i = range.start; i < range.end; i++) {
                OP4(indexProcessVec[i]);
            }
        });
    }
    else
    {
        memcpy(labelsMat.data, labelsBuffer.data(), indexSize * sizeof(int));
    }

    src.release();
}

void ScanSegmentImpl::OP1(int v)
{
    cv::Rect seedRect = seedRects.data()[v];
    for (int y = seedRect.y; y < seedRect.y + seedRect.height; y++) {
        for (int x = seedRect.x; x < seedRect.x + seedRect.width; x++) {
            expandCluster(offsetVec[v].data(), cv::Point(x, y));
        }
    }
}

void ScanSegmentImpl::OP2(std::pair<int, int> const& p)
{
    for (int i = p.first; i < p.second; i++) {
        labelsBuffer.data()[i] = clusterBuffer.data()[labelsBuffer.data()[i]];
        if (labelsBuffer.data()[i] == UNKNOWN) {
            pixelBuffer.data()[i] = 255;
        }
        else {
            pixelBuffer.data()[i] = 0;
        }
    }
}

void ScanSegmentImpl::OP3(int v)
{
    cv::Rect seedRectExt = seedRectsExt.data()[v];
    cv::Mat seedLabels = labelsMat(seedRectExt).clone();
    watershedEx(src(seedRectExt), seedLabels);
    seedLabels(offsetRects.data()[v]).copyTo(labelsMat(seedRects.data()[v]));
    seedLabels.release();
}

void ScanSegmentImpl::OP4(std::pair<int, int> const& p)
{
    for (int i = p.first; i < p.second; i++) {
        if (pixelBuffer.data()[i] == 0) {
            ((int*)labelsMat.data)[i] = labelsBuffer.data()[i] - 1;
        }
        else {
            ((int*)labelsMat.data)[i] -= 1;
        }
    }
}

// expand clusters from a point
void ScanSegmentImpl::expandCluster(int* offsetBuffer, const cv::Point& point)
{
    int pointIndex = (point.y * width) + point.x;
    if (labelsBuffer.data()[pointIndex] == UNCLASSIFIED) {
        int offsetStart = 0;
        int offsetEnd = 0;
        int currentClusterID = clusterID.fetch_add(1);

        calculateCluster(offsetBuffer, &offsetEnd, pointIndex, currentClusterID);

        if (offsetStart == offsetEnd) {
            labelsBuffer.data()[pointIndex] = UNKNOWN;
        }
        else {
            // set cluster id and get core point index
            labelsBuffer.data()[pointIndex] = currentClusterID;

            while (offsetStart < offsetEnd) {
                int intoffset2 = *(offsetBuffer + offsetStart);
                offsetStart++;
                calculateCluster(offsetBuffer, &offsetEnd, intoffset2, currentClusterID);
            }

            // add origin point
            offsetBuffer[offsetEnd] = pointIndex;
            offsetEnd++;

            // store to buffer
            int currentClusterIndex = clusterIndex.fetch_add(2);
            clusterBuffer.data()[currentClusterIndex] = currentClusterID;
            clusterBuffer.data()[currentClusterIndex + 1] = offsetEnd;
        }
    }
}

void ScanSegmentImpl::calculateCluster(int* offsetBuffer, int* offsetEnd, int pointIndex, int currentClusterID)
{
    for (int i = 0; i < neighbourCount; i++) {
        if (*offsetEnd < clusterSize) {
            int intoffset2 = pointIndex + neighbourLocBuffer[i];
            if (intoffset2 >= 0 && intoffset2 < indexSize && labelsBuffer.data()[intoffset2] == UNCLASSIFIED) {
                int diff1 = (int)labBuffer[pointIndex][0] - (int)labBuffer[intoffset2][0];
                int diff2 = (int)labBuffer[pointIndex][1] - (int)labBuffer[intoffset2][1];
                int diff3 = (int)labBuffer[pointIndex][2] - (int)labBuffer[intoffset2][2];

                if ((diff1 * diff1) + (diff2 * diff2) + (diff3 * diff3) <= (int)adjTolerance) {
                    labelsBuffer.data()[intoffset2] = currentClusterID;
                    offsetBuffer[*offsetEnd] = intoffset2;
                    (*offsetEnd)++;
                }
            }
        }
        else { break; }
    }
}

int ScanSegmentImpl::allocWSNodes(std::vector<ScanSegmentImpl::WSNode>& storage)
{
    int sz = (int)storage.size();
    int newsz = MAX(128, sz * 3 / 2);

    storage.resize(newsz);
    if (sz == 0)
    {
        storage[0].next = 0;
        sz = 1;
    }
    for (int i = sz; i < newsz - 1; i++)
        storage[i].next = i + 1;
    storage[newsz - 1].next = 0;
    return sz;
}

//the modified version of watershed algorithm from OpenCV
void ScanSegmentImpl::watershedEx(const cv::Mat& src, cv::Mat& dst)
{
    // https://github.com/Seaball/watershed_with_mask

    // Labels for pixels
    const int IN_QUEUE = -2; // Pixel visited
    // possible bit values = 2^8
    const int NQ = 256;

    cv::Size size = src.size();
    int channel = 3;
    // Vector of every created node
    std::vector<WSNode> storage;
    int free_node = 0, node;
    // Priority queue of queues of nodes
    // from high priority (0) to low priority (255)
    WSQueue q[NQ];
    // Non-empty queue with highest priority
    int active_queue;
    int i, j;
    // Color differences
    int db, dg, dr;
    int subs_tab[513];

    // MAX(a,b) = b + MAX(a-b,0)
#define ws_max(a,b) ((b) + subs_tab[(a)-(b)+NQ])
// MIN(a,b) = a - MAX(a-b,0)
#define ws_min(a,b) ((a) - subs_tab[(a)-(b)+NQ])

    // Create a new node with offsets mofs and iofs in queue idx
#define ws_push(idx,mofs,iofs)          \
{                                       \
    if (!free_node)                     \
        free_node = allocWSNodes(storage); \
    node = free_node;                   \
    free_node = storage[free_node].next; \
    storage[node].next = 0;             \
    storage[node].mask_ofs = mofs;      \
    storage[node].img_ofs = iofs;       \
    if (q[idx].last)                   \
        storage[q[idx].last].next = node; \
    else                                \
        q[idx].first = node;            \
    q[idx].last = node;                 \
}

    // Get next node from queue idx
#define ws_pop(idx,mofs,iofs)           \
{                                       \
    node = q[idx].first;                \
    q[idx].first = storage[node].next;  \
    if (!storage[node].next)           \
        q[idx].last = 0;                \
    storage[node].next = free_node;     \
    free_node = node;                   \
    mofs = storage[node].mask_ofs;      \
    iofs = storage[node].img_ofs;       \
}

// Get highest absolute channel difference in diff
#define c_diff(ptr1,ptr2,diff)           \
{                                        \
    db = std::abs((ptr1)[0] - (ptr2)[0]); \
    dg = std::abs((ptr1)[1] - (ptr2)[1]); \
    dr = std::abs((ptr1)[2] - (ptr2)[2]); \
    diff = ws_max(db, dg);                \
    diff = ws_max(diff, dr);              \
    CV_Assert(0 <= diff && diff <= 255);  \
}

    CV_Assert(src.type() == CV_8UC3 && dst.type() == CV_32SC1);
    CV_Assert(src.size() == dst.size());

    // Current pixel in input image
    const uchar* img = src.ptr();
    // Step size to next row in input image
    int istep = int(src.step / sizeof(img[0]));

    // Current pixel in mask image
    int* mask = dst.ptr<int>();
    // Step size to next row in mask image
    int mstep = int(dst.step / sizeof(mask[0]));

    for (i = 0; i < 256; i++)
        subs_tab[i] = 0;
    for (i = 256; i <= 512; i++)
        subs_tab[i] = i - 256;

    //for (j = 0; j < size.width; j++)
    //mask[j] = mask[j + mstep*(size.height - 1)] = 0;

    // initial phase: put all the neighbor pixels of each marker to the ordered queue -
    // determine the initial boundaries of the basins
    for (i = 1; i < size.height - 1; i++) {
        img += istep; mask += mstep;
        mask[0] = mask[size.width - 1] = 0; // boundary pixels

        for (j = 1; j < size.width - 1; j++) {
            int* m = mask + j;
            if (m[0] < 0)
                m[0] = 0;
            if (m[0] == 0 && (m[-1] > 0 || m[1] > 0 || m[-mstep] > 0 || m[mstep] > 0))
            {
                // Find smallest difference to adjacent markers
                const uchar* ptr = img + j * channel;
                int idx = 256, t;
                if (m[-1] > 0) {
                    c_diff(ptr, ptr - channel, idx);
                }
                if (m[1] > 0) {
                    c_diff(ptr, ptr + channel, t);
                    idx = ws_min(idx, t);
                }
                if (m[-mstep] > 0) {
                    c_diff(ptr, ptr - istep, t);
                    idx = ws_min(idx, t);
                }
                if (m[mstep] > 0) {
                    c_diff(ptr, ptr + istep, t);
                    idx = ws_min(idx, t);
                }

                // Add to according queue
                CV_Assert(0 <= idx && idx <= 255);
                ws_push(idx, i * mstep + j, i * istep + j * channel);
                m[0] = IN_QUEUE;//initial unvisited
            }
        }
    }
    // find the first non-empty queue
    for (i = 0; i < NQ; i++)
        if (q[i].first)
            break;

    // if there is no markers, exit immediately
    if (i == NQ)
        return;

    active_queue = i;//first non-empty priority queue
    img = src.ptr();
    mask = dst.ptr<int>();

    // recursively fill the basins
    int diff = 0, temp = 0;
    for (;;)
    {
        int mofs, iofs;
        int lab = 0, t;
        int* m;
        const uchar* ptr;

        // Get non-empty queue with highest priority
        // Exit condition: empty priority queue
        if (q[active_queue].first == 0)
        {
            for (i = active_queue + 1; i < NQ; i++)
                if (q[i].first)
                    break;
            if (i == NQ)
            {
                std::vector<WSNode>().swap(storage);
                break;
            }
            active_queue = i;
        }

        // Get next node
        ws_pop(active_queue, mofs, iofs);
        int top = 1, bottom = 1, left = 1, right = 1;
        if (0 <= mofs && mofs < mstep)//pixel on the top
            top = 0;
        if ((mofs % mstep) == 0)//pixel in the left column
            left = 0;
        if ((mofs + 1) % mstep == 0)//pixel in the right column
            right = 0;
        if (mstep * (size.height - 1) <= mofs && mofs < mstep * size.height)//pixel on the bottom
            bottom = 0;

        // Calculate pointer to current pixel in input and marker image
        m = mask + mofs;
        ptr = img + iofs;
        // Check surrounding pixels for labels to determine label for current pixel
        if (left) {//the left point can be visited
            t = m[-1];
            if (t > 0) {
                lab = t;
                c_diff(ptr, ptr - channel, diff);
            }
        }
        if (right) {// Right point can be visited
            t = m[1];
            if (t > 0) {
                if (lab == 0) {//and this point didn't be labeled before
                    lab = t;
                    c_diff(ptr, ptr + channel, diff);
                }
                else if (t != lab) {
                    c_diff(ptr, ptr + channel, temp);
                    diff = ws_min(diff, temp);
                    if (diff == temp)
                        lab = t;
                }
            }
        }
        if (top) {
            t = m[-mstep]; // Top
            if (t > 0) {
                if (lab == 0) {//and this point didn't be labeled before
                    lab = t;
                    c_diff(ptr, ptr - istep, diff);
                }
                else if (t != lab) {
                    c_diff(ptr, ptr - istep, temp);
                    diff = ws_min(diff, temp);
                    if (diff == temp)
                        lab = t;
                }
            }
        }
        if (bottom) {
            t = m[mstep]; // Bottom
            if (t > 0) {
                if (lab == 0) {
                    lab = t;
                }
                else if (t != lab) {
                    c_diff(ptr, ptr + istep, temp);
                    diff = ws_min(diff, temp);
                    if (diff == temp)
                        lab = t;
                }
            }
        }
        // Set label to current pixel in marker image
        CV_Assert(lab != 0);//lab must be labeled with a nonzero number
        m[0] = lab;

        // Add adjacent, unlabeled pixels to corresponding queue
        if (left) {
            if (m[-1] == 0)//left pixel with marker 0
            {
                c_diff(ptr, ptr - channel, t);
                ws_push(t, mofs - 1, iofs - channel);
                active_queue = ws_min(active_queue, t);
                m[-1] = IN_QUEUE;
            }
        }

        if (right)
        {
            if (m[1] == 0)//right pixel with marker 0
            {
                c_diff(ptr, ptr + channel, t);
                ws_push(t, mofs + 1, iofs + channel);
                active_queue = ws_min(active_queue, t);
                m[1] = IN_QUEUE;
            }
        }

        if (top)
        {
            if (m[-mstep] == 0)//top pixel with marker 0
            {
                c_diff(ptr, ptr - istep, t);
                ws_push(t, mofs - mstep, iofs - istep);
                active_queue = ws_min(active_queue, t);
                m[-mstep] = IN_QUEUE;
            }
        }

        if (bottom) {
            if (m[mstep] == 0)//down pixel with marker 0
            {
                c_diff(ptr, ptr + istep, t);
                ws_push(t, mofs + mstep, iofs + istep);
                active_queue = ws_min(active_queue, t);
                m[mstep] = IN_QUEUE;
            }
        }
    }
}

void ScanSegmentImpl::getLabels(OutputArray labels_out)
{
    labels_out.assign(labelsMat);
}

void ScanSegmentImpl::getLabelContourMask(OutputArray image, bool thick_line)
{
    image.create(height, width, CV_8UC1);
    cv::Mat dst = image.getMat();
    dst.setTo(cv::Scalar(0));

    const int dx8[8] = { -1, -1, 0, 1, 1, 1, 0, -1 };
    const int dy8[8] = { 0, -1, -1, -1, 0, 1, 1, 1 };

    for (int j = 0; j < height; j++)
    {
        for (int k = 0; k < width; k++)
        {
            int neighbors = 0;
            for (int i = 0; i < 8; i++)
            {
                int x = k + dx8[i];
                int y = j + dy8[i];

                if ((x >= 0 && x < width) && (y >= 0 && y < height))
                {
                    int index = y * width + x;
                    int mainindex = j * width + k;
                    if (((int*)labelsMat.data)[mainindex] != ((int*)labelsMat.data)[index])
                    {
                        if (thick_line || !*dst.ptr<uchar>(y, x))
                            neighbors++;
                    }
                }
            }
            if (neighbors > 1)
                *dst.ptr<uchar>(j, k) = (uchar)255;
        }
    }
}

} // namespace ximgproc
} // namespace cv
