////////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2021, Dr Seng Cheong Loke (lokesengcheong@gmail.com)
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//

 /*
 * BibTeX reference
@article{loke2021accelerated,
  title={Accelerated superpixel image segmentation with a parallelized DBSCAN algorithm},
  author={Loke, Seng Cheong and MacDonald, Bruce A and Parsons, Matthew and W{\"u}nsche, Burkhard Claus},
  journal={Journal of Real-Time Image Processing},
  pages={1--16},
  year={2021},
  publisher={Springer}
}
  */

#include "precomp.hpp"

#include <numeric>
#include <algorithm>

namespace cv {
	namespace ximgproc {

		class ScanSegmentImpl : public ScanSegment
		{
#define UNKNOWN 0
#define BORDER -1
#define UNCLASSIFIED -2
#define NONE -3

		public:

			ScanSegmentImpl(int image_width, int image_height, int num_superpixels, int threads, bool merge_small);

			virtual ~ScanSegmentImpl();

			virtual int getNumberOfSuperpixels() CV_OVERRIDE { return clusterCount; }

			virtual void iterate(InputArray img) CV_OVERRIDE;

			virtual void getLabels(OutputArray labels_out) CV_OVERRIDE;

			virtual void getLabelContourMask(OutputArray image, bool thick_line = false) CV_OVERRIDE;

		private:
			static const int neighbourCount = 8;    // number of pixel neighbours
			static const int smallClustersDiv = 10000;  // divide total pixels by this to give smallClusters
			const float tolerance100 = 10.0f;       // colour tolerance for image size of 100x100px

			int processthreads;						// concurrent threads for parallel processing
			int width, height;                      // image size
			int superpixels;                        // number of superpixels
			bool merge;                             // merge small superpixels
			int indexSize;                          // size of label mat vector
			int clusterSize;                        // max size of clusters
			bool setupComplete;                     // is setup complete
			int clusterCount;                       // number of superpixels from the most recent iterate
			float adjTolerance;						// adjusted colour tolerance

			int horzDiv, vertDiv;                   // number of horizontal and vertical segments
			float horzLength, vertLength;           // length of each segment
			int effectivethreads;                   // effective number of concurrent threads
			int smallClusters;                      // clusters below this pixel count are considered small for merging
			cv::Rect* seedRects;					// array of seed rectangles
			cv::Rect* seedRectsExt;					// array of extended seed rectangles
			cv::Rect* offsetRects;					// array of offset rectangles
			cv::Point* neighbourLoc;				// neighbour locations

			std::vector<int> indexNeighbourVec;		// indices for parallel processing
			std::vector<std::pair<int, int>> indexProcessVec;

			int* labelsBuffer;						// label buffer
			int* clusterBuffer;                     // cluster buffer
			cv::Vec3b* labBuffer;					// lab buffer
			uchar* pixelBuffer;                     // pixel buffer
			int neighbourLocBuffer[neighbourCount]; // neighbour locations
			std::vector<int*> offsetVec;            // vector of offsets

			std::atomic<int> clusterIndex, locationIndex, clusterID;    // atomic indices

			cv::Mat src, labelsMat;	// mats

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

			class PP1 : public cv::ParallelLoopBody
			{
			public:
				PP1(ScanSegmentImpl* const scanSegment)
					: ss(scanSegment) {}
				virtual ~PP1() {}

				virtual void operator()(const cv::Range& range) const
				{
					for (int v = range.start; v < range.end; v++)
					{
						ss->OP1(v);
					}
				}
			private:
				ScanSegmentImpl* const ss;
			};

			class PP2 : public cv::ParallelLoopBody
			{
			public:
				PP2(ScanSegmentImpl* const scanSegment, std::vector<std::pair<int, int>>* const countVec)
					: ss(scanSegment), ctv(countVec) {}
				virtual ~PP2() {}

				virtual void operator()(const cv::Range& range) const
				{
					for (int v = range.start; v < range.end; v++)
					{
						ss->OP2((*ctv)[v]);
					}
				}
			private:
				ScanSegmentImpl* const ss;
				std::vector<std::pair<int, int>>* ctv;
			};

			class PP3 : public cv::ParallelLoopBody
			{
			public:
				PP3(ScanSegmentImpl* const scanSegment)
					: ss(scanSegment) {}
				virtual ~PP3() {}

				virtual void operator()(const cv::Range& range) const
				{
					for (int v = range.start; v < range.end; v++)
					{
						ss->OP3(v);
					}
				}
			private:
				ScanSegmentImpl* const ss;
			};

			class PP4 : public cv::ParallelLoopBody
			{
			public:
				PP4(ScanSegmentImpl* const scanSegment, std::vector<std::pair<int, int>>* const countVec)
					: ss(scanSegment), ctv(countVec) {}
				virtual ~PP4() {}

				virtual void operator()(const cv::Range& range) const
				{
					for (int v = range.start; v < range.end; v++)
					{
						ss->OP4((*ctv)[v]);
					}
				}
			private:
				ScanSegmentImpl* const ss;
				std::vector<std::pair<int, int>>* ctv;
			};

			void OP1(int v);
			void OP2(std::pair<int, int> const& p);
			void OP3(int v);
			void OP4(std::pair<int, int> const& p);
			void expandCluster(int* labelsBuffer, int* neighbourLocBuffer, int* clusterBuffer, int* offsetBuffer, const cv::Point& point, int adjTolerance, std::atomic<int>* clusterIndex, std::atomic<int>* locationIndex, std::atomic<int>* clusterID);
			void calculateCluster(int* labelsBuffer, int* neighbourLocBuffer, int* offsetBuffer, int* offsetEnd, int pointIndex, int adjTolerance, int currentClusterID);
			static int allocWSNodes(std::vector<WSNode>& storage);
			static void watershedEx(const cv::Mat& src, cv::Mat& dst);
		};

		CV_EXPORTS Ptr<ScanSegment> createScanSegment(int image_width, int image_height, int num_superpixels, int threads, bool merge_small)
		{
			return makePtr<ScanSegmentImpl>(image_width, image_height, num_superpixels, threads, merge_small);
		}

		ScanSegmentImpl::ScanSegmentImpl(int image_width, int image_height, int num_superpixels, int threads, bool merge_small)
		{
			// set the number of process threads
			processthreads = std::thread::hardware_concurrency();
			if (threads > 0) {
				processthreads = MIN(processthreads, threads);
			}

			width = image_width;
			height = image_height;
			superpixels = num_superpixels;
			merge = merge_small;
			indexSize = height * width;
			clusterSize = (int)(1.1f * (float)(width * height) / (float)superpixels);
			clusterCount = 0;
			labelsMat = cv::Mat(height, width, CV_32SC1);

			// divide bounds area into uniformly distributed rectangular segments
			int shortCount = (int)floorf(sqrtf((float)processthreads));
			int longCount = processthreads / shortCount;
			horzDiv = width > height ? longCount : shortCount;
			vertDiv = width > height ? shortCount : longCount;
			horzLength = (float)width / (float)horzDiv;
			vertLength = (float)height / (float)vertDiv;
			effectivethreads = horzDiv * vertDiv;
			smallClusters = 0;

			// get array of seed rects
			seedRects = static_cast<cv::Rect*>(malloc(horzDiv * vertDiv * sizeof(cv::Rect)));
			seedRectsExt = static_cast<cv::Rect*>(malloc(horzDiv * vertDiv * sizeof(cv::Rect)));
			offsetRects = static_cast<cv::Rect*>(malloc(horzDiv * vertDiv * sizeof(cv::Rect)));
			for (int y = 0; y < vertDiv; y++) {
				for (int x = 0; x < horzDiv; x++) {
					int xStart = (int)((float)x * horzLength);
					int yStart = (int)((float)y * vertLength);
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

					seedRects[(y * horzDiv) + x] = seedRect;
					seedRectsExt[(y * horzDiv) + x] = cv::Rect(bnd_l, bnd_t, bnd_r - bnd_l + 1, bnd_b - bnd_t + 1);
					offsetRects[(y * horzDiv) + x] = cv::Rect(seedRect.x - bnd_l, seedRect.y - bnd_t, seedRect.width, seedRect.height);
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
			std::vector<cv::Point> tempLoc{ cv::Point(-1, -1), cv::Point(0, -1), cv::Point(1, -1), cv::Point(-1, 0), cv::Point(1, 0), cv::Point(-1, 1), cv::Point(0, 1), cv::Point(1, 1) };
			neighbourLoc = static_cast<cv::Point*>(malloc(8 * sizeof(cv::Point)));
			memcpy(neighbourLoc, tempLoc.data(), 8 * sizeof(cv::Point));

			labelsBuffer = static_cast<int*>(malloc(indexSize * sizeof(int)));
			clusterBuffer = static_cast<int*>(malloc(indexSize * sizeof(int)));
			pixelBuffer = static_cast<uchar*>(malloc(indexSize));
			offsetVec = std::vector<int*>(effectivethreads);
			int offsetSize = (clusterSize + 1) * sizeof(int);
			bool offsetAllocated = true;
			for (int i = 0; i < effectivethreads; i++) {
				offsetVec[i] = static_cast<int*>(malloc(offsetSize));
				if (offsetVec[i] == NULL) {
					offsetAllocated = false;
				}
			}
			for (int i = 0; i < neighbourCount; i++) {
				neighbourLocBuffer[i] = (neighbourLoc[i].y * width) + neighbourLoc[i].x;
			}

			if (labelsBuffer != NULL && clusterBuffer != NULL && pixelBuffer != NULL && offsetAllocated) {
				setupComplete = true;
			}
			else {
				setupComplete = false;

				if (labelsBuffer == NULL) {
					CV_Error(Error::StsInternal, "Cannot initialise labels buffer");
				}
				if (clusterBuffer == NULL) {
					CV_Error(Error::StsInternal, "Cannot initialise cluster buffer");
				}
				if (pixelBuffer == NULL) {
					CV_Error(Error::StsInternal, "Cannot initialise pixel buffer");
				}
				if (!offsetAllocated) {
					CV_Error(Error::StsInternal, "Cannot initialise offset buffers");
				}
			}
		}

		ScanSegmentImpl::~ScanSegmentImpl()
		{
			// clean up
			if (neighbourLoc != NULL) {
				free(neighbourLoc);
			}
			if (seedRects != NULL) {
				free(seedRects);
			}
			if (seedRectsExt != NULL) {
				free(seedRectsExt);
			}
			if (offsetRects != NULL) {
				free(offsetRects);
			}
			if (labelsBuffer != NULL) {
				free(labelsBuffer);
			}
			if (clusterBuffer != NULL) {
				free(clusterBuffer);
			}
			if (pixelBuffer != NULL) {
				free(pixelBuffer);
			}
			for (int i = 0; i < effectivethreads; i++) {
				if (offsetVec[i] != NULL) {
					free(offsetVec[i]);
				}
			}
			if (!src.empty()) {
				src.release();
			}
			if (!labelsMat.empty()) {
				labelsMat.release();
			}
		}

		void ScanSegmentImpl::iterate(InputArray img)
		{
			// ensure setup successfully completed
			CV_Assert(setupComplete);

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
			locationIndex.store(0);
			clusterID.store(1);

			smallClusters = indexSize / smallClustersDiv;

			// set labels to NONE
			labelsMat.setTo(NONE);

			// set labels buffer to UNCLASSIFIED
			std::fill(labelsBuffer, labelsBuffer + indexSize, UNCLASSIFIED);

			// apply light blur
			cv::medianBlur(src, src, 3);

			// start at the center of the rect, then run through the remainder
			labBuffer = reinterpret_cast<cv::Vec3b*>(src.data);
			cv::parallel_for_(cv::Range(0, (int)indexNeighbourVec.size()), PP1(reinterpret_cast<ScanSegmentImpl*>(this)));

			if (merge) {
				// get cutoff size for clusters
				std::vector<std::pair<int, int>> countVec;
				int clusterIndexSize = clusterIndex.load();
				countVec.reserve(clusterIndexSize / 2);
				for (int i = 1; i < clusterIndexSize; i += 2) {
					int count = clusterBuffer[i];
					if (count >= smallClusters) {
						int clusterID = clusterBuffer[i - 1];
						countVec.push_back(std::make_pair(clusterID, count));
					}
				}

				// sort descending
				std::sort(countVec.begin(), countVec.end(), [](auto& left, auto& right) {
					return left.second > right.second;
					});

				int countSize = (int)countVec.size();
				int cutoff = MAX(smallClusters, countVec[MIN(countSize - 1, superpixels - 1)].second);
				clusterCount = (int)std::count_if(countVec.begin(), countVec.end(), [&cutoff](std::pair<int, int> p) {return p.second > cutoff; });

				// change labels to 1 -> clusterCount, 0 = UNKNOWN, reuse clusterbuffer
				std::fill_n(clusterBuffer, indexSize, UNKNOWN);
				int countLimit = cutoff == -1 ? (int)countVec.size() : clusterCount;
				for (int i = 0; i < countLimit; i++) {
					clusterBuffer[countVec[i].first] = i + 1;
				}

				cv::parallel_for_(cv::Range(0, (int)indexProcessVec.size()), PP2(reinterpret_cast<ScanSegmentImpl*>(this), &indexProcessVec));

				// make copy of labels buffer
				memcpy(labelsMat.data, labelsBuffer, indexSize * sizeof(int));

				// run watershed
				cv::parallel_for_(cv::Range(0, (int)indexNeighbourVec.size()), PP3(reinterpret_cast<ScanSegmentImpl*>(this)));

				// copy back to labels mat
				cv::parallel_for_(cv::Range(0, (int)indexProcessVec.size()), PP4(reinterpret_cast<ScanSegmentImpl*>(this), &indexProcessVec));
			}
			else {
				memcpy(labelsMat.data, labelsBuffer, indexSize * sizeof(int));
			}

			src.release();
		}

		void ScanSegmentImpl::OP1(int v)
		{
			cv::Rect seedRect = seedRects[v];
			for (int y = seedRect.y; y < seedRect.y + seedRect.height; y++) {
				for (int x = seedRect.x; x < seedRect.x + seedRect.width; x++) {
					expandCluster(labelsBuffer, neighbourLocBuffer, clusterBuffer, offsetVec[v], cv::Point(x, y), (int)adjTolerance, &clusterIndex, &locationIndex, &clusterID);
				}
			}
		}

		void ScanSegmentImpl::OP2(std::pair<int, int> const& p)
		{
			std::pair<int, int>& q = const_cast<std::pair<int, int>&>(p);
			for (int i = q.first; i < q.second; i++) {
				labelsBuffer[i] = clusterBuffer[labelsBuffer[i]];
				if (labelsBuffer[i] == UNKNOWN) {
					pixelBuffer[i] = 255;
				}
				else {
					pixelBuffer[i] = 0;
				}
			}
		}

		void ScanSegmentImpl::OP3(int v)
		{
			cv::Rect seedRect = seedRects[v];
			cv::Rect seedRectExt = seedRectsExt[v];

			cv::Mat seedLabels = labelsMat(seedRectExt).clone();
			watershedEx(src(seedRectExt), seedLabels);
			seedLabels(offsetRects[v]).copyTo(labelsMat(seedRects[v]));
			seedLabels.release();
		}

		void ScanSegmentImpl::OP4(std::pair<int, int> const& p)
		{
			std::pair<int, int>& q = const_cast<std::pair<int, int>&>(p);
			for (int i = q.first; i < q.second; i++) {
				if (pixelBuffer[i] == 0) {
					((int*)labelsMat.data)[i] = labelsBuffer[i] - 1;
				}
				else {
					((int*)labelsMat.data)[i] -= 1;
				}
			}
		}

		// expand clusters from a point
		void ScanSegmentImpl::expandCluster(int* labelsBuffer, int* neighbourLocBuffer, int* clusterBuffer, int* offsetBuffer, const cv::Point& point, int adjTolerance, std::atomic<int>* clusterIndex, std::atomic<int>* locationIndex, std::atomic<int>* clusterID)
		{
			int pointIndex = (point.y * width) + point.x;
			if (labelsBuffer[pointIndex] == UNCLASSIFIED) {
				int offsetStart = 0;
				int offsetEnd = 0;
				int currentClusterID = clusterID->fetch_add(1);

				calculateCluster(labelsBuffer, neighbourLocBuffer, offsetBuffer, &offsetEnd, pointIndex, adjTolerance, currentClusterID);

				if (offsetStart == offsetEnd) {
					labelsBuffer[pointIndex] = UNKNOWN;
				}
				else {
					// set cluster id and get core point index
					labelsBuffer[pointIndex] = currentClusterID;

					while (offsetStart < offsetEnd) {
						int intoffset2 = *(offsetBuffer + offsetStart);
						offsetStart++;
						calculateCluster(labelsBuffer, neighbourLocBuffer, offsetBuffer, &offsetEnd, intoffset2, adjTolerance, currentClusterID);
					}

					// add origin point
					offsetBuffer[offsetEnd] = pointIndex;
					offsetEnd++;

					// store to buffer
					int currentClusterIndex = clusterIndex->fetch_add(2);
					clusterBuffer[currentClusterIndex] = currentClusterID;
					clusterBuffer[currentClusterIndex + 1] = offsetEnd;
				}
			}
		}

		void ScanSegmentImpl::calculateCluster(int* labelsBuffer, int* neighbourLocBuffer, int* offsetBuffer, int* offsetEnd, int pointIndex, int adjTolerance, int currentClusterID)
		{
			for (int i = 0; i < neighbourCount; i++) {
				if (*offsetEnd < clusterSize) {
					int intoffset2 = pointIndex + neighbourLocBuffer[i];
					if (intoffset2 >= 0 && intoffset2 < indexSize && labelsBuffer[intoffset2] == UNCLASSIFIED) {
						int diff1 = (int)labBuffer[pointIndex][0] - (int)labBuffer[intoffset2][0];
						int diff2 = (int)labBuffer[pointIndex][1] - (int)labBuffer[intoffset2][1];
						int diff3 = (int)labBuffer[pointIndex][2] - (int)labBuffer[intoffset2][2];

						if ((diff1 * diff1) + (diff2 * diff2) + (diff3 * diff3) <= adjTolerance) {
							labelsBuffer[intoffset2] = currentClusterID;
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
    if (!free_node)                    \
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
    assert(0 <= diff && diff <= 255);  \
        }

			CV_Assert(src.type() == CV_8UC3 || src.type() == CV_8UC1 && dst.type() == CV_32SC1);
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
						assert(0 <= idx && idx <= 255);
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
				int diff, temp;
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
				assert(lab != 0);//lab must be labeled with a nonzero number
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
