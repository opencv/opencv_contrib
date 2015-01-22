/*M///////////////////////////////////////////////////////////////////////////////////////
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
 // Copyright (C) 2014, Biagio Montesano, all rights reserved.
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
 //M*/

#include "precomp.hpp"

//using namespace cv;
namespace cv
{
	namespace line_descriptor
	{

		/* constructor */
		BinaryDescriptorMatcher::BinaryDescriptorMatcher()
		{
			dataset = new mihasher(256, 32);
			nextAddedIndex = 0;
			numImages = 0;
			descrInDS = 0;
		}

		BinaryDescriptorMatcher::~BinaryDescriptorMatcher()
		{
			clear();
		}

		/* constructor with smart pointer */
		Ptr<BinaryDescriptorMatcher> BinaryDescriptorMatcher::createBinaryDescriptorMatcher()
		{
			return Ptr < BinaryDescriptorMatcher >(new BinaryDescriptorMatcher());
		}

		/* store new descriptors to be inserted in dataset */
		void BinaryDescriptorMatcher::add(const std::vector<Mat>& descriptors)
		{
			for (size_t i = 0; i < descriptors.size(); i++)
			{
				descriptorsMat.push_back(descriptors[i]);

				indexesMap.insert(std::pair<int, int>(nextAddedIndex, numImages));
				nextAddedIndex += descriptors[i].rows;
				numImages++;
			}
		}

		/* store new descriptors into dataset */
		void BinaryDescriptorMatcher::train()
		{
			if (!dataset)
				dataset = new mihasher(256, 32);

			if (descriptorsMat.rows > 0)
				dataset->populate((UINT8*)descriptorsMat.data, descriptorsMat.rows, descriptorsMat.cols);

			descrInDS = descriptorsMat.rows;
			descriptorsMat.release();
		}

		/* clear dataset and internal data */
		void BinaryDescriptorMatcher::clear()
		{
			descriptorsMat.release();
			indexesMap.clear();
			if (dataset)
				delete dataset;
			dataset = 0;
			nextAddedIndex = 0;
			numImages = 0;
			descrInDS = 0;
		}

		/* retrieve Hamming distances */
		void BinaryDescriptorMatcher::checkKDistances(UINT32 * numres, int k, std::vector<int> & k_distances, int row, int string_length) const
		{
			int k_to_found = k;

			UINT32 * numres_tmp = numres + ((string_length + 1) * row);
			for (int j = 0; j < (string_length + 1) && k_to_found > 0; j++)
			{
				if ((*(numres_tmp + j)) > 0)
				{
					for (int i = 0; i < (int)(*(numres_tmp + j)) && k_to_found > 0; i++)
					{
						k_distances.push_back(j);
						k_to_found--;
					}
				}
			}
		}

		/* for every input descriptor,
		 find the best matching one (from one image to a set) */
		void BinaryDescriptorMatcher::match(const Mat& queryDescriptors, std::vector<DMatch>& matches, const std::vector<Mat>& masks)
		{
			/* check data validity */
			if (queryDescriptors.rows == 0)
			{
				std::cout << "Error: query descriptors'matrix is empty" << std::endl;
				return;
			}

			if (masks.size() != 0 && (int)masks.size() != numImages)
			{
				std::cout << "Error: the number of images in dataset is " << numImages << " but match function received " << masks.size()
					<< " masks. Program will be terminated" << std::endl;

				return;
			}

			/* add new descriptors to dataset, if needed */
			train();

			/* set number of requested matches to return for each query */
			dataset->setK(1);

			/* prepare structures for query */
			UINT32 *results = new UINT32[queryDescriptors.rows];
			UINT32 * numres = new UINT32[(256 + 1) * (queryDescriptors.rows)];

			/* execute query */
			dataset->batchquery(results, numres, (UINT8*)queryDescriptors.data, queryDescriptors.rows, queryDescriptors.cols);
			/* compose matches */
			for (int counter = 0; counter < queryDescriptors.rows; counter++)
			{
				/* create a map iterator */
				std::map<int, int>::iterator itup;

				/* get info about original image of each returned descriptor */
				itup = indexesMap.upper_bound(results[counter] - 1);
				itup--;
				/* data validity check */
				if (!masks.empty() && (masks[itup->second].rows != queryDescriptors.rows || masks[itup->second].cols != 1))
				{
					std::stringstream ss;
					ss << "Error: mask " << itup->second << " in knnMatch function " << "should have " << queryDescriptors.rows << " and "
						<< "1 column. Program will be terminated";
					//throw std::runtime_error( ss.str() );
				}
				/* create a DMatch object if required by mask or if there is
				 no mask at all */
				else if (masks.empty() || masks[itup->second].at < uchar >(counter) != 0)
				{
					std::vector<int> k_distances;
					checkKDistances(numres, 1, k_distances, counter, 256);

					DMatch dm;
					dm.queryIdx = counter;
					dm.trainIdx = results[counter] - 1;
					dm.imgIdx = itup->second;
					dm.distance = (float)k_distances[0];

					matches.push_back(dm);
				}

			}

			/* delete data */
			delete[] results;
			delete[] numres;
		}

		/* for every input descriptor, find the best matching one (for a pair of images) */
		void BinaryDescriptorMatcher::match(const Mat& queryDescriptors, const Mat& trainDescriptors, std::vector<DMatch>& matches, const Mat& mask) const
		{

			/* check data validity */
			if (queryDescriptors.rows == 0 || trainDescriptors.rows == 0)
			{
				std::cout << "Error: descriptors matrices cannot be void" << std::endl;
				return;
			}

			if (!mask.empty() && (mask.rows != queryDescriptors.rows && mask.cols != 1))
			{
				std::cout << "Error: input mask should have " << queryDescriptors.rows << " rows and 1 column. " << "Program will be terminated" << std::endl;

				return;
			}

			/* create a new mihasher object */
			mihasher *mh = new mihasher(256, 32);

			/* populate mihasher */
			cv::Mat copy = trainDescriptors.clone();
			mh->populate((UINT8*)copy.data, copy.rows, copy.cols);
			mh->setK(1);

			/* prepare structures for query */
			UINT32 *results = new UINT32[queryDescriptors.rows];
			UINT32 * numres = new UINT32[(256 + 1) * (queryDescriptors.rows)];

			/* execute query */
			mh->batchquery(results, numres, (UINT8*)queryDescriptors.data, queryDescriptors.rows, queryDescriptors.cols);

			/* compose matches */
			for (int counter = 0; counter < queryDescriptors.rows; counter++)
			{
				/* create a DMatch object if required by mask or if there is
				 no mask at all */
				if (mask.empty() || (!mask.empty() && mask.at < uchar >(counter) != 0))
				{
					std::vector<int> k_distances;
					checkKDistances(numres, 1, k_distances, counter, 256);

					DMatch dm;
					dm.queryIdx = counter;
					dm.trainIdx = results[counter] - 1;
					dm.imgIdx = 0;
					dm.distance = (float)k_distances[0];

					matches.push_back(dm);
				}
			}

			/* delete data */
			delete mh;
			delete[] results;
			delete[] numres;

		}

		/* for every input descriptor,
		 find the best k matching descriptors (for a pair of images) */
		void BinaryDescriptorMatcher::knnMatch(const Mat& queryDescriptors, const Mat& trainDescriptors, std::vector<std::vector<DMatch> >& matches, int k,
			const Mat& mask, bool compactResult) const

		{
			/* check data validity */
			if (queryDescriptors.rows == 0 || trainDescriptors.rows == 0)
			{
				std::cout << "Error: descriptors matrices cannot be void" << std::endl;
				return;
			}

			if (!mask.empty() && (mask.rows != queryDescriptors.rows || mask.cols != 1))
			{
				std::cout << "Error: input mask should have " << queryDescriptors.rows << " rows and 1 column. " << "Program will be terminated" << std::endl;

				return;
			}

			/* create a new mihasher object */
			mihasher *mh = new mihasher(256, 32);

			/* populate mihasher */
			cv::Mat copy = trainDescriptors.clone();
			mh->populate((UINT8*)copy.data, copy.rows, copy.cols);

			/* set K */
			mh->setK(k);

			/* prepare structures for query */
			UINT32 *results = new UINT32[k * queryDescriptors.rows];
			UINT32 * numres = new UINT32[(256 + 1) * (queryDescriptors.rows)];

			/* execute query */
			mh->batchquery(results, numres, (UINT8*)queryDescriptors.data, queryDescriptors.rows, queryDescriptors.cols);

			/* compose matches */
			int index = 0;
			for (int counter = 0; counter < queryDescriptors.rows; counter++)
			{
				/* initialize a vector of matches */
				std::vector < DMatch > tempVec;

				/* chech whether query should be ignored */
				if (!mask.empty() && mask.at < uchar >(counter) == 0)
				{
					/* if compact result is not requested, add an empty vector */
					if (!compactResult)
						matches.push_back(tempVec);
				}

				/* query matches must be considered */
				else
				{
					std::vector<int> k_distances;
					checkKDistances(numres, k, k_distances, counter, 256);
					for (int j = index; j < index + k; j++)
					{
						DMatch dm;
						dm.queryIdx = counter;
						dm.trainIdx = results[j] - 1;
						dm.imgIdx = 0;
						dm.distance = (float)k_distances[j - index];

						tempVec.push_back(dm);
					}

					matches.push_back(tempVec);
				}

				/* increment pointer */
				index += k;
			}

			/* delete data */
			delete mh;
			delete[] results;
			delete[] numres;
		}

		/* for every input descriptor,
		 find the best k matching descriptors (from one image to a set) */
		void BinaryDescriptorMatcher::knnMatch(const Mat& queryDescriptors, std::vector<std::vector<DMatch> >& matches, int k, const std::vector<Mat>& masks,
			bool compactResult)
		{

			/* check data validity */
			if (queryDescriptors.rows == 0)
			{
				std::cout << "Error: descriptors matrix cannot be void" << std::endl;
				return;
			}

			if (masks.size() != 0 && (int)masks.size() != numImages)
			{
				std::cout << "Error: the number of images in dataset is " << numImages << " but knnMatch function received " << masks.size()
					<< " masks. Program will be terminated" << std::endl;

				return;
			}

			/* add new descriptors to dataset, if needed */
			train();

			/* set number of requested matches to return for each query */
			dataset->setK(k);

			/* prepare structures for query */
			UINT32 *results = new UINT32[k * queryDescriptors.rows];
			UINT32 * numres = new UINT32[(256 + 1) * (queryDescriptors.rows)];

			/* execute query */
			dataset->batchquery(results, numres, (UINT8*)queryDescriptors.data, queryDescriptors.rows, queryDescriptors.cols);

			/* compose matches */
			int index = 0;
			for (int counter = 0; counter < queryDescriptors.rows; counter++)
			{
				/* create a void vector of matches */
				std::vector < DMatch > tempVector;

				/* loop over k results returned for every query */
				for (int j = index; j < index + k; j++)
				{
					/* retrieve which image returned index refers to */
					int currentIndex = results[j] - 1;
					std::map<int, int>::iterator itup;
					itup = indexesMap.upper_bound(currentIndex);
					itup--;

					/* data validity check */
					if (!masks.empty() && (masks[itup->second].rows != queryDescriptors.rows || masks[itup->second].cols != 1))
					{
						std::cout << "Error: mask " << itup->second << " in knnMatch function " << "should have " << queryDescriptors.rows << " and "
							<< "1 column. Program will be terminated" << std::endl;

						return;
					}

					/* decide if, according to relative mask, returned match should be
					 considered */
					else if (masks.size() == 0 || masks[itup->second].at < uchar >(counter) != 0)
					{
						std::vector<int> k_distances;
						checkKDistances(numres, k, k_distances, counter, 256);

						DMatch dm;
						dm.queryIdx = counter;
						dm.trainIdx = results[j] - 1;
						dm.imgIdx = itup->second;
						dm.distance = (float)k_distances[j - index];

						tempVector.push_back(dm);
					}
				}

				/* decide whether temporary vector should be saved */
				if ((tempVector.size() == 0 && !compactResult) || tempVector.size() > 0)
					matches.push_back(tempVector);

				/* increment pointer */
				index += k;
			}

			/* delete data */
			delete[] results;
			delete[] numres;
		}

		/* for every input desciptor, find all the ones falling in a
		 certaing matching radius (for a pair of images) */
		void BinaryDescriptorMatcher::radiusMatch(const Mat& queryDescriptors, const Mat& trainDescriptors, std::vector<std::vector<DMatch> >& matches,
			float maxDistance, const Mat& mask, bool compactResult) const

		{

			/* check data validity */
			if (queryDescriptors.rows == 0 || trainDescriptors.rows == 0)
			{
				std::cout << "Error: descriptors matrices cannot be void" << std::endl;
				return;
			}

			if (!mask.empty() && (mask.rows != queryDescriptors.rows && mask.cols != 1))
			{
				std::cout << "Error: input mask should have " << queryDescriptors.rows << " rows and 1 column. " << "Program will be terminated" << std::endl;

				return;
			}

			/* create a new Mihasher */
			mihasher* mh = new mihasher(256, 32);

			/* populate Mihasher */
			//Mat copy = queryDescriptors.clone();
			Mat copy = trainDescriptors.clone();
			mh->populate((UINT8*)copy.data, copy.rows, copy.cols);

			/* set K */
			mh->setK(trainDescriptors.rows);

			/* prepare structures for query */
			UINT32 *results = new UINT32[trainDescriptors.rows * queryDescriptors.rows];
			UINT32 * numres = new UINT32[(256 + 1) * (queryDescriptors.rows)];

			/* execute query */
			mh->batchquery(results, numres, (UINT8*)queryDescriptors.data, queryDescriptors.rows, queryDescriptors.cols);

			/* compose matches */
			int index = 0;
			for (int i = 0; i < queryDescriptors.rows; i++)
			{
				std::vector<int> k_distances;
				checkKDistances(numres, trainDescriptors.rows, k_distances, i, 256);

				std::vector < DMatch > tempVector;
				for (int j = index; j < index + trainDescriptors.rows; j++)
				{
					//      if( numres[j] <= maxDistance )
					if (k_distances[j - index] <= maxDistance)
					{
						if (mask.empty() || mask.at < uchar >(i) != 0)
						{
							DMatch dm;
							dm.queryIdx = i;
							dm.trainIdx = (int)(results[j] - 1);
							dm.imgIdx = 0;
							dm.distance = (float)k_distances[j - index];

							tempVector.push_back(dm);
						}
					}
				}

				/* decide whether temporary vector should be saved */
				if ((tempVector.size() == 0 && !compactResult) || tempVector.size() > 0)
					matches.push_back(tempVector);

				/* increment pointer */
				index += trainDescriptors.rows;

			}

			/* delete data */
			delete mh;
			delete[] results;
			delete[] numres;
		}

		/* for every input descriptor, find all the ones falling in a
		 certain matching radius (from one image to a set) */
		void BinaryDescriptorMatcher::radiusMatch(const Mat& queryDescriptors, std::vector<std::vector<DMatch> >& matches, float maxDistance,
			const std::vector<Mat>& masks, bool compactResult)
		{

			/* check data validity */
			if (queryDescriptors.rows == 0)
			{
				std::cout << "Error: descriptors matrices cannot be void" << std::endl;
				return;
			}

			if (masks.size() != 0 && (int)masks.size() != numImages)
			{
				std::cout << "Error: the number of images in dataset is " << numImages << " but radiusMatch function received " << masks.size()
					<< " masks. Program will be terminated" << std::endl;

				return;
			}

			/* populate dataset */
			train();

			/* set K */
			dataset->setK(descrInDS);

			/* prepare structures for query */
			UINT32 *results = new UINT32[descrInDS * queryDescriptors.rows];
			UINT32 * numres = new UINT32[(256 + 1) * (queryDescriptors.rows)];

			/* execute query */
			dataset->batchquery(results, numres, (UINT8*)queryDescriptors.data, queryDescriptors.rows, queryDescriptors.cols);

			/* compose matches */
			int index = 0;
			for (int counter = 0; counter < queryDescriptors.rows; counter++)
			{
				std::vector < DMatch > tempVector;
				for (int j = index; j < index + descrInDS; j++)
				{
					std::vector<int> k_distances;
					checkKDistances(numres, descrInDS, k_distances, counter, 256);

					if (k_distances[j - index] <= maxDistance)
					{
						int currentIndex = results[j] - 1;
						std::map<int, int>::iterator itup;
						itup = indexesMap.upper_bound(currentIndex);
						itup--;

						/* data validity check */
						if (!masks.empty() && (masks[itup->second].rows != queryDescriptors.rows || masks[itup->second].cols != 1))
						{
							std::cout << "Error: mask " << itup->second << " in radiusMatch function " << "should have " << queryDescriptors.rows << " and "
								<< "1 column. Program will be terminated" << std::endl;

							return;
						}

						/* add match if necessary */
						else if (masks.empty() || masks[itup->second].at < uchar >(counter) != 0)
						{

							DMatch dm;
							dm.queryIdx = counter;
							dm.trainIdx = results[j] - 1;
							dm.imgIdx = itup->second;
							dm.distance = (float)k_distances[j - index];

							tempVector.push_back(dm);
						}
					}
				}

				/* decide whether temporary vector should be saved */
				if ((tempVector.size() == 0 && !compactResult) || tempVector.size() > 0)
					matches.push_back(tempVector);

				/* increment pointer */
				index += descrInDS;
			}

			/* delete data */
			delete[] results;
			delete[] numres;
		}

		// all the below code is ported from "https://github.com/norouzi/mih" with bug fixes
		void BinaryDescriptorMatcher::mihasher::batchquery(UINT32 *results, UINT32 *numres, UINT8 *queries, UINT32 numq, int dim1queries)
		{
			counter = new bitarray;
			counter->init(N);

			UINT32 *res = new UINT32[K*(D + 1)];
			UINT64 *chunks = new UINT64[m];

			UINT32 *presults = results;
			UINT32 *pnumres = numres;
			
			UINT8 *pq = queries;

			for (unsigned int i = 0; i < numq; i++) {
				query(presults, pnumres, pq, chunks, res);

				presults += K;
				pnumres += B + 1;
				pq += dim1queries;
			}

			delete[] res;
			delete[] chunks;

			delete counter;
		}

		// Temp variables: chunks, res -- we did not want to malloc inside
		// query, so these arrays are passed from outside

		void BinaryDescriptorMatcher::mihasher::query(UINT32 *results, UINT32* numres, UINT8 *queryData, UINT64 *chunks, UINT32 *res)
		{
			UINT32 maxres = K ? K : (UINT32) N;			// if K == 0 that means we want everything to be processed.
			// So maxres = N in that case. Otherwise K limits the results processed.

			UINT32 n = 0; 				// number of results so far obtained (up to a distance of s per chunk)
			UINT32 nc = 0;				// number of candidates tested with full codes (not counting duplicates)
			UINT32 nd = 0;                      	// counting everything retrieved (duplicates are counted multiple times)
			UINT32 nl = 0;				// number of lookups (and xors)
			UINT32 *arr;
			int size = 0;
			UINT32 index;
			int hammd;
			
			counter->erase();
			memset(numres, 0, (B + 1)*sizeof(*numres));

			split(chunks, queryData, m, mplus, b);

			int s;			// the growing search radius per substring

			int curb = b;		// current b: for the first mplus substrings it is b, for the rest it is (b-1)

			for (s = 0; s <= d && n < maxres; s++) {
				for (int k = 0; k < m; k++) {
					if (k < mplus)
						curb = b;
					else
						curb = b - 1;
					UINT64 chunksk = chunks[k];
					nl += xornum[s + 1] - xornum[s];	// number of bit-strings with s number of 1s

					UINT64 bitstr = 0; 			// the bit-string with s number of 1s
					for (int i = 0; i < s; i++)
						power[i] = i;			// power[i] stores the location of the i'th 1
					power[s] = curb + 1;			// used for stopping criterion (location of (s+1)th 1)

					int bit = s - 1;			// bit determines the 1 that should be moving to the left
					// we start from the left-most 1, and move it to the left until it touches another one

					bool infiniteWhile = true;
					while (infiniteWhile) {			// the loop for changing bitstr
						if (bit != -1) {
							bitstr ^= (power[bit] == bit) ? (UINT64)1 << power[bit] : (UINT64)3 << (power[bit] - 1);
							power[bit]++;
							bit--;
						}
						else { // bit == -1
							/* the binary code bitstr is available for processing */
							arr = H[k].query(chunksk ^ bitstr, &size); // lookup
							if (size) {			// the corresponding bucket is not empty
								nd += size;
								for (int c = 0; c < size; c++) {
									index = arr[c];
									if (!counter->get(index)) { // if it is not a duplicate
										counter->set(index);
										hammd = cv::line_descriptor::match(codes + (UINT64)index*(B_over_8), queryData, B_over_8);
										nc++;
										if (hammd <= D && numres[hammd] < maxres) {
											res[hammd * K + numres[hammd]] = index + 1;
										}
										numres[hammd]++;
									}
								}
							}
							/* end of processing */

							while (++bit < s && power[bit] == power[bit + 1] - 1) {
								bitstr ^= (UINT64)1 << (power[bit] - 1);
								power[bit] = bit;
							}
							if (bit == s)
								break;
						}
					}

					n = n + numres[s*m + k]; // this line is very tricky ;)
					// The k'th substring (0 based) is the last chance of an
					// item at a Hamming distance of s*m+k to be
					// found. Because if until the k'th substring, an item
					// with distance of s*m+k is not found, then it means that
					// all of the substrings so far have a distance of (s+1)
					// or more, and the remaining substrings have a distance
					// of s or more (total > s*m+k).

					if (n >= maxres)
						break;
				}
			}

			n = 0;
			for (s = 0; s <= D && (int)n < K; s++) {
				for (unsigned int c = 0; c < numres[s] && (int)n < K; c++)
					results[n++] = res[s*K + c];
			}
		}

		BinaryDescriptorMatcher::mihasher::mihasher(int _B, int _m)
		{
			B = _B;
			B_over_8 = B / 8;
			m = _m;
			b = (int)ceil((double)B / m);

			D = B;
			d = (int)ceil((double)D / m);

			mplus = B - m * (b - 1);
			// mplus     is the number of chunks with b bits
			// (m-mplus) is the number of chunks with (b-1) bits

			xornum = new UINT32[d + 2];
			xornum[0] = 0;
			for (int i = 0; i <= d; i++)
				xornum[i + 1] = xornum[i] + (UINT32)choose(b, i);

			H = new SparseHashtable[m];
			// H[i].init might fail
			for (int i = 0; i < mplus; i++)
				H[i].init(b);
			for (int i = mplus; i < m; i++)
				H[i].init(b - 1);
		}

		void BinaryDescriptorMatcher::mihasher::setK(int _K)
		{
			K = _K;
		}

		BinaryDescriptorMatcher::mihasher::~mihasher()
		{
			delete[] xornum;
			delete[] H;
		}

		void BinaryDescriptorMatcher::mihasher::populate(UINT8 *_codes, UINT32 _N, int dim1codes)
		{
			N = _N;
			codes = _codes;
			UINT64 * chunks = new UINT64[m];

			UINT8 * pcodes = codes;
			for (UINT64 i = 0; i < N; i++, pcodes += dim1codes) {
				split(chunks, pcodes, m, mplus, b);

				for (int k = 0; k < m; k++)
					H[k].insert(chunks[k], (UINT32)i);

			}

			delete[] chunks;
		}


		const int BinaryDescriptorMatcher::SparseHashtable::MAX_B = 37;

		BinaryDescriptorMatcher::SparseHashtable::SparseHashtable() {
			table = 0;
			size = 0;
			b = 0;
		}

		int BinaryDescriptorMatcher::SparseHashtable::init(int _b) {
			b = _b;
			if ((b < 5) || (b > MAX_B) || (b > (int)(sizeof(UINT64) * 8)))
				return 1;

			size = UINT64_1 << (b - 5);	// size = 2 ^ b
			table = new BucketGroup[(int)size];
			//table = (BucketGroup*)calloc(size, sizeof(BucketGroup));

			return 0;
		}

		BinaryDescriptorMatcher::SparseHashtable::~SparseHashtable() {
			delete[] table;
		}

		void BinaryDescriptorMatcher::SparseHashtable::insert(UINT64 index, UINT32 data) {
			table[index >> 5].insert((int)(index % 32), data);
		}

		UINT32* BinaryDescriptorMatcher::SparseHashtable::query(UINT64 index, int *tableSize) {
			return table[index >> 5].query((int)(index % 32), tableSize);
		}

		BinaryDescriptorMatcher::BucketGroup::BucketGroup() {
			empty = 0;
			group = NULL;
		}

		BinaryDescriptorMatcher::BucketGroup::~BucketGroup() {
			if (group != NULL)
				delete group;
		}

		UINT32 BinaryDescriptorMatcher::BucketGroup::ones32(UINT32 i)
		{
			i = i - ((i >> 1) & 0x55555555);
			i = (i & 0x33333333) + ((i >> 2) & 0x33333333);
			return (((i + (i >> 4)) & 0x0F0F0F0F) * 0x01010101) >> 24;
		}

		void BinaryDescriptorMatcher::BucketGroup::insert(int subindex, UINT32 data) {
			if (group == NULL) {
				group = new Array32();
				group->push(0);
			}

			UINT32 lowerbits = ((UINT32)1 << subindex) - 1;
			int end = ones32(empty & lowerbits);
			if (!(empty & ((UINT32)1 << subindex))) {
				group->insert(end, group->arr[end + 2]);
				empty |= (UINT32)1 << subindex;
			}

			int totones = ones32(empty);
			group->insert(totones + 1 + group->arr[2 + end + 1], data);
			for (int i = end + 1; i < totones + 1; i++)
				group->arr[2 + i]++;
		}

		UINT32* BinaryDescriptorMatcher::BucketGroup::query(int subindex, int *size) {
			if (empty & ((UINT32)1 << subindex)) {
				UINT32 lowerbits = ((UINT32)1 << subindex) - 1;
				int end = ones32(empty & lowerbits);
				int totones = ones32(empty);
				*size = group->arr[2 + end + 1] - group->arr[2 + end];
				return group->arr + 2 + totones + 1 + group->arr[2 + end];
			}
			else {
				*size = 0;
				return NULL;
			}
		}

		/* no need for the static keywork in the definition */
		double BinaryDescriptorMatcher::Array32::ARRAY_RESIZE_FACTOR = 1.1;    // minimum is 1.0
		int BinaryDescriptorMatcher::Array32::ARRAY_RESIZE_ADD_FACTOR = 4;  // minimum is 1

		void BinaryDescriptorMatcher::Array32::set_array_resize_factor(double arf) {
			ARRAY_RESIZE_FACTOR = arf;
		}

		BinaryDescriptorMatcher::Array32::Array32() {
			arr = NULL;
		}

		BinaryDescriptorMatcher::Array32& BinaryDescriptorMatcher::Array32::operator = (const Array32 &rhs) {
			if (&rhs != this)
				this->arr = rhs.arr;
			return *this;
		}

		BinaryDescriptorMatcher::Array32::~Array32() {
			cleanup();
		}

		void BinaryDescriptorMatcher::Array32::cleanup() {
			free(arr);			// sometimes call free on NULL pointers, but that should be fine.
		}

		void BinaryDescriptorMatcher::Array32::push(UINT32 arrayData) {
			if (arr) {
				if (arr[0] == arr[1]) {
					arr[1] = std::max((int)ceil(arr[1] * ARRAY_RESIZE_FACTOR), (int)(arr[1] + ARRAY_RESIZE_ADD_FACTOR));
					arr = (UINT32*)realloc(arr, sizeof(UINT32)*(2 + arr[1]));
				}
				arr[2 + arr[0]] = arrayData;
				arr[0]++;
			}
			else {
				arr = (UINT32*)malloc((2 + ARRAY_RESIZE_ADD_FACTOR)*sizeof(UINT32));
				arr[0] = 1;
				arr[1] = 1;
				arr[2] = arrayData;
			}
		}

		void BinaryDescriptorMatcher::Array32::insert(UINT32 index, UINT32 arrayData) {
			if (arr) {
				if (arr[0] == arr[1]) {
					arr[1] = (UINT32)ceil(arr[0] * ARRAY_RESIZE_FACTOR);
					arr = (UINT32*)realloc(arr, sizeof(UINT32)*(2 + arr[1]));
				}

				memmove(arr + (2 + index) + 1, arr + (2 + index), (arr[0] - index)*sizeof(*arr));
				// /* alternative to memmove */
				// for (int i=2+arr[0]; i>2+index; i--)
				//     arr[i] = arr[i-1];

				arr[2 + index] = arrayData;
				arr[0]++;
			}
			else {
				arr = (UINT32*)malloc(3 * sizeof(UINT32));
				arr[0] = 1;
				arr[1] = 1;
				arr[2] = arrayData;
			}
		}



		UINT32* BinaryDescriptorMatcher::Array32::data() {
			return arr ? arr + 2 : NULL;
		}

		UINT32 BinaryDescriptorMatcher::Array32::size() {
			return arr ? arr[0] : 0;
		}

		UINT32 BinaryDescriptorMatcher::Array32::capacity() {
			return arr ? arr[1] : 0;
		}

		void BinaryDescriptorMatcher::Array32::print() {
			for (unsigned int i = 0; i<size(); i++) {
				printf("%d, ", arr[i + 2]);
			}
			printf("\n");
		}

		void BinaryDescriptorMatcher::Array32::init(int arraySize) {
			if (arr == NULL) {
				arr = (UINT32*)malloc((2 + arraySize)*sizeof(UINT32));
				arr[0] = 0;
				arr[1] = arraySize;
			}
		}

	}
}

