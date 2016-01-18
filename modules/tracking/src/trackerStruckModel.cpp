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
 // Copyright (C) 2013, OpenCV Foundation, all rights reserved.
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
#include "TrackerStruckModel.hpp"

/**
 * TrackerStruckModel
 */
namespace cv
{

    /* Constructor */
    TrackerStruckModel::TrackerStruckModel(const Rect& boundingBox){
		bb = boundingBox;

		Ptr<TrackerStateEstimatorMILBoosting::TrackerMILTargetState> initState = Ptr<TrackerStateEstimatorMILBoosting::TrackerMILTargetState>(
			new TrackerStateEstimatorMILBoosting::TrackerMILTargetState(Point2f((float)bb.x, (float)bb.y), bb.width, bb.height,
				true, Mat()));

		trajectory.push_back(initState);
    }

	void TrackerStruckModel::setCurrentBoundingBox(const Rect & value)
	{
		bb = value;
	}

	void TrackerStruckModel::setCurrentSamples(const std::vector<Mat>& samples)
	{
		currentSamples.clear();
		currentSamples = samples;
	}

	void TrackerStruckModel::setSearchRadius(int radius)
	{
		searchRadius = radius;
	}

	ConfidenceMap TrackerStruckModel::getCurrentConfidenceMap()
	{
		return currentConfidenceMap;
	}

	void TrackerStruckModel::responseToConfidenceMap(const const std::vector<Mat>& resps, ConfidenceMap & map)
	{
		for (int i = 0; i < (int)resps.size(); i++) {

			for (int j = 0; j < resps[i].cols; j++) {

				Size currentSize;
				Point currentOfs;
				currentSamples[j].locateROI(currentSize, currentOfs);

				Point2f pos((float)currentOfs.x, (float)currentOfs.y);

				Mat resp = resps.at(i).col(j);

				Ptr<TrackerStateEstimatorStruckSVM::TrackerStruckTargetState> state = Ptr<TrackerStateEstimatorStruckSVM::TrackerStruckTargetState>(
					new TrackerStateEstimatorStruckSVM::TrackerStruckTargetState(pos, bb.width, bb.height, resp));

				state->isCentre(bb.x == pos.x && bb.y == pos.y);

				if (!state->isCentre()) {
					map.push_back(std::make_pair(state, 0.0f));

					double dist = norm(Point2f((bb.x + bb.width) / 2.0, (bb.y + bb.height) / 2.0) -
						Point2f((pos.x + bb.width) / 2.0, (pos.y + bb.height) / 2.0));
					state->isUpdateOnly(dist > searchRadius);
				}
				else
					// bb state stays at top
					map.insert(map.begin(), std::make_pair(state, 0.0f));
			}
		}
	}
        
    void TrackerStruckModel::modelEstimationImpl( const std::vector<Mat>& responses )
    {
		responseToConfidenceMap(responses, currentConfidenceMap);
    }

    void TrackerStruckModel::modelUpdateImpl()
    {


    }    
    
}