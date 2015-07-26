#include <precomp.hpp>

namespace cv
{
	//Multitracker
	bool MultiTracker::addTarget(const Mat& image, const Rect2d& boundingBox, char* tracker_algorithm_name)
	{
		Ptr<Tracker> tracker = Tracker::create(tracker_algorithm_name);
		if (tracker == NULL)
			return false;

		if (!tracker->init(image, boundingBox))
			return false;

		//Add BB of target
		boundingBoxes.push_back(boundingBox);

		//Add Tracker to stack
		trackers.push_back(tracker);

		//Assign a random color to target
		colors.push_back(Scalar(rand() % 256, rand() % 256, rand() % 256));

		//Target counter
		targetNum++;

		return true;
	}

	bool MultiTracker::update(const Mat& image)
	{
		for (int i = 0; i < trackers.size(); i++)
			if (!trackers[i]->update(image, boundingBoxes[i]))
				return false;

		return true;
	}

	//Multitracker TLD
	/*Optimized update method for TLD Multitracker */
	bool MultiTrackerTLD::update(const Mat& image)
	{
		for (int i = 0; i < trackers.size(); i++)
			if (!trackers[i]->update(image, boundingBoxes[i]))
				return false;

		return true;
	}
}
