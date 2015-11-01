#ifndef DEEP_LEARNING_IMAGES_HPP
#define DEEP_LEARNING_IMAGES_HPP

#include "settings.h"
#include <sferes/phen/parameters.hpp>

using namespace sferes;

// Parameters required by Caffe separated from those introduced by Sferes
struct ParamsCaffe
{
  struct image
	{
		// Size of the square image 256x256
		SFERES_CONST int size = 256;
		SFERES_CONST int crop_size = 227;
		SFERES_CONST bool use_crops = true;

		SFERES_CONST bool color = true;	// true: color, false: grayscale images

		// GPU configurations
		SFERES_CONST bool use_gpu = false;

		// GPU on Moran can only handle max of 512 images in a batch at a time.
		SFERES_CONST int batch = 1;
		SFERES_CONST int num_categories = 1000;		// ILSVR2012 ImageNet has 1000 categories
		static int category_id;
		SFERES_CONST bool record_lineage = false;	// Flag to save the parent's assigned class
	};
};

#endif /* DEEP_LEARNING_IMAGES_HPP */
