//| This file is a part of the sferes2 framework.
//| Copyright 2009, ISIR / Universite Pierre et Marie Curie (UPMC)
//| Main contributor(s): Jean-Baptiste Mouret, mouret@isir.fr
//|
//| This software is a computer program whose purpose is to facilitate
//| experiments in evolutionary computation and evolutionary robotics.
//|
//| This software is governed by the CeCILL license under French law
//| and abiding by the rules of distribution of free software.  You
//| can use, modify and/ or redistribute the software under the terms
//| of the CeCILL license as circulated by CEA, CNRS and INRIA at the
//| following URL "http://www.cecill.info".
//|
//| As a counterpart to the access to the source code and rights to
//| copy, modify and redistribute granted by the license, users are
//| provided only with a limited warranty and the software's author,
//| the holder of the economic rights, and the successive licensors
//| have only limited liability.
//|
//| In this respect, the user's attention is drawn to the risks
//| associated with loading, using, modifying and/or developing or
//| reproducing the software by the user in light of its specific
//| status of free software, that may mean that it is complicated to
//| manipulate, and that also therefore means that it is reserved for
//| developers and experienced professionals having in-depth computer
//| knowledge. Users are therefore encouraged to load and test the
//| software's suitability as regards their requirements in conditions
//| enabling the security of their systems and/or data to be ensured
//| and, more generally, to use and operate it in the same conditions
//| as regards security.
//|
//| The fact that you are presently reading this means that you have
//| had knowledge of the CeCILL license and that you accept its terms.


#ifndef PHEN_GRAYSCALE_IMAGE_DIRECT_HPP
#define PHEN_GRAYSCALE_IMAGE_DIRECT_HPP

#include <map>
#include <modules/nn2/nn.hpp>

#include <modules/nn2/params.hpp>
#include <modules/nn2/gen_hyper_nn.hpp>


// New stuff added ------------------------------------------

#include <cmath>
#include <boost/lexical_cast.hpp>
#include <boost/algorithm/string/join.hpp>
#include <string>

#include "cvmat_serialization.h" // Serialize cv::Mat
#include <glog/logging.h>	// Google Logging

#include "phen_image.hpp" // Base Image class

// New stuff added ------------------------------------------

namespace sferes
{
  namespace phen
  {
     // hyperneat-inspired phenotype, based on a cppn
    SFERES_INDIV(GrayscaleImageDirect, Image)
    {
      public:

        GrayscaleImageDirect():_developed(false)
        {
        }

        void develop()
        {
        	// Check if phenotype has not been developed
        	if (!_developed)
        	{
						// Initialize the image to be a white background image
						reset_image();

						unsigned i = 0;	// Index to access genome

						// Change specific color of every pixel in the image
						for (int x = 0; x < _image.cols; ++x)
						{
							for (int y = 0; y < _image.rows; ++y)
							{
								float output = this->_gen.data(i);

								// Change pixel intensity of grayscale images
								// Ref: http://docs.opencv.org/doc/user_guide/ug_mat.html
								_image.at<uchar>(cv::Point(x,y)) = convert_to_color_scale(255, output);

								++i;	// Move to the next pixel location in genome
							}
						}

						_developed = true;	// Raise the flag that this phenotype has been developed.
        	}
        }

        /**
				 * Programmatically put the patterns in here.
				 */
				void reset_image()
				{
					// Paint background : black
					_image = cv::Mat(Params::image::size, Params::image::size, CV_8UC1, cv::Scalar(0, 0, 0));
				}

				/**
				 * Convert [-1, 1] range to a color scale
				 * [0, 255] for Saturation / Brightness or
				 * [0, 180] for Hue
				 */
				static int convert_to_color_scale(const int scale, const float value)
				{
					int color = value * scale;

					if (value < 0)
					{
						color *= -1;
					}

					return color;
				}

        void write_png_image(const std::string fileName, const cv::Mat& map)
				{
					// Read the target bitmap
					try
					{
						// Parameters for cv::imwrite
						std::vector<int> write_params;
						write_params.push_back(CV_IMWRITE_PNG_COMPRESSION);
						write_params.push_back(0);	// Fastest writing without compression

						// Write to a file
						imwrite(fileName, map, write_params);
					}
					catch (std::runtime_error& ex)
					{
						std::cout << "Failed to write image: " << fileName << std::endl;
						fprintf(stderr, "Exception converting image to PNG format: %s\n", ex.what());
					}
				}

        void log_best_image_fitness(const std::string title)
				{
          std::vector < std::string > list;
					list.push_back (title);
					list.push_back (".png");
					const std::string fileName = boost::algorithm::join (list, "");

					write_png_image(fileName, _image);

					std::cout << "Written to " << title << std::endl;
				}

        cv::Mat& image() {
          return _image;
        }
        const cv::Mat& image() const {
          return _image;
        }

        template<class Archive>
				void serialize(Archive & ar, const unsigned int version) {
        	dbg::trace trace("phen", DBG_HERE);
        	sferes::phen::Indiv<Gen, Fit, Params,  typename stc::FindExact<GrayscaleImageDirect<Gen, Fit, Params, Exact>, Exact>::ret>::serialize(ar, version);
					ar & BOOST_SERIALIZATION_NVP(_image);
					ar & BOOST_SERIALIZATION_NVP(_developed);
				}

      protected:
        cv::Mat _image;
        bool _developed;
    };
  }
}


#endif
