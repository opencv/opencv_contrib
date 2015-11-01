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


#ifndef PHEN_COLOR_IMAGE_HPP
#define PHEN_COLOR_IMAGE_HPP

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
    SFERES_INDIV(ColorImage, Image)
    {
      public:
        typedef Gen gen_t;
        typedef typename gen_t::nn_t gen_nn_t;
        SFERES_CONST size_t nb_cppn_inputs = 2 + 2;
        SFERES_CONST size_t nb_cppn_outputs = 3;	// Red, Green, Blue

        ColorImage():_developed(false)
				{
				}

        void develop()
        {
        	// Check if phenotype has not been developed
        	if (!_developed)
        	{
						// Initialize the image to be a white background image
						reset_image();

						this->gen().init();
						 // develop the parameters
						BGL_FORALL_VERTICES_T(v, this->gen().get_graph(),
																	typename gen_t::nn_t::graph_t)
						{
							this->gen().get_graph()[v].get_afparams().develop();
							this->gen().get_graph()[v].get_pfparams().develop();
						}
						BGL_FORALL_EDGES_T(e, this->gen().get_graph(),
															 typename gen_t::nn_t::graph_t)
						this->gen().get_graph()[e].get_weight().develop();

						assert(nb_cppn_inputs == this->gen().get_nb_inputs());
						assert(nb_cppn_outputs == this->gen().get_nb_outputs());

						// Change specific color of every pixel in the image
						for (int x = 0; x < _image.cols; ++x)
						{
							for (int y = 0; y < _image.rows; ++y)
							{
								std::vector<float> output =	cppn_value(x, y);	// HLS array

								cv::Vec3b color = _image.at<cv::Vec3b>(cv::Point(x,y));

								color[0] = this->convert_to_color_scale(255, output[0]);	// H
								color[1] = this->convert_to_color_scale(255, output[1]);	// L
								color[2] = this->convert_to_color_scale(255, output[2]);	// S

								_image.at<cv::Vec3b>(cv::Point(x,y)) = color;
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
					// Paint background : white
					_image = cv::Mat(Params::image::size, Params::image::size, CV_8UC3, cv::Scalar(255,255,255));
				}

				double normalize_map_xy_to_grid(const int & r_xyVal, const int & r_numVoxelsXorY)
				{
					// turn the xth or yth node into its coordinates on a grid from -1 to 1, e.g. x values (1,2,3,4,5) become (-1, -.5 , 0, .5, 1)
					// this works with even numbers, and for x or y grids only 1 wide/tall, which was not the case for the original
					// e.g. see findCluster for the orignal versions where it was not a funciton and did not work with odd or 1 tall/wide #s

					double coord;

					if (r_numVoxelsXorY==1) coord = 0;
					else coord = -1 + ( r_xyVal * 2.0/(r_numVoxelsXorY-1) );

					return(coord);
				}

				std::vector<float> cppn_value(size_t i, size_t j)
        {
          // Euclidean distance from center
          const float xNormalized = normalize_map_xy_to_grid(i, Params::image::size);
					const float yNormalized = normalize_map_xy_to_grid(j, Params::image::size);
					const float distanceFromCenter = sqrt(pow(double(xNormalized),2.0)+pow(double(yNormalized),2.0));

					// CPPN inputs
          std::vector<float> in(nb_cppn_inputs);
          this->gen().init();
          in[0] = i;										// x
          in[1] = j;										// y
          in[2] = distanceFromCenter;		// distance from center
          in[3] = 1.0;									// bias

          for (size_t k = 0; k < this->gen().get_depth(); ++k)
            this->gen().step(in);

          // Get the CPPN output
          std::vector<float> out(nb_cppn_outputs);
          out[0] = this->gen().get_outf(0);		// Hue
          out[1] = this->gen().get_outf(1);		// Lightness
          out[2] = this->gen().get_outf(2);		// Saturation

          return out;
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
						cv::Mat output;

						// Convert HLS into BGR because imwrite uses BGR color space
						cv::cvtColor(map, output, CV_HLS2BGR);

//						// DEBUGGING
//						// Convert to 256x256
//						cv::Size size(256, 256);
//						resize(output, output, size);

						// Parameters for cv::imwrite
						std::vector<int> write_params;
						write_params.push_back(CV_IMWRITE_PNG_COMPRESSION);
						write_params.push_back(0);	// Fastest writing without compression

						// Write to a file
						imwrite(fileName, output, write_params);
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

        /**
				 * Returns image in BGR color space.
				 */
				void imageBGR(cv::Mat& output) {
					// Convert image to BGR before evaluating
//					cv::Mat output;

					// Convert HLS into BGR because imwrite uses BGR color space
					cv::cvtColor(_image, output, CV_HLS2BGR);
				}

				template<class Archive>
				void serialize(Archive & ar, const unsigned int version) {
					dbg::trace trace("phen", DBG_HERE);
					sferes::phen::Indiv<Gen, Fit, Params,  typename stc::FindExact<ColorImage<Gen, Fit, Params, Exact>, Exact>::ret>::serialize(ar, version);
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
