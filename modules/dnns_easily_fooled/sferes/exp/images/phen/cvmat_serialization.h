/*
 * cvmat_serialization.h
 *
 *  Created on: Jul 11, 2014
 *      Author: anh
 */

#ifndef CVMAT_SERIALIZATION_H_
#define CVMAT_SERIALIZATION_H_

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include <boost/serialization/split_free.hpp>
#include <boost/serialization/vector.hpp>

BOOST_SERIALIZATION_SPLIT_FREE(cv::Mat)
namespace boost {
  namespace serialization {

    /** Serialization support for cv::Mat */
  	template<class Archive>
    void save(Archive & ar, const cv::Mat& m, const unsigned int version)
    {
      size_t elem_size = m.elemSize();
      size_t elem_type = m.type();

      int cols = m.cols;
      int rows = m.rows;

      ar & BOOST_SERIALIZATION_NVP(cols);
      ar & BOOST_SERIALIZATION_NVP(rows);
      ar & BOOST_SERIALIZATION_NVP(elem_size);
      ar & BOOST_SERIALIZATION_NVP(elem_type);

      const size_t data_size = cols * rows * elem_size;
      ar & boost::serialization::make_array(m.ptr(), data_size);
    }

    /** Serialization support for cv::Mat */
  	template<class Archive>
    void load(Archive & ar, cv::Mat& m, const unsigned int version)
    {
      int cols, rows;
      size_t elem_size, elem_type;

      ar & BOOST_SERIALIZATION_NVP(cols);
      ar & BOOST_SERIALIZATION_NVP(rows);
      ar & BOOST_SERIALIZATION_NVP(elem_size);
      ar & BOOST_SERIALIZATION_NVP(elem_type);

      m.create(rows, cols, elem_type);

      size_t data_size = m.cols * m.rows * elem_size;
      ar & boost::serialization::make_array(m.ptr(), data_size);
    }

  }
}



#endif /* CVMAT_SERIALIZATION_H_ */
