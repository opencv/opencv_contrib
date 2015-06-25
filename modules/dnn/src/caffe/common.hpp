#ifndef CAFFE_COMMON_HPP_
#define CAFFE_COMMON_HPP_

#include <opencv2/core.hpp>
#include <iostream>

#include "glog_emulator.hpp"
//#include <gflags/gflags.h>
//#include <glog/logging.h>

#include <climits>
#include <cmath>
#include <fstream>  // NOLINT(readability/streams)
#include <iostream>  // NOLINT(readability/streams)
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <utility>  // pair
#include <vector>

namespace caffe {

// Common functions and classes from std that caffe often uses.
using std::fstream;
using std::ios;
using std::isnan;
using std::isinf;
using std::iterator;
using std::make_pair;
using std::map;
using std::ostringstream;
using std::pair;
using std::set;
using std::string;
using std::stringstream;
using std::vector;

}  // namespace caffe
#endif  // CAFFE_COMMON_HPP_
