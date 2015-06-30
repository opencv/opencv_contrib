// Ceres Solver - A fast non-linear least squares minimizer
// Copyright 2012 Google Inc. All rights reserved.
// http://code.google.com/p/ceres-solver/
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// * Redistributions of source code must retain the above copyright notice,
//   this list of conditions and the following disclaimer.
// * Redistributions in binary form must reproduce the above copyright notice,
//   this list of conditions and the following disclaimer in the documentation
//   and/or other materials provided with the distribution.
// * Neither the name of Google Inc. nor the names of its contributors may be
//   used to endorse or promote products derived from this software without
//   specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
// Author: strandmark@google.com (Petter Strandmark)
//
// Simple class for accessing PGM images.

#ifndef CERES_EXAMPLES_PGM_IMAGE_H_
#define CERES_EXAMPLES_PGM_IMAGE_H_

#include <algorithm>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "glog/logging.h"

namespace ceres {
namespace examples {

template<typename Real>
class PGMImage {
 public:
  // Create an empty image
  PGMImage(int width, int height);
  // Load an image from file
  explicit PGMImage(std::string filename);
  // Sets an image to a constant
  void Set(double constant);

  // Reading dimensions
  int width() const;
  int height() const;
  int NumPixels() const;

  // Get individual pixels
  Real* MutablePixel(int x, int y);
  Real  Pixel(int x, int y) const;
  Real* MutablePixelFromLinearIndex(int index);
  Real  PixelFromLinearIndex(int index) const;
  int LinearIndex(int x, int y) const;

  // Adds an image to another
  void operator+=(const PGMImage& image);
  // Adds a constant to an image
  void operator+=(Real a);
  // Multiplies an image by a constant
  void operator*=(Real a);

  // File access
  bool WriteToFile(std::string filename) const;
  bool ReadFromFile(std::string filename);

  // Accessing the image data directly
  bool SetData(const std::vector<Real>& new_data);
  const std::vector<Real>& data() const;

 protected:
  int height_, width_;
  std::vector<Real> data_;
};

// --- IMPLEMENTATION

template<typename Real>
PGMImage<Real>::PGMImage(int width, int height)
  : height_(height), width_(width), data_(width*height, 0.0) {
}

template<typename Real>
PGMImage<Real>::PGMImage(std::string filename) {
  if (!ReadFromFile(filename)) {
    height_ = 0;
    width_  = 0;
  }
}

template<typename Real>
void PGMImage<Real>::Set(double constant) {
  for (int i = 0; i < data_.size(); ++i) {
    data_[i] = constant;
  }
}

template<typename Real>
int PGMImage<Real>::width() const {
  return width_;
}

template<typename Real>
int PGMImage<Real>::height() const {
  return height_;
}

template<typename Real>
int PGMImage<Real>::NumPixels() const {
  return width_ * height_;
}

template<typename Real>
Real* PGMImage<Real>::MutablePixel(int x, int y) {
  return MutablePixelFromLinearIndex(LinearIndex(x, y));
}

template<typename Real>
Real PGMImage<Real>::Pixel(int x, int y) const {
  return PixelFromLinearIndex(LinearIndex(x, y));
}

template<typename Real>
Real* PGMImage<Real>::MutablePixelFromLinearIndex(int index) {
  CHECK(index >= 0);
  CHECK(index < width_ * height_);
  CHECK(index < data_.size());
  return &data_[index];
}

template<typename Real>
Real  PGMImage<Real>::PixelFromLinearIndex(int index) const {
  CHECK(index >= 0);
  CHECK(index < width_ * height_);
  CHECK(index < data_.size());
  return data_[index];
}

template<typename Real>
int PGMImage<Real>::LinearIndex(int x, int y) const {
  return x + width_*y;
}

// Adds an image to another
template<typename Real>
void PGMImage<Real>::operator+= (const PGMImage<Real>& image) {
  CHECK(data_.size() == image.data_.size());
  for (int i = 0; i < data_.size(); ++i) {
    data_[i] += image.data_[i];
  }
}

// Adds a constant to an image
template<typename Real>
void PGMImage<Real>::operator+= (Real a) {
  for (int i = 0; i < data_.size(); ++i) {
    data_[i] += a;
  }
}

// Multiplies an image by a constant
template<typename Real>
void PGMImage<Real>::operator*= (Real a) {
  for (int i = 0; i < data_.size(); ++i) {
    data_[i] *= a;
  }
}

template<typename Real>
bool PGMImage<Real>::WriteToFile(std::string filename) const {
  std::ofstream outputfile(filename.c_str());
  outputfile << "P2" << std::endl;
  outputfile << "# PGM format" << std::endl;
  outputfile << " # <width> <height> <levels> " << std::endl;
  outputfile << " # <data> ... " << std::endl;
  outputfile << width_ << ' ' << height_ << " 255 " << std::endl;

  // Write data
  int num_pixels = width_*height_;
  for (int i = 0; i < num_pixels; ++i) {
    // Convert to integer by rounding when writing file
    outputfile << static_cast<int>(data_[i] + 0.5) << ' ';
  }

  return outputfile;  // Returns true/false
}

namespace  {

// Helper function to read data from a text file, ignoring "#" comments.
template<typename T>
bool GetIgnoreComment(std::istream* in, T& t) {
  std::string word;
  bool ok;
  do {
    ok = true;
    (*in) >> word;
    if (word.length() > 0 && word[0] == '#') {
      // Comment; read the whole line
      ok = false;
      std::getline(*in, word);
    }
  } while (!ok);

  // Convert the string
  std::stringstream sin(word);
  sin >> t;

  // Check for success
  if (!in || !sin) {
    return false;
  }
  return true;
}
}  // namespace

template<typename Real>
bool PGMImage<Real>::ReadFromFile(std::string filename) {
  std::ifstream inputfile(filename.c_str());

  // File must start with "P2"
  char ch1, ch2;
  inputfile >> ch1 >> ch2;
  if (!inputfile || ch1 != 'P' || (ch2 != '2' && ch2 != '5')) {
    return false;
  }

  // Read the image header
  int two_fifty_five;
  if (!GetIgnoreComment(&inputfile, width_)  ||
      !GetIgnoreComment(&inputfile, height_) ||
      !GetIgnoreComment(&inputfile, two_fifty_five) ) {
    return false;
  }
  // Assert that the number of grey levels is 255.
  if (two_fifty_five != 255) {
    return false;
  }

  // Now read the data
  int num_pixels = width_*height_;
  data_.resize(num_pixels);
  if (ch2 == '2') {
    // Ascii file
    for (int i = 0; i < num_pixels; ++i) {
      int pixel_data;
      bool res = GetIgnoreComment(&inputfile, pixel_data);
      if (!res) {
        return false;
      }
      data_[i] = pixel_data;
    }
    // There cannot be anything else in the file (except comments). Try reading
    // another number and return failure if that succeeded.
    int temp;
    bool res = GetIgnoreComment(&inputfile, temp);
    if (res) {
      return false;
    }
  } else {
    // Read the line feed character
    if (inputfile.get() != '\n') {
      return false;
    }
    // Binary file
    // TODO(strandmark): Will not work on Windows (linebreak conversion).
    for (int i = 0; i < num_pixels; ++i) {
      unsigned char pixel_data = inputfile.get();
      if (!inputfile) {
        return false;
      }
      data_[i] = pixel_data;
    }
    // There cannot be anything else in the file. Try reading another byte
    // and return failure if that succeeded.
    inputfile.get();
    if (inputfile) {
      return false;
    }
  }

  return true;
}

template<typename Real>
bool PGMImage<Real>::SetData(const std::vector<Real>& new_data) {
  // This function cannot change the dimensions
  if (new_data.size() != data_.size()) {
    return false;
  }
  std::copy(new_data.begin(), new_data.end(), data_.begin());
  return true;
}

template<typename Real>
const std::vector<Real>& PGMImage<Real>::data() const {
  return data_;
}

}  // namespace examples
}  // namespace ceres


#endif  // CERES_EXAMPLES_PGM_IMAGE_H_
