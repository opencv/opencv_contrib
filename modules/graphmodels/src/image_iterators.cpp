#include "opencv2/graphmodels/image_iterators.hpp"
#include <fstream>
#include <sstream>
#include <iterator>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

using namespace cv;

namespace cv
{
namespace graphmodels
{

inline void resizeOCV(Mat &img, unsigned int width, unsigned int height)
{
    Mat out;
    resize(img, out, Size(width, height));
    img = out;
}

inline void rotateOCV(Mat &img, float angle)
{
    Mat rot = getRotationMatrix2D(Point2f(img.cols/2, img.rows/2), angle, 1.0);
    Mat out;
    warpAffine(img, out, rot, Size(img.cols, img.rows));
    img = out;
}

inline void cropOCV(Mat &img, Mat &out, int left, int top, int right, int bottom)
{
    Size s(right - left, bottom - top);
    Point2f center(left + s.width/2, top + s.height/2);
    getRectSubPix(img, s, center, out);
}

inline void cropOCV(Mat &img, int left, int top, int right, int bottom)
{
    Mat out;
    cropOCV(img, out, left, top, right, bottom);
    img = out;
}

inline void mirrorOCV(Mat &img)
{
    Mat out;
    flip(img, out, 0); // 0 - x, 1 - y, -1 - both
    img = out;
}

inline unsigned int spectrumOCV(Mat &img)
{
    return 3; // TODO
}

#define PI 3.14159265

#ifndef MIN
#define MIN(x,y) ((x < y) ? x : y)
#endif
#ifndef MAX
#define MAX(x,y) ((x > y) ? x : y)
#endif

template <typename T>
RawImageFileIterator<T>::RawImageFileIterator(
    const string& filelist, const int image_size, const int raw_image_size,
    const bool flip, const bool translate, const bool random_jitter,
    const int max_angle, const float min_scale) :
    row_(0), image_id_(-1), position_(0), image_size_(image_size),
    num_positions_((flip ? 2 : 1) * (translate ? 5 : 1)),
    raw_image_size_(raw_image_size), random_jitter_(random_jitter),
    max_angle_(max_angle), min_scale_(min_scale) {

  ifstream f(filelist, ios::in);
  if (!f.is_open()) {
    cerr << "Could not open data file : " << filelist << endl;
    exit(1);
  }
  while (!f.eof()) {
    string str;
    f >> str;
    if (!f.eof()) filenames_.push_back(str);
  }
  f.close();

  dataset_size_ = filenames_.size();

  distribution_ = random_jitter_ ? new uniform_real_distribution<float>(0, 1) : NULL;
}

template<typename T>
RawImageFileIterator<T>::~RawImageFileIterator() {
  if (random_jitter_) {
    delete distribution_;
  }
}

template<typename T>
int RawImageFileIterator<T>::GetDataSetSize() const {
  return num_positions_ * dataset_size_;
}

template<typename T>
void RawImageFileIterator<T>::SetMaxDataSetSize(int max_dataset_size) {
  if (max_dataset_size > 0 && dataset_size_ > max_dataset_size) {
    dataset_size_ = max_dataset_size;
  }
}

template <typename T>
void RawImageFileIterator<T>::GetNext(T* data_ptr) {
  GetNext(data_ptr, row_, position_);
  position_++;
  if (position_ == num_positions_) {
    position_ = 0;
    row_++;
    if (row_ == dataset_size_) row_ = 0;
  }
}

template <typename T>
void RawImageFileIterator<T>::GetCoordinates(
    int width, int height, int position, int* left, int* top, bool* flip) const {
  *flip = position >= 5;
  position %= 5;
  int x_slack = width - image_size_;
  int y_slack = height - image_size_;
  switch(position) {
    case 0 :  // Center. 
            *left = x_slack / 2;
            *top = y_slack / 2;
            break;
    case 1 :  // Top left.
            *left = 0;
            *top = 0;
            break;
    case 2 :  // Top right.
            *left = x_slack;
            *top = 0;
            break;
    case 3 :  // Bottom right.
            *left = x_slack;
            *top = y_slack;
            break;
    case 4 :  // Bottom left.
            *left = 0;
            *top = y_slack;
            break;
  }
}
template <typename T>
void RawImageFileIterator<T>::Resize(Mat &image) const {
  int width = image.cols, height = image.rows;
  if (width != raw_image_size_ || height != raw_image_size_) {
    int new_width, new_height;
    if (width > height) {
      new_height = raw_image_size_;
      new_width = (width * raw_image_size_) / height;
    } else {
      new_width = raw_image_size_;
      new_height = (height * raw_image_size_) / width;
    }
    resizeOCV(image, new_width, new_height);
  }
}

template<typename T>
void RawImageFileIterator<T>::SampleNoiseDistributions(const int chunk_size) {
  if (!random_jitter_) return;
  angles_.resize(chunk_size);
  trans_x_.resize(chunk_size);
  trans_y_.resize(chunk_size);
  scale_.resize(chunk_size);
  for (int i = 0; i < chunk_size; i++) {
    angles_[i] = 2 * max_angle_ * ((*distribution_)(generator_) - 0.5);
    trans_x_[i] = (*distribution_)(generator_);
    trans_y_[i] = (*distribution_)(generator_);
    scale_[i] = min_scale_ + (1 - min_scale_) * (*distribution_)(generator_);
  }
}
template<typename T>
void RawImageFileIterator<T>::RectifyBBox(box& b, int width, int height, int row) const {
  float trans_x = 0.5, trans_y = 0.5, scale = 1;
  if (random_jitter_) {
    int ind = row % angles_.size();
    //float angle = angles_[ind];
    trans_x = trans_x_[ind];
    trans_y = trans_y_[ind];
    scale = 1; //scale_[ind];
  }

  int size = (int)(scale * ((width < height) ? width : height));
  int left = (int)((width - size) * trans_x);
  int top = (int)((height - size) * trans_y);
  float resize_scale = (float)raw_image_size_ / size;
  b.xmin = (b.xmin - left) * resize_scale;
  b.ymin = (b.ymin - top) * resize_scale;
  b.xmax = (b.xmax - left) * resize_scale;
  b.ymax = (b.ymax - top) * resize_scale;
}

template<typename T>
void RawImageFileIterator<T>::AddRandomJitter(Mat &image, int row) const {
  // Add Random rotation, scale, translation.

  int ind = row % angles_.size();
  float angle = angles_[ind];
  float trans_x = trans_x_[ind];
  float trans_y = trans_y_[ind];
  float scale = scale_[ind];

  // Translation.
  int width = image.cols, height = image.rows;
  int size = (int)(scale * ((width < height) ? width : height));
  int left = (int)((width - size) * trans_x);
  int top = (int)((height - size) * trans_y);
  cropOCV(image, left, top, left + size, top + size);

  // Resize (so that after rotation, we can crop out the central raw_image_size_ * raw_image_size_ image).
  int rot_adjusted_size = (int)(raw_image_size_ * (sin(fabs(angle)*PI/180) + cos(fabs(angle)*PI/180)));
  resizeOCV(image, rot_adjusted_size, rot_adjusted_size);

  // Rotation.
  rotateOCV(image, angle);

  // Crop out the border created by rotation.
  left = image.cols / 2 - raw_image_size_ / 2;
  top = image.rows / 2 - raw_image_size_ / 2;
  cropOCV(image, left, top, left + raw_image_size_, top + raw_image_size_);
}

template <typename T>
void RawImageFileIterator<T>::LoadImageFile(const int row, Mat &image) {
  image = imread(filenames_[row].c_str());
}

template <typename T>
void RawImageFileIterator<T>::Get(T* data_ptr, const int row, const int position) {
  Mat image;
  LoadImageFile(row, image);
  if (random_jitter_) {
    AddRandomJitter(image, row);
  }
  Resize(image);
  ExtractRGB(image, data_ptr, position);
}

template <typename T>
void RawImageFileIterator<T>::GetNext(T* data_ptr, const int row, const int position) {
  if (image_id_ != row) {  // Load a new image from disk.
    image_id_ = row;
    LoadImageFile(row, image_);
    
    if (random_jitter_) {
      AddRandomJitter(image_, row);
    }
    // Resize it so that the shorter side is raw_image_size_.
    Resize(image_);
  }
  ExtractRGB(image_, data_ptr, position);
}

template<typename T>
void RawImageFileIterator<T>::ExtractRGB(Mat &image, T* data_ptr, int position) const {
  int width = image.cols, height = image.rows;
  int left = 0, top = 0;
  bool flip = false;
  GetCoordinates(width, height, position, &left, &top, &flip);

  Mat out;
  cropOCV(image, out, left, top, left + image_size_, top + image_size_);

  if (flip) mirrorOCV(out); // TODO: test

  int num_image_colors = spectrumOCV(out);
  int num_pixels = image_size_ * image_size_;
  if (num_image_colors >= 3) {  // Image has 3 channels.
    memcpy(data_ptr, out.data, 3 * num_pixels * sizeof(T));
  } else if (num_image_colors == 1) {  // Image has 1 channel.
    for (int i = 0; i < 3; i++) {
      memcpy(data_ptr + i * num_pixels, out.data, num_pixels * sizeof(T));
    }
  } else {
    cerr << "Image has " << num_image_colors << "colors." << endl;
    exit(1);
  }
}
template class RawImageFileIterator<float>;
template class RawImageFileIterator<unsigned char>;


template <typename T>
SlidingWindowIterator<T>::SlidingWindowIterator(const int window_size, const int stride):
  window_size_(window_size), stride_(stride), num_windows_(0),
  center_x_(0), center_y_(0), done_(true) {}

template <typename T>
void SlidingWindowIterator<T>::SetImage(const string& filename) {
  image_ = imread(filename.c_str());
  center_x_ = 0;
  center_y_ = 0;
  int num_modules_x = (image_.cols - window_size_ % 2) / stride_ + 1;
  int num_modules_y = (image_.rows - window_size_ % 2) / stride_ + 1;
  num_windows_ = num_modules_x * num_modules_y;
  done_ = false;
}

template <typename T>
void SlidingWindowIterator<T>::Reset() {
  done_ = true;
}

template <typename T>
void SlidingWindowIterator<T>::GetNext(T* data_ptr) {
  GetNext(data_ptr, center_x_, center_y_);
  center_x_ += stride_;
  if (center_x_ >= image_.cols) {
    center_x_ = 0;
    center_y_ += stride_;
    if (center_y_ >= image_.rows) {
      center_y_ = 0;
      done_ = true;
    }
  }
}

template <typename T>
bool SlidingWindowIterator<T>::Done() {
  return done_;
}

template <typename T>
void SlidingWindowIterator<T>::GetNext(T* data_ptr, int center_x, int center_y) {
  int left    = center_x - window_size_ / 2,
      right   = left + window_size_,
      top     = center_y - window_size_ / 2,
      bottom  = top + window_size_;

  Mat out;
  cropOCV(image_, out, left, top, right, bottom);

  int num_pixels = window_size_ * window_size_ * 3;
  memcpy(data_ptr, out.data, num_pixels * sizeof(float));
}

template class SlidingWindowIterator<float>;
template class SlidingWindowIterator<unsigned char>;

template <typename T>
BBoxImageFileIterator<T>::BBoxImageFileIterator(
  const string& filelist, const string& bbox_file, const int image_size,
  const int raw_image_size, const bool flip, const bool translate,
  const bool random_jitter, const int max_angle, const float min_scale,
  const float context_factor, const bool center_on_bbox) :
  RawImageFileIterator<T>(filelist, image_size, raw_image_size, flip, translate, random_jitter, max_angle, min_scale),
  context_factor_(context_factor), center_on_bbox_(center_on_bbox) {
  ifstream f(bbox_file, ios::in);
  string line;
  // The format for each line is -
  // <width> <height> <xmin1> <ymin1> <xmax1> <ymax1> <xmin2> <ymin2> ...
  while (getline(f, line)) {
    istringstream iss(line);
    vector<string> tokens;
    copy(istream_iterator<string>(iss), istream_iterator<string>(),
         back_inserter<vector<string> >(tokens));
    int num_tokens = tokens.size();
    if (num_tokens <= 2 || (num_tokens - 2) % 4 != 0) {
      cerr << "Error parsing line " << line << endl;
      exit(1);
    }
    int num_boxes = (num_tokens - 2) / 4;
    vector<box> b_list (num_boxes);
    for (int i = 0; i < num_boxes; i++) {
      b_list[i].xmin = atoi(tokens[2+4*i  ].c_str());
      b_list[i].ymin = atoi(tokens[2+4*i+1].c_str());
      b_list[i].xmax = atoi(tokens[2+4*i+2].c_str());
      b_list[i].ymax = atoi(tokens[2+4*i+3].c_str());
    }
    data_.push_back(b_list);
  }
  f.close();
  distribution_ = new uniform_real_distribution<float>(0, 1);
}

template<typename T>
BBoxImageFileIterator<T>::~BBoxImageFileIterator() {
  delete distribution_;
}

template<typename T>
void BBoxImageFileIterator<T>::SampleNoiseDistributions(const int chunk_size) {
  RawImageFileIterator<T>::SampleNoiseDistributions(chunk_size);
  box_rv_.resize(chunk_size);
  for (int i = 0; i < chunk_size; i++) {
    box_rv_[i] = (*distribution_)(generator_);
  }
}

template <typename T>
void BBoxImageFileIterator<T>::GetCropCoordinates(int row, int width, int height, int* xmin, int* xmax, int* ymin, int* ymax) const {
  int box_id = (int)(box_rv_[row % box_rv_.size()] * data_[row].size());  // Choose a random box.
  const box& b = data_[row][box_id];
  int box_width = b.xmax - b.xmin;
  int box_height = b.ymax - b.ymin;
  int image_size = MAX(box_width, box_height) * context_factor_;

  int width_slack = (image_size - box_width) / 2;
  if (center_on_bbox_) {
    width_slack = MIN(width_slack, b.xmin);
    width_slack = MIN(width_slack, width - b.xmax);
  }

  int height_slack = (image_size - box_height) / 2;
  if (center_on_bbox_) {
    height_slack = MIN(height_slack, b.ymin);
    height_slack = MIN(height_slack, height - b.ymax);
  }

  *xmin = MAX(0, b.xmin - width_slack);
  *ymin = MAX(0, b.ymin - height_slack);
  *xmax = MIN(width, b.xmax + width_slack);
  *ymax = MIN(height, b.ymax + height_slack);
}

/** Crop the bounding box region. */
template <typename T>
void BBoxImageFileIterator<T>::LoadImageFile(const int row, Mat &image) {
  RawImageFileIterator<T>::LoadImageFile(row, image);
  int xmin, ymin, xmax, ymax;
  GetCropCoordinates(row, image.cols, image.rows, &xmin, &xmax, &ymin, &ymax);
  cropOCV(image, xmin, ymin, xmax, ymax);
}


template<typename T>
void BBoxImageFileIterator<T>::RectifyBBox(box& b, int width, int height, int row) const {
  int xmin, ymin, xmax, ymax;
  GetCropCoordinates(row, width, height, &xmin, &xmax, &ymin, &ymax);
  b.xmin -= xmin; 
  b.xmax -= xmin; 
  b.ymin -= ymin; 
  b.ymax -= ymin; 
  RawImageFileIterator<T>::RectifyBBox(b, xmax - xmin, ymax - ymin, row);
}

template class BBoxImageFileIterator<float>;
template class BBoxImageFileIterator<unsigned char>;


template <typename T>
CropIterator<T>::CropIterator(const int image_size, const float context_factor, const bool warp_bbox):
  image_size_(image_size), done_(true), index_(0), context_factor_(context_factor), warp_bbox_(warp_bbox) {}

template <typename T>
void CropIterator<T>::SetImage(const string& filename, const vector<box>& crops) {
  image_ = imread(filename.c_str());
  crops_ = crops;
  done_ = false;
  index_ = 0;
}

template <typename T>
void CropIterator<T>::Reset() {
  done_ = true;
}

template <typename T>
bool CropIterator<T>::Done() {
  return done_;
}

template <typename T>
void CropIterator<T>::GetNext(T* data_ptr) {
  const box& b = crops_[index_++];

  int left, top, right, bottom;

  if (warp_bbox_) {
    left = b.xmin;
    top = b.ymin;
    right = b.xmax;
    bottom = b.ymax;
    left = MAX(0, left);
    top  = MAX(0, top);
    right = MIN(image_.cols, right);
    bottom = MIN(image_.rows, bottom);
  } else {
    int width = b.xmax - b.xmin;
    int height = b.ymax - b.ymin;
    int size = int(MAX(width, height) * context_factor_);
    int width_slack = size - width;
    int height_slack = size - height;
    left = b.xmin - width_slack / 2;
    top  = b.ymin - height_slack / 2;
    right = b.xmin + size;
    bottom = b.ymin + size;
  }
  /*
  int left = MAX(0, b.xmin - width_slack / 2);
  int top  = MAX(0, b.ymin - height_slack / 2);
  int right = MIN(image.cols, b.xmax + width_slack - width_slack / 2);
  int bottom = MIN(image.rows, b.ymax + height_slack - height_slack / 2);
  */

  Mat out;
  cropOCV(image_, out, left, top, right, bottom);

  resizeOCV(out, image_size_, image_size_);

  int num_image_colors = spectrumOCV(out);
  int num_pixels = image_size_ * image_size_;
  if (num_image_colors >= 3) {  // Image has 3 channels.
    memcpy(data_ptr, out.data, 3 * num_pixels * sizeof(T));
  } else if (num_image_colors == 1) {  // Image has 1 channel.
    for (int i = 0; i < 3; i++) {
      memcpy(data_ptr + i * num_pixels, out.data, num_pixels * sizeof(T));
    }
  } else {
    cerr << "Image has " << num_image_colors << "colors." << endl;
    exit(1);
  }
  if (index_ == crops_.size()) {
    done_ = true;
    index_ = 0;
  }
}

template class CropIterator<float>;
template class CropIterator<unsigned char>;

}
}
