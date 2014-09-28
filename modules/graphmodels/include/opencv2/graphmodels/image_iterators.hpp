#ifndef IMAGE_ITERATORS_
#define IMAGE_ITERATORS_

#include <string>
#include <iostream>
#include <vector>
#include <random>

using namespace std;

#include <opencv2/core.hpp>

namespace cv
{
namespace graphmodels
{

typedef struct {
  float xmin, ymin, xmax, ymax;
} box;

/** An iterator over list of image files.
 * Takes a list of image files and iterates over them.
 * Images are cropped, jittered, flipped etc.
 */ 
template<typename T>
class CV_EXPORTS RawImageFileIterator {
 public:
  RawImageFileIterator(const string& filelist, const int image_size,
                       const int raw_image_size, const bool flip,
                       const bool translate, const bool random_jitter,
                       const int max_angle=0, const float min_scale=1.0);
  virtual ~RawImageFileIterator();

  // Memory must already be allocated : num_dims * num_dims * 3.
  void GetNext(T* data_ptr);
  void GetNext(T* data_ptr, const int row, const int position);
  void Get(T* data_ptr, const int row, const int position);

  void Seek(int row) { row_ = row; }
  int Tell() const { return row_; }
  void SetMaxDataSetSize(int max_dataset_size);
  virtual int GetDataSetSize() const;
  virtual void SampleNoiseDistributions(const int chunk_size);
  virtual void RectifyBBox(box& b, int width, int height, int row) const;

protected:
  virtual void LoadImageFile(const int row, cv::Mat &image);

 private:
  void Resize(cv::Mat &image) const;
  void AddRandomJitter(cv::Mat &image, int row) const;
  void ExtractRGB(cv::Mat &image, T* data_ptr, int position) const;
  void GetCoordinates(int width, int height, int position, int* left, int* top, bool* flip) const;

  default_random_engine generator_;
  uniform_real_distribution<float> * distribution_;
  vector<float> angles_, trans_x_, trans_y_, scale_;
  int dataset_size_, row_, image_id_, position_;
  vector<string> filenames_;
  cv::Mat image_;
  const int image_size_, num_positions_, raw_image_size_;
  const bool random_jitter_;
  const int max_angle_;
  const float min_scale_;
};

/** An iterator over sliding windows of an image.*/
template<typename T>
class CV_EXPORTS SlidingWindowIterator {
 public:
  SlidingWindowIterator(const int window_size, const int stride);
  void SetImage(const string& filename);
  int GetNumWindows() { return num_windows_;}
  void GetNext(T* data_ptr);
  void GetNext(T* data_ptr, int left, int top);
  bool Done();
  void Reset();

 private:
  const int window_size_, stride_;
  int num_windows_, center_x_, center_y_;
  cv::Mat image_;
  bool done_;
};

/** A Bounding-box aware image file iterator.*/
template<typename T>
class CV_EXPORTS BBoxImageFileIterator : public RawImageFileIterator<T> {
 public:
  BBoxImageFileIterator(const string& filelist, const string& bbox_file,
                        const int image_size, const int raw_image_size,
                        const bool flip, const bool translate,
                        const bool random_jitter, const int max_angle=0,
                        const float min_scale=1.0, const float context_factor=1,
                        const bool center_on_bbox=false);
  virtual ~BBoxImageFileIterator();
  
  virtual void SampleNoiseDistributions(const int chunk_size);
  virtual void RectifyBBox(box& b, int width, int height, int row) const;

protected:
  virtual void LoadImageFile(const int row, cv::Mat &image);

 private:
  void GetCropCoordinates(int row, int width, int height, int* xmin, int* xmax, int* ymin, int* ymax) const;
  default_random_engine generator_;
  uniform_real_distribution<float> * distribution_;
  vector<vector<box>> data_;
  vector<float> box_rv_;
  const float context_factor_;
  const bool center_on_bbox_;
};

/** An iterator over cropped windows of an image.*/
template<typename T>
class CV_EXPORTS CropIterator {
 public:
  CropIterator(const int image_size, const float context_factor, const bool warp_bbox);
  void SetImage(const string& filename, const vector<box>& crops);
  void GetNext(T* data_ptr);
  bool Done();
  void Reset();

 private:
  const int image_size_;
  cv::Mat image_;
  vector<box> crops_;
  bool done_;
  int index_;
  const float context_factor_;
  const bool warp_bbox_;
};

}
}

#endif
