#ifndef DATAHANDLER_H_
#define DATAHANDLER_H_
#include "layer.hpp"
#include "image_iterators.hpp"
#include <random>
#include <thread>

#include "opencv2/core.hpp"

namespace cv
{
namespace graphmodels
{

class DataIterator;

/** Makes data accessible to the model.
 * Provides a GetBatch() method that is used by the model to fetch
 * mini-batches.
 * Handles multiple streams of data.
 */ 
class CV_EXPORTS DataHandler {
 public:
  DataHandler(const config::DatasetConfig& config);
  virtual ~DataHandler();

  void GetBatch(vector<Layer*>& data_layers);
  int GetBatchSize() const { return batch_size_; }
  int GetDataSetSize() const { return dataset_size_; }
  int GetMultiplicity() const { return multiplicity_; }
  void Seek(int row);
  void Preprocess(Matrix& input, Matrix& output);
  void Sync();
  void SetFOV(const int size, const int stride, const int pad1,
              const int pad2, const int patch_size, const int num_fov_x,
              const int num_fov_y);
  void AllocateMemory();

 protected:
  void SetupShuffler();
  void ShuffleIndices();
  void LoadChunk(DataIterator& it, Matrix& mat);
  void LoadChunk(DataIterator& it, Matrix& mat, vector<int>& random_rows);
  void LoadChunkParallel(DataIterator& it, Matrix& mat);
  void LoadChunkParallel(DataIterator& it, Matrix& mat, vector<int>& random_rows);

  void PipelinedDiskAccess();
  void DiskAccess();
  void StartPreload();
  void WaitForPreload();

  default_random_engine generator_;
  uniform_int_distribution<int> * distribution_;
  map<string, DataIterator*> data_it_;
  map<string, Matrix> data_;
  vector<string> layer_names_;
  thread* preload_thread_;
  Matrix rand_perm_indices_;
  int batch_size_, chunk_size_, max_reuse_count_, reuse_counter_,
      random_access_chunk_size_, dataset_size_, start_, multiplicity_counter_;
  bool restart_, nothing_on_gpu_, fits_on_gpu_;
  const bool pipeline_loads_, randomize_cpu_, randomize_gpu_;
  const int multiplicity_;
};

/** Base class for implementing data iterators.
 * Each data iterator handles a single stream of data.
 * All derived classes must implement the GetNext() and Get() methods
 * and override the Seek() method appripriately.
 */ 
class CV_EXPORTS DataIterator {
 public:
  DataIterator(const config::DataStreamConfig& config);
  virtual ~DataIterator() {};
  virtual void GetNext(float* data_out) = 0;
  virtual void Get(float* data_out, const int row) const = 0;
  virtual void Seek(int row);
  virtual int Tell() const;
  virtual void Prep(const int chunk_size);
  virtual void Preprocess(Matrix& m);
  virtual void AddNoise(Matrix& input, Matrix& output);
  virtual void SetMaxDataSetSize(int max_dataset_size);
  virtual void SetFOV(const int size, const int stride, const int pad1,
                      const int pad2, const int patch_size,
                      const int num_fov_x, const int num_fov_y);
  virtual void SetNoiseSource(DataIterator* it);

  int GetDims() const;
  int GetDataSetSize() const;
  void AddPCANoise(Matrix& m);
  void SampleNoise(int batch_size, int dest_num_dims, int multiplicity_id);
  void Jitter(Matrix& source, Matrix& dest);
  int GetGPUId() const { return gpu_id_;}
  bool DoParallelDiskAccess() const { return parallel_disk_access_; }
  bool NeedsNoiseFromLayer() const { return !noise_layer_name_.empty(); }
  const string& GetNoiseLayerName() const { return noise_layer_name_; }
  Matrix& GetWidthOffset() { return width_offset_;}
  Matrix& GetHeightOffset() { return height_offset_;}
  Matrix& GetFlipBit() { return flip_bit_;}
  int GetDestDims() const { return dest_num_dims_; }
  int GetNumColors() const { return num_colors_; }
  static DataIterator* ChooseDataIterator(const config::DataStreamConfig& config);


 protected:
  void LoadMeans(const string& data_file);

  int num_dims_, dataset_size_, row_, dest_num_dims_;
  Matrix mean_, std_, pca_noise1_, pca_noise2_, eig_values_, eig_vectors_,
         width_offset_, height_offset_, flip_bit_;
  const string file_pattern_, noise_layer_name_;
  const int num_colors_, gpu_id_;
  const bool translate_, flip_, normalize_, pixelwise_normalize_,
             add_pca_noise_, parallel_disk_access_;
  const float pca_noise_stddev_;
  bool jitter_used_;
  DataIterator* noise_source_;
};

/** A dummy iterator.
 * Returns random numbers.
 */ 
class CV_EXPORTS DummyDataIterator : public DataIterator {
 public:
  DummyDataIterator(const config::DataStreamConfig& config);
  void GetNext(float* data_out);
  void Get(float* data_out, const int row) const;
};

/** An iterator over a dataset in an HDF5 file.
 * T specifies the type of data being iterated over.*/
template <typename T>
class CV_EXPORTS HDF5DataIterator : public DataIterator {
 public:
  HDF5DataIterator(const config::DataStreamConfig& config);
  ~HDF5DataIterator();
  void GetNext(float* data_out);
  void Get(float* data_out, const int row) const;

 protected:
  hid_t file_, dataset_, dapl_id_, m_dataspace_, type_;
  hsize_t start_[2], count_[2];
  T* buf_;
};

/** An iterator over images stored as individual files.*/
class CV_EXPORTS ImageDataIterator : public DataIterator {
 public:
  ImageDataIterator(const config::DataStreamConfig& config);
  ~ImageDataIterator();
  virtual void GetNext(float* data_out);
  virtual void Get(float* data_out, const int row) const;
  virtual void Seek(int row);
  virtual int Tell() const;
  virtual void Prep(const int chunk_size);
  virtual void SetMaxDataSetSize(int max_dataset_size);
  void RectifyBBox(box& b, int width, int height, int row) const;

 protected:
  RawImageFileIterator<unsigned char> *it_;
  unsigned char* buf_;
  const int raw_image_size_, image_size_;
};

class CV_EXPORTS CropDataIterator : public DataIterator {
 public:
  CropDataIterator(const config::DataStreamConfig& config);
  ~CropDataIterator();
  virtual void GetNext(float* data_out);
  virtual void Get(float* data_out, const int row) const;
  virtual void Seek(int row);
  virtual int Tell() const;

 protected:
  CropIterator<unsigned char> *it_;
  vector<string> file_names_;
  vector<vector<box>> crops_;
  unsigned char* buf_;
  const int image_size_;
  int file_id_;
};

/** An iterator over sliding windows of an image dataset.*/
class CV_EXPORTS SlidingWindowDataIterator : public DataIterator {
 public:
  SlidingWindowDataIterator(const config::DataStreamConfig& config);
  ~SlidingWindowDataIterator();
  virtual void GetNext(float* data_out);
  virtual void Get(float* data_out, const int row) const;
  virtual void Seek(int row);
  virtual int Tell() const;
  virtual void SetMaxDataSetSize(int max_dataset_size);
 
 protected:
  SlidingWindowIterator<unsigned char> *it_;
  unsigned char* buf_;
  vector<string> file_names_;
  const int stride_, raw_image_size_, image_size_;
  int file_id_;
};

/** An iterator over data stored in a text file.
 * Each data vector on a new line.
 * Whitespace separated.
 */
class CV_EXPORTS TextDataIterator : public DataIterator {
 public:
  TextDataIterator(const config::DataStreamConfig& config);
  ~TextDataIterator();
  virtual void GetNext(float* data_out);
  virtual void Get(float* data_out, const int row) const;

 protected:
  float* data_;
};

/** An iterator over bounding boxes.*/
class CV_EXPORTS BoundingBoxIterator : public DataIterator {
 public:
  BoundingBoxIterator(const config::DataStreamConfig& config);

  virtual void GetNext(float* data_out);
  virtual void Get(float* data_out, const int row) const;
  virtual void SetFOV(const int size, const int stride, const int pad1,
                      const int pad2, const int patch_size,
                      const int num_fov_x, const int num_fov_y);
  virtual void AddNoise(Matrix& input, Matrix& output);
  virtual void SetNoiseSource(DataIterator* it);

  static float VisibleFraction(const box& b, const box& fov);
  static float Intersection(const box& b1, const box& b2);
  static float Area(const box& b);

 protected:
  vector<vector<box>> data_;
  vector<int> img_width_, img_height_;
  int patch_size_;
  Matrix fovs_;
  vector<box> fov_box_;
  ImageDataIterator* jitter_source_;
};

}
}

#endif
