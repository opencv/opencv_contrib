#include "opencv2/graphmodels/datahandler.hpp"
#include "opencv2/graphmodels/image_iterators.hpp"
#include <algorithm>
#include <iostream>
#include <fstream>
#include <sstream>

namespace cv
{
namespace graphmodels
{

DataHandler::DataHandler(const config::DatasetConfig& config) :
  preload_thread_(NULL),
  batch_size_(config.batch_size()),
  chunk_size_(config.chunk_size()),
  max_reuse_count_(config.max_reuse_count()),
  reuse_counter_(0),
  random_access_chunk_size_(config.random_access_chunk_size()),
  dataset_size_(-1),
  start_(0),
  multiplicity_counter_(0),
  restart_(true),
  nothing_on_gpu_(true),
  fits_on_gpu_(false),
  pipeline_loads_(config.pipeline_loads()),
  randomize_cpu_(config.randomize_cpu()),
  randomize_gpu_(config.randomize_gpu()),
  multiplicity_(config.multiplicity()) {

  // Create data streams.
  for (const config::DataStreamConfig& dsc:config.data_config()) {
    const string& layer_name = dsc.layer_name();
    layer_names_.push_back(layer_name);
    data_it_[layer_name] = DataIterator::ChooseDataIterator(dsc);
    data_it_[layer_name]->SetMaxDataSetSize(config.max_dataset_size());
    int dataset_size = data_it_[layer_name]->GetDataSetSize();
    if (dataset_size_ == -1) {
      dataset_size_ = dataset_size;
    } else {
      if (dataset_size != dataset_size_) {
        cerr << "All data streams must have the same size." << endl;
        exit(1);
      }
    }
  }
  if (chunk_size_ <= 0 || chunk_size_ > dataset_size_) {
    chunk_size_ = dataset_size_;
    fits_on_gpu_ = true;
  }
  // GPU shuffler.
  if (randomize_gpu_) {
    SetupShuffler();
  }
  if (randomize_cpu_) {
    distribution_ = new uniform_int_distribution<int>(0, dataset_size_ - 1);
  }
  Seek(0);
    
  for (const string& layer_name : layer_names_) {
    DataIterator* it = data_it_[layer_name];
    if (it == NULL) continue;
    if (it->NeedsNoiseFromLayer()) {
      it->SetNoiseSource(data_it_[it->GetNoiseLayerName()]);
    }
  }
}

void DataHandler::AllocateMemory() {
  for (const string& layer_name : layer_names_) {
    DataIterator* it = data_it_[layer_name];
    if (it == NULL) continue;
    int num_dims = it->GetDims();
    Matrix::SetDevice(it->GetGPUId());
    Matrix& data = data_[layer_name];
    data.AllocateGPUMemory(num_dims, chunk_size_);
  }
}

DataHandler::~DataHandler() {
  for (auto it : data_it_) {
    delete it.second;
  }
}

void DataHandler::SetupShuffler() {
  rand_perm_indices_.AllocateGPUMemory(1, chunk_size_);
  float* cpu_rand_perm_indices = rand_perm_indices_.GetHostData();
  for (int i = 0; i < chunk_size_; i++) {
    cpu_rand_perm_indices[i] = i;
  }
  rand_perm_indices_.CopyToDevice();
}

void DataHandler::Seek(int row) {
  Sync();
  start_ = row;
  reuse_counter_ = 0;
  multiplicity_counter_ = 0;
  restart_ = true;
  for (auto it : data_it_) {
    it.second->Seek(row);
  }
}

void DataHandler::Sync() {
  if (pipeline_loads_) WaitForPreload();
}

void DataHandler::ShuffleIndices() {
  float* cpu_rand_perm_indices = rand_perm_indices_.GetHostData();
  const int dataset_size = rand_perm_indices_.GetCols();
  random_shuffle(cpu_rand_perm_indices, cpu_rand_perm_indices + dataset_size);
  rand_perm_indices_.CopyToDevice();
}

void DataHandler::GetBatch(vector<Layer*>& data_layers) {
  int end = start_ + batch_size_;
  if (end > chunk_size_ || restart_) {
    if (reuse_counter_ < max_reuse_count_ && !restart_) {
      reuse_counter_++;
    } else if (nothing_on_gpu_ || !fits_on_gpu_) {
      if (restart_ && pipeline_loads_) StartPreload();
      nothing_on_gpu_ = false;
      reuse_counter_ = 0;
      PipelinedDiskAccess();
    }
    restart_ = false;
    if (randomize_gpu_) {
      ShuffleIndices();
      for (Layer* l : data_layers) {
        const string& layer_name = l->GetName();
        Matrix& data = data_[layer_name];
        data.ShuffleColumns(rand_perm_indices_);
      }
    }
    start_ = 0;
    end = batch_size_;
  }

  // Sample jitter/noise.
  // This is done for all layers first because some layers may need the noise
  // from other layers to set their data. For example, if jitter is added to
  // the input, the bounding boxes in the output need to change accordingly.
  for (Layer* l : data_layers) {
    const string& layer_name = l->GetName();
    DataIterator* it = data_it_[layer_name];
    if (it == NULL) continue;
    Matrix::SetDevice(it->GetGPUId());
    Matrix& dest = l->IsInput() ? l->GetState() : l->GetData();
    it->SampleNoise(dest.GetRows(), dest.GetCols(), multiplicity_counter_);
  }

  for (Layer* l : data_layers) {
    const string& layer_name = l->GetName();
    DataIterator* it = data_it_[layer_name];
    if (it == NULL) continue;
    Matrix::SetDevice(it->GetGPUId());
    Matrix data_slice;
    data_[layer_name].GetSlice(data_slice, start_, end);
    Matrix& dest = l->IsInput() ? l->GetState() : l->GetData();
    
    // Add noise, if asked for, and copy to dest.
    it->AddNoise(data_slice, dest);
  }
  multiplicity_counter_++;
  if (multiplicity_counter_ == multiplicity_) {
    multiplicity_counter_ = 0;
    start_ = end;
  }
}

void DataHandler::PipelinedDiskAccess() {
  if (pipeline_loads_) {
    WaitForPreload();
  } else {
    DiskAccess();
  }

  for (const string& layer_name: layer_names_) {
    Matrix& data = data_[layer_name];
    DataIterator* it = data_it_[layer_name];
    if (it == NULL) continue;
    Matrix::SetDevice(it->GetGPUId());
    data.CopyToDevice();
    it->Preprocess(data);  // Does centering, normalization etc.
  }

  if (pipeline_loads_) {
    StartPreload();
  }
}

void DataHandler::StartPreload() {
  preload_thread_ = new thread(&DataHandler::DiskAccess, this);
}

void DataHandler::WaitForPreload() {
  if (preload_thread_ != NULL) {
    preload_thread_->join();
    delete preload_thread_;
    preload_thread_ = NULL;
  }
}

void DataHandler::DiskAccess() {
  vector<int> random_rows;
  if (randomize_cpu_) {
    for (int i = 0; i < chunk_size_; i += random_access_chunk_size_) {
      random_rows.push_back((*distribution_)(generator_));
    }
  }
  for (const string& layer_name: layer_names_) {
    DataIterator* it = data_it_[layer_name];
    if (it == NULL) continue;
    it->Prep(chunk_size_);
  }
  for (const string& layer_name: layer_names_) {
    DataIterator* it = data_it_[layer_name];
    if (it == NULL) continue;
    Matrix& data = data_[layer_name];
    if (randomize_cpu_) {
      if (it->DoParallelDiskAccess()) {
        LoadChunkParallel(*it, data, random_rows);
      } else {
        LoadChunk(*it, data, random_rows);
      }
    } else {
      if (it->DoParallelDiskAccess()) {
        LoadChunkParallel(*it, data);
      } else {
        LoadChunk(*it, data);
      }
    }
  }
}

void DataHandler::LoadChunk(DataIterator& it, Matrix& mat) {
  float* data_ptr = mat.GetHostData();
  int num_dims = it.GetDims();
  for (int i = 0; i < chunk_size_; i++) {
    it.GetNext(data_ptr);
    data_ptr += num_dims;
  }
}

void DataHandler::LoadChunk(DataIterator& it, Matrix& mat, vector<int>& random_rows) {
  float* data_ptr = mat.GetHostData();
  int num_dims = it.GetDims();
  int j = 0;
  for (int i = 0; i < chunk_size_; i++) {
    if (i % random_access_chunk_size_ == 0) {
      it.Seek(random_rows[j++]);
    } 
    it.GetNext(data_ptr);
    data_ptr += num_dims;
  }
}

void DataHandler::LoadChunkParallel(DataIterator& it, Matrix& mat) {
  float* data_ptr = mat.GetHostData();
  int num_dims = it.GetDims();
  int row = it.Tell();
  #pragma omp parallel for
  for (int i = 0; i < chunk_size_; i++) {
    it.Get(data_ptr + i * num_dims, (row + i) % dataset_size_);
  }
  it.Seek((row + chunk_size_) % dataset_size_);
}


void DataHandler::LoadChunkParallel(DataIterator& it, Matrix& mat, vector<int>& random_rows) {
  float* data_ptr = mat.GetHostData();
  int num_dims = it.GetDims();
  #pragma omp parallel for
  for (int i = 0; i < chunk_size_; i++) {
    int row = (random_rows[i / random_access_chunk_size_] + i % random_access_chunk_size_) % dataset_size_;
    it.Get(data_ptr + i * num_dims, row);
  }
}

void DataHandler::SetFOV(const int size, const int stride,
                         const int pad1, const int pad2,
                         const int patch_size,
                         const int num_fov_x, const int num_fov_y) {

  for (const string& layer_name : layer_names_) {
    DataIterator* it = data_it_[layer_name];
    if (it == NULL) continue;
    it->SetFOV(size, stride, pad1, pad2, patch_size, num_fov_x, num_fov_y);
  }
}

DataIterator* DataIterator::ChooseDataIterator(const config::DataStreamConfig& config) {
  DataIterator* it = NULL;
  hid_t file, dataset, datatype;
  int size;
  switch (config.data_type()) {
    case config::DataStreamConfig::DUMMY:
      it = new DummyDataIterator(config);
      break;
    case config::DataStreamConfig::HDF5:
      file = H5Fopen(config.file_pattern().c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
      dataset = H5Dopen(file, config.dataset_name().c_str(), H5P_DEFAULT);
      datatype = H5Dget_type(dataset);
      size = H5Tget_size(datatype);
      if (H5Tget_class(datatype) == H5T_INTEGER) { // integer
        if (H5Tget_sign(datatype) == H5T_SGN_2) {  // signed integers.
          switch (size) {
            case 1: it = new HDF5DataIterator<char>(config); break;
            case 4: it = new HDF5DataIterator<int>(config); break;
            case 8: it = new HDF5DataIterator<long>(config); break;
          }
        } else {  // unsigned integers.
          switch (size) {
            case 1: it = new HDF5DataIterator<unsigned char>(config); break;
            case 4: it = new HDF5DataIterator<unsigned int>(config); break;
            case 8: it = new HDF5DataIterator<unsigned long>(config); break;
          }
        }
      } else {  // floating-point.
        switch (size) {
          case 4: it = new HDF5DataIterator<float>(config); break;
          case 8: it = new HDF5DataIterator<double>(config); break;
        }
      }
      H5Tclose(datatype);
      H5Dclose(dataset);
      H5Fclose(file);
      break;
    case config::DataStreamConfig::IMAGE_RAW:
      it = new ImageDataIterator(config);
      break;
    case config::DataStreamConfig::SLIDING_WINDOW:
      it = new SlidingWindowDataIterator(config);
      break;
    case config::DataStreamConfig::CROPS:
      it = new CropDataIterator(config);
      break;
    case config::DataStreamConfig::TXT:
      it = new TextDataIterator(config);
      break;
    case config::DataStreamConfig::BOUNDING_BOX:
      it = new BoundingBoxIterator(config);
      break;
    default:
      cerr << "Unknown data type " << (int)config.data_type() << endl;
      exit(1);
  }
  return it;
}

DataIterator::DataIterator(const config::DataStreamConfig& config):
  num_dims_(0), dataset_size_(0), row_(0),
  file_pattern_(config.file_pattern()),
  noise_layer_name_(config.noise_layer_name()),
  num_colors_(config.num_colors()),
  gpu_id_(config.gpu_id()),
  translate_(config.can_translate()),
  flip_(config.can_flip()),
  normalize_(config.normalize() || config.pixelwise_normalize()),
  pixelwise_normalize_(config.pixelwise_normalize()),
  add_pca_noise_(config.pca_noise_stddev() > 0),
  parallel_disk_access_(config.parallel_disk_access()),
  pca_noise_stddev_(config.pca_noise_stddev()),
  jitter_used_(true),
  noise_source_(NULL) {}

void DataIterator::SetMaxDataSetSize(int max_dataset_size) {
  if (max_dataset_size > 0 && dataset_size_ > max_dataset_size) {
    dataset_size_ = max_dataset_size;
  }
}

void DataIterator::SetNoiseSource(DataIterator* it) {
  noise_source_ = it;
}

void DataIterator::LoadMeans(const string& data_file) {
  hid_t file = H5Fopen(data_file.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
  if (pixelwise_normalize_) {
    hid_t file = H5Fopen(data_file.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    Matrix pixel_mean, pixel_std;
    pixel_mean.AllocateAndReadHDF5(file, "pixel_mean");
    pixel_std.AllocateAndReadHDF5(file, "pixel_std");
    pixel_mean.Reshape(1, -1);
    pixel_std.Reshape(1, -1);
    int num_channels = pixel_mean.GetCols();
    int num_pixels = num_dims_ / num_channels;
    mean_.AllocateGPUMemory(num_pixels, num_channels);
    std_.AllocateGPUMemory(num_pixels, num_channels);

    mean_.AddRowVec(pixel_mean);
    std_.AddRowVec(pixel_std);

    mean_.Reshape(-1, 1);
    std_.Reshape(-1, 1);
    if (add_pca_noise_) {
      eig_values_.AllocateAndReadHDF5(file, "S");
      eig_vectors_.AllocateAndReadHDF5(file, "U");
    }
  }
  else {
    mean_.AllocateAndReadHDF5(file, "mean");
    std_.AllocateAndReadHDF5(file, "std");
  }
  H5Fclose(file);
}

int DataIterator::GetDims() const {
  return num_dims_;
}

int DataIterator::GetDataSetSize() const {
  return dataset_size_;
}

void DataIterator::Seek(int row) {
  row_ = row;
}

int DataIterator::Tell() const {
  return row_;
}

void DataIterator::Prep(const int chunk_size) {
}

// m is on the GPU, stored so that different cases are far apart.
void DataIterator::Preprocess(Matrix& m) {
  if (normalize_) {
    m.AddColVec(mean_, -1);
    m.DivideByColVec(std_);
  }
}

void DataIterator::AddPCANoise(Matrix& m) {
  if (pca_noise1_.GetRows() != m.GetRows()) {
    pca_noise1_.AllocateGPUMemory(m.GetRows(), eig_vectors_.GetRows());
    pca_noise2_.AllocateGPUMemory(m.GetRows(), eig_vectors_.GetCols());
  }
  pca_noise1_.FillWithRandn();
  pca_noise1_.MultByRowVec(eig_values_);
  Matrix::Dot(pca_noise1_, eig_vectors_, pca_noise2_, 0, 1);
  m.AddToEachPixel(pca_noise2_, pca_noise_stddev_);
}


void DataIterator::AddNoise(Matrix& input, Matrix& output) {
  if (flip_ || translate_ || (output.GetCols() <  input.GetRows())) {
    Jitter(input, output);
  } else {
    input.CopyTranspose(output);
  }
  if (add_pca_noise_) {
    AddPCANoise(output);
  }
}

void DataIterator::SampleNoise(int batch_size, int dest_num_dims, int multiplicity_id) {
  if (dest_num_dims != num_dims_ || flip_) {
    int patch_size = (int)sqrt(dest_num_dims / num_colors_);
    int image_size = (int)sqrt(num_dims_ / num_colors_);
    int max_offset = image_size - patch_size;
    dest_num_dims_ = dest_num_dims;

    if (width_offset_.GetCols() != batch_size) {
      width_offset_.AllocateGPUMemory(1, batch_size);
      height_offset_.AllocateGPUMemory(1, batch_size);
      flip_bit_.AllocateGPUMemory(1, batch_size);
    }

    if (translate_) {  // Random jitter.
      width_offset_.FillWithRand();
      height_offset_.FillWithRand();
      width_offset_.Mult(max_offset + 1);  // Rounded down.
      height_offset_.Mult(max_offset + 1);  // Rounded down.
    } else {  // Take center or corner patch.
      int w = 0, h = 0;
      switch (multiplicity_id % 5) {
        case 0: w = max_offset/2; h = max_offset/2; break;
        case 1: w = 0; h = 0; break;
        case 2: w = max_offset; h = 0; break;
        case 3: w = max_offset; h = max_offset; break;
        case 4: w = 0; h = max_offset; break;
      }
      width_offset_.Set(w);
      height_offset_.Set(h);
    }
    if (flip_) {
      flip_bit_.FillWithRand();  // flip if > 0.5.
    } else {
      flip_bit_.Set(multiplicity_id/5);
    }
  }
}

void DataIterator::Jitter(Matrix& source, Matrix& dest) {
  int patch_size = (int)sqrt(dest.GetCols() / num_colors_);
  int image_size = (int)sqrt(source.GetRows() / num_colors_);

  // Extract jittered images.
  Matrix::ExtractPatches(source, dest, width_offset_, height_offset_, flip_bit_, image_size, image_size, patch_size, patch_size);
}

void DataIterator::SetFOV(const int size, const int stride,
                          const int pad1, const int pad2, const int patch_size,
                          const int num_fov_x, const int num_fov_y) {
  // no op.
}

DummyDataIterator::DummyDataIterator(const config::DataStreamConfig& config):
  DataIterator(config) {
    num_dims_ = 100;
    dataset_size_ = 100000;
  if (normalize_) {
    LoadMeans(config.mean_file());
  }
}

void DummyDataIterator::Get(float* data_out, const int row) const {
  for (int i = 0; i < num_dims_; i++) {
    data_out[i] = rand();
  }
}

void DummyDataIterator::GetNext(float* data_out) {
  Get(data_out, 0);
}

template <typename T>
HDF5DataIterator<T>::HDF5DataIterator(const config::DataStreamConfig& config):
  DataIterator(config) {
  
  // Open the hdf5 file and dataset.
  file_ = H5Fopen(file_pattern_.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
  size_t cache_size = 1024 * 1024 * 1024;
  dapl_id_ = H5Pcreate(H5P_DATASET_ACCESS);
  H5Pset_chunk_cache(dapl_id_, H5D_CHUNK_CACHE_NSLOTS_DEFAULT, cache_size, H5D_CHUNK_CACHE_W0_DEFAULT);
  dataset_ = H5Dopen(file_, config.dataset_name().c_str(), dapl_id_);
  hid_t dataspace = H5Dget_space(dataset_);
  hsize_t dims_out[2];
  const int ndims = H5Sget_simple_extent_ndims(dataspace);
  H5Sget_simple_extent_dims(dataspace, dims_out, NULL);
  dataset_size_ = dims_out[0];
  num_dims_ = (ndims == 1) ? 1: dims_out[1];
  start_[0] = 0;
  start_[1] = 0;
  count_[0] = 1;
  count_[1] = num_dims_;
  H5Sclose(dataspace);
  m_dataspace_ = H5Screate_simple(2, count_, NULL);
  type_ = H5Dget_type(dataset_);

  buf_ = new T[num_dims_];

  /*
   * Check if gzip compression is available and can be used for both
   * compression and decompression.
   */
  htri_t avail = H5Zfilter_avail(H5Z_FILTER_DEFLATE);
  if (!avail) {
    printf ("gzip filter not available.\n");
  }
  unsigned int filter_info;
  H5Zget_filter_info (H5Z_FILTER_DEFLATE, &filter_info);
  if ( !(filter_info & H5Z_FILTER_CONFIG_ENCODE_ENABLED) ||
      !(filter_info & H5Z_FILTER_CONFIG_DECODE_ENABLED) ) {
    printf ("gzip filter not available for encoding and decoding.\n");
  }
  if (normalize_) {
    LoadMeans(config.mean_file());
  }
}

template <typename T>
HDF5DataIterator<T>::~HDF5DataIterator() {
  delete[] buf_;
  H5Tclose(type_);
  H5Sclose(m_dataspace_);
  H5Pclose(dapl_id_);
  H5Dclose(dataset_);
  H5Fclose(file_);
}

template <typename T>
void HDF5DataIterator<T>::Get(float* data_ptr, const int row) const {
  hsize_t start[2];
  start[0] = row;
  start[1] = 0;
  T* buf = new T[num_dims_];
  hid_t f_dataspace = H5Dget_space(dataset_);
  H5Sselect_hyperslab(f_dataspace, H5S_SELECT_SET, start, NULL, count_, NULL);
  H5Dread(dataset_, type_, m_dataspace_, f_dataspace, H5P_DEFAULT, buf);
  H5Sclose(f_dataspace);

  // Copy and type-cast from buf_ to data_ptr.
  for (int i = 0; i < num_dims_; i++) {
    data_ptr[i] = static_cast<float>(buf[i]);
  }
  delete[] buf;
}

template <typename T>
void HDF5DataIterator<T>::GetNext(float* data_ptr) {
  start_[0] = row_;
  hid_t f_dataspace = H5Dget_space(dataset_);
  H5Sselect_hyperslab(f_dataspace, H5S_SELECT_SET, start_, NULL, count_, NULL);
  H5Dread(dataset_, type_, m_dataspace_, f_dataspace, H5P_DEFAULT, buf_);
  H5Sclose(f_dataspace);

  // Copy and type-cast from buf_ to data_ptr.
  for (int i = 0; i < num_dims_; i++) {
    data_ptr[i] = static_cast<float>(buf_[i]);
  }

  row_++;
  if (row_ == dataset_size_) row_ = 0;
}

ImageDataIterator::ImageDataIterator(const config::DataStreamConfig& config):
  DataIterator(config),
  raw_image_size_(config.raw_image_size()),
  image_size_(config.image_size()) {
  
  bool avg10_full_image = config.avg10_full_image();
 
  if (config.bbox_file().empty()) {
    it_ = new RawImageFileIterator<unsigned char>(
        file_pattern_, image_size_, raw_image_size_, avg10_full_image, avg10_full_image,
        config.random_rotate_raw_image(), config.random_rotate_max_angle(),
        config.min_scale());
  } else {
    it_ = new BBoxImageFileIterator<unsigned char>(
        file_pattern_, config.bbox_file(), image_size_, raw_image_size_, avg10_full_image, avg10_full_image,
        config.random_rotate_raw_image(), config.random_rotate_max_angle(),
        config.min_scale(), config.context_factor(), config.center_on_bbox());
  }

  dataset_size_ = it_->GetDataSetSize();
  num_dims_ = image_size_ * image_size_ * num_colors_;
  buf_ = new unsigned char[num_dims_];
  if (normalize_) {
    LoadMeans(config.mean_file());
  }
}

ImageDataIterator::~ImageDataIterator() {
  delete[] buf_;
  delete it_;
}

void ImageDataIterator::Seek(int row) {
  it_->Seek(row);
}

int ImageDataIterator::Tell() const {
  return it_->Tell();
}

void ImageDataIterator::SetMaxDataSetSize(int max_dataset_size) {
  it_->SetMaxDataSetSize(max_dataset_size);
  dataset_size_ = it_->GetDataSetSize();
}

void ImageDataIterator::Prep(const int chunk_size) {
  it_->SampleNoiseDistributions(chunk_size);
}

void ImageDataIterator::RectifyBBox(box& b, int width, int height, int row) const {
  it_->RectifyBBox(b, width, height, row);
}

void ImageDataIterator::GetNext(float* data_out) {
  it_->GetNext(buf_);
  for (int i = 0; i < image_size_ * image_size_ * num_colors_; i++) {
    data_out[i] = static_cast<float>(buf_[i]);
  }
}

void ImageDataIterator::Get(float* data_out, const int row) const {
  unsigned char* buf = new unsigned char[num_dims_];
  it_->Get(buf, row, 0);
  for (int i = 0; i < image_size_ * image_size_ * num_colors_; i++) {
    data_out[i] = static_cast<float>(buf[i]);
  }
  delete[] buf;
}

SlidingWindowDataIterator::SlidingWindowDataIterator(
    const config::DataStreamConfig& config):
  DataIterator(config), stride_(config.stride()),
  raw_image_size_(config.raw_image_size()),
  image_size_(config.image_size()),
  file_id_(0) {
  num_dims_ = image_size_ * image_size_ * num_colors_;
  it_ = new SlidingWindowIterator<unsigned char>(image_size_, stride_);
  buf_ = new unsigned char[num_dims_];
  ReadLines(file_pattern_, file_names_);
  dataset_size_ = file_names_.size() * it_->GetNumWindows();
  if (normalize_) {
    LoadMeans(config.mean_file());
  }
}

SlidingWindowDataIterator::~SlidingWindowDataIterator() {
  delete[] buf_;
  delete it_;
}

void SlidingWindowDataIterator::SetMaxDataSetSize(int max_dataset_size) {
  max_dataset_size *= it_->GetNumWindows();
  if (max_dataset_size > 0 && dataset_size_ > max_dataset_size) {
    dataset_size_ = max_dataset_size;
  }
}

void SlidingWindowDataIterator::Seek(int row) {
  file_id_ = row;
}

int SlidingWindowDataIterator::Tell() const {
  return file_id_;
}

void SlidingWindowDataIterator::Get(float* data_out, const int row) const {
  cerr << "Not implemented" << endl;
  exit(1);
}

void SlidingWindowDataIterator::GetNext(float* data_out) {
  if (it_->Done()) {
    it_->SetImage(file_names_[file_id_++]);
    if (file_id_ == file_names_.size()) file_id_ = 0;
  }
  it_->GetNext(buf_);
  for (int i = 0; i < image_size_ * image_size_ * num_colors_; i++) {
    data_out[i] = static_cast<float>(buf_[i]);
  }
}

CropDataIterator::CropDataIterator(const config::DataStreamConfig& config):
  DataIterator(config), image_size_(config.image_size()), file_id_(0) {
  num_dims_ = image_size_ * image_size_ * num_colors_;
  it_ = new CropIterator<unsigned char>(image_size_, config.context_factor(), config.warp_bbox());
  buf_ = new unsigned char[num_dims_];
  ReadLines(file_pattern_, file_names_);


  ifstream f(config.bbox_file(), ios::in);
  string line;
  dataset_size_ = 0;
  // The format for each line is -
  // <width> <height> <xmin1> <ymin1> <xmax1> <ymax1> <xmin2> <ymin2> ...
  while (getline(f, line)) {
    istringstream iss(line);
    vector<string> tokens;
    copy(istream_iterator<string>(iss), istream_iterator<string>(),
         back_inserter<vector<string> >(tokens));
    int num_tokens = tokens.size();
    if (num_tokens == 0 || num_tokens % 5 != 0) {
      cerr << "Error parsing line " << line << endl;
      exit(1);
    }
    int num_boxes = num_tokens / 5;
    vector<box> b_list (num_boxes);
    for (int i = 0; i < num_boxes; i++) {
      b_list[i].xmin = atoi(tokens[5*i  ].c_str());
      b_list[i].ymin = atoi(tokens[5*i+1].c_str());
      b_list[i].xmax = atoi(tokens[5*i+2].c_str());
      b_list[i].ymax = atoi(tokens[5*i+3].c_str());
      // 5th number is the box weight, not relevant here.
    }
    crops_.push_back(b_list);
    dataset_size_ += b_list.size();
  }
  f.close();

  if (normalize_) {
    LoadMeans(config.mean_file());
  }
}

CropDataIterator::~CropDataIterator() {
  delete[] buf_;
  delete it_;
}

void CropDataIterator::Seek(int row) {
  file_id_ = row;
}

int CropDataIterator::Tell() const {
  return file_id_;
}

void CropDataIterator::Get(float* data_out, const int row) const {
  cerr << "Not implemented" << endl;
  exit(1);
}

void CropDataIterator::GetNext(float* data_out) {
  if (it_->Done()) {
    it_->SetImage(file_names_[file_id_], crops_[file_id_]);
    file_id_++;
    if (file_id_ == crops_.size()) file_id_ = 0;
  }
  it_->GetNext(buf_);
  for (int i = 0; i < image_size_ * image_size_ * num_colors_; i++) {
    data_out[i] = static_cast<float>(buf_[i]);
  }
}

TextDataIterator::TextDataIterator(const config::DataStreamConfig& config):
  DataIterator(config) {
  ifstream f(file_pattern_, ios::in);
  string line;
  getline(f, line);
  istringstream iss(line);
  vector<string> tokens;
  copy(istream_iterator<string>(iss), istream_iterator<string>(),
       back_inserter<vector<string> >(tokens));
  num_dims_ = tokens.size();
  dataset_size_ = 1;
  while (getline(f, line)) dataset_size_++;
  data_ = new float[num_dims_ * dataset_size_];
  f.close();
  ifstream g(file_pattern_, ios::in);
  for (int i = 0; i < dataset_size_; i++) {
    g >> data_[i];
  }
  g.close();
  if (normalize_) {
    LoadMeans(config.mean_file());
  }
}

TextDataIterator::~TextDataIterator() {
  delete[] data_;
}

void TextDataIterator::Get(float* data_out, const int row) const {
  memcpy(data_out, data_ + num_dims_ * row, sizeof(float) * num_dims_);
}

void TextDataIterator::GetNext(float* data_out) {
  Get(data_out, row_);
  row_++;
  if (row_ == dataset_size_) row_ = 0;
}

BoundingBoxIterator::BoundingBoxIterator(const config::DataStreamConfig& config):
  DataIterator(config), jitter_source_(NULL) {
  ifstream f(file_pattern_, ios::in);
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
    int width = atoi(tokens[0].c_str());
    int height = atoi(tokens[1].c_str());
    int num_boxes = (num_tokens - 2) / 4;
    vector<box> b_list (num_boxes);
    for (int i = 0; i < num_boxes; i++) {
      b_list[i].xmin = atoi(tokens[2+4*i  ].c_str());
      b_list[i].ymin = atoi(tokens[2+4*i+1].c_str());
      b_list[i].xmax = atoi(tokens[2+4*i+2].c_str());
      b_list[i].ymax = atoi(tokens[2+4*i+3].c_str());
    }
    data_.push_back(b_list);
    img_width_.push_back(width);
    img_height_.push_back(height);
  }
  f.close();
  dataset_size_ = data_.size();
}

float BoundingBoxIterator::Intersection(const box& b1, const box& b2) {
  float x_overlap = MAX(0, MAX(b1.xmin, b2.xmin) - MIN(b1.xmax, b2.xmax));
  float y_overlap = MAX(0, MAX(b1.ymin, b2.ymin) - MIN(b1.ymax, b2.ymax));
  return x_overlap * y_overlap;
}

float BoundingBoxIterator::Area(const box& b) {
  return (b.xmax - b.xmin) * (b.ymax - b.ymin);
}

float BoundingBoxIterator::VisibleFraction(const box& b, const box& fov) {
  return Intersection(b, fov) / Area(b);
}

void BoundingBoxIterator::Get(float* data_out, const int row) const {
  int num_fovs = num_dims_ / 4;
  vector<box> b_list(data_[row]);
  int width = img_width_[row];
  int height = img_height_[row];
  for (box& b: b_list) {
    if (jitter_source_ != NULL) {
      jitter_source_->RectifyBBox(b, width, height, row);
    }
    b.xmin /= patch_size_;
    b.xmax /= patch_size_;
    b.ymin /= patch_size_;
    b.ymax /= patch_size_;
  }
  for (int i = 0; i < num_fovs; i++) {
    const box &fov = fov_box_[i];
    
    // Find the bounding box that is most visible from this field of view.
    int best_box = 0;
    if (b_list.size() > 1) {
      float f_best = VisibleFraction(b_list[0], fov), f;
      for (int j = 1; j < b_list.size(); j++) {
        f = VisibleFraction(b_list[j], fov);
        if (f > f_best) {
          best_box = j;
          f_best = f;
        }
      }
    }
    const box& b = b_list[best_box];
    data_out[i               ] = b.xmin;
    data_out[i +     num_fovs] = b.ymin;
    data_out[i + 2 * num_fovs] = b.xmax;
    data_out[i + 3 * num_fovs] = b.ymax;
  }
}

void BoundingBoxIterator::GetNext(float* data_out) {
  Get(data_out, row_);
  row_++;
  if (row_ == dataset_size_) row_ = 0;
}

void BoundingBoxIterator::SetFOV(
    const int size, const int stride, const int pad1, const int pad2,
    const int patch_size, const int num_fov_x, const int num_fov_y) {
  patch_size_ = patch_size;

  float fov_size   = (float)size / patch_size;
  float fov_stride = (float)stride / patch_size;
  float fov_pad1   = (float)pad1 / patch_size;
  
  int num_fovs = num_fov_y * num_fov_x;
  num_dims_ = 4 * num_fovs;
  fovs_.AllocateGPUMemory(1, num_dims_);

  float* fov = fovs_.GetHostData();
  fov_box_.resize(num_fovs);

  for (int i = 0; i < num_fovs; i++) {
    box &b = fov_box_[i];
    
    int r = i / num_fov_x, c = i % num_fov_x;
    b.xmin = -fov_pad1 + fov_stride * c;
    b.ymin = -fov_pad1 + fov_stride * r;
    b.xmax = b.xmin + fov_size;
    b.ymax = b.ymin + fov_size;
    
    fov[i]                = b.xmin;
    fov[i + num_fovs]     = b.ymin;
    fov[i + 2 * num_fovs] = b.xmin;
    fov[i + 3 * num_fovs] = b.ymin;
  }
  fovs_.CopyToDevice();
}

void BoundingBoxIterator::AddNoise(Matrix& input, Matrix& output) {
  if (noise_source_ != NULL) {
    input.CopyTranspose(output);
    output.RectifyBBox(noise_source_->GetWidthOffset(),
                       noise_source_->GetHeightOffset(),
                       noise_source_->GetFlipBit(),
                       patch_size_, patch_size_);
    output.AddRowVec(fovs_, -1);
  }
}
void BoundingBoxIterator::SetNoiseSource(DataIterator* it) {
  DataIterator::SetNoiseSource(it);
  jitter_source_ = dynamic_cast<ImageDataIterator*>(it);
  // It's fine if this turns out to be NULL. 
}

}
}
