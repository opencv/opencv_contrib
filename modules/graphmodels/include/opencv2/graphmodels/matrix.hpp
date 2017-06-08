#ifndef OPENCV_MATRIX_HPP
#define OPENCV_MATRIX_HPP

#include <string>
#include <vector>

#include "opencv2/graphmodels/cudamat/cudamat_conv.cuh"
#include "opencv2/graphmodels/cudamat/cudamat.cuh"

#include "opencv2/core/cuda.hpp"

#include "hdf5.h"

namespace cv
{
namespace graphmodels
{

/** A GPU matrix class.*/
class CV_EXPORTS Matrix {
 public:
  Matrix();
  Matrix(const int rows, const int cols, const bool on_gpu);
  ~Matrix();
  
  void Tie(Matrix &m);
  void AllocateGPUMemory(const int rows, const int cols, const std::string& name);
  void AllocateGPUMemory(const int rows, const int cols);
  void AllocateMainMemory(const int rows, const int cols);
  void Set(const float val);
  void Set(Matrix& val);
  void CopyP2PAsync(Matrix& val);
  void GetSlice(Matrix& slice, int start, int end);
  void FillWithRand();
  void FillWithRandn();
  void CopyToHost();
  void CopyToDevice();
  void CopyToDeviceSlice(const int start, const int end);
  void CopyToHostSlice(const int start, const int end);
  void CopyFromMainMemory(Matrix& mat);
  void Reshape(const int rows, const int cols);
  float Norm();
  void Print();
  void PrintToFile(const std::string& filename);
  void WriteToFile(FILE* file);
  void ReadFromFile(FILE* file);
  void WriteHDF5(hid_t file, const std::string& name);
  void ReadHDF5(hid_t file, const std::string& name);
  void AllocateAndReadHDF5(hid_t file, const std::string& name);
  std::string GetShapeString();
  cudamat* GetMat() { return &mat_; }
  cudamat* GetMatTranspose() { return &mat_t_; }
  float* GetHostData() { return mat_.data_host; }
  int GetRows() const {return mat_.size[0];}
  int GetCols() const {return mat_.size[1];}
  int GetNumEls() const {return mat_.size[1] * mat_.size[0]; }

  int GetGPUId() const { return gpu_id_; }
  void SetReady();
  void WaitTillReady();

  // GPU computing methods.
  void Add(float val);
  void Add(Matrix& m);
  void Add(Matrix& m, float alpha);
  void SquashRelu();
  void AddRowVec(Matrix& v);
  void AddRowVec(Matrix& v, float alpha);
  void AddColVec(Matrix& v, float alpha);
  void MultByRowVec(Matrix& v);
  void DivideByColVec(Matrix& v);
  float Sum();
  void SumRows(Matrix& target, float alpha, float beta);
  void SumCols(Matrix& target, float alpha, float beta);
  void Mult(float val);
  void Mult(Matrix& val);
  void Divide(float val);
  void Divide(Matrix& val);
  void Subtract(Matrix& m, Matrix& target);
  void LowerBound(float val);
  void Sqrt();
  void UpperBoundMod(float val);
  void SqSumAxis(Matrix& target, int axis, float beta, float alpha);
  void NormLimitByAxis(int axis, float val, bool constraint);
  void Dropout(float dropprob, float fill_value, float scale_factor);
  void ApplyDerivativeOfReLU(Matrix& state);
  void ApplySoftmax();
  void ApplyLogistic();
  void ApplyDerivativeOfLogistic(Matrix& state);
  float EuclidNorm();
  float VDot(Matrix& m);
  void CopyTransposeBig(Matrix& m);
  void CopyTranspose(Matrix& m);
  void ShuffleColumns(Matrix& rand_perm_indices);
  void AddToEachPixel(Matrix& v, float mult);
  void RectifyBBox(Matrix& width_offset, Matrix& height_offset, Matrix& flip,
                   int patch_width, int patch_height);
  static void LogisticCEDeriv(Matrix& state, Matrix& gt, Matrix& deriv);
  static void LogisticCorrect(Matrix& state, Matrix& gt, Matrix& output);
  static void SoftmaxCEDeriv(Matrix& state, Matrix& gt, Matrix& deriv);
  static void SoftmaxCorrect(Matrix& state, Matrix& gt, Matrix& output);
  static void SoftmaxCE(Matrix& state, Matrix& gt, Matrix& output);
  static void SoftmaxDistCE(Matrix& state, Matrix& gt, Matrix& output);
  static void HingeLossDeriv(Matrix& state, Matrix& gt, Matrix& deriv,
                             bool quadratic, float margin);

  static void Dot(Matrix& a, Matrix& b, Matrix& c, float alpha, float beta);
  static void Dot(Matrix& a, Matrix& b, Matrix& c, float alpha, float beta,
                  bool transpose_a, bool transpose_b);

  static void ConvUp(Matrix& input, Matrix& w, Matrix& output, int image_size,
                     int num_modules_y, int num_modules_x, int padding,
                     int stride, int num_input_channels, float scale_targets);

  static void ConvDown(Matrix& deriv_output, Matrix& w, Matrix& deriv_input,
                       int image_size_y, int image_size_x, int num_modules_y,
                       int padding, int stride, int num_input_channels,
                       float scale_targets);

  static void ConvOutp(Matrix& input, Matrix& deriv_output, Matrix& dw,
                       int image_size_y, int num_modules_y, int num_modules_x,
                       int kernel_size, int padding, int stride,
                       int num_input_channels, int partial_sum,
                       float scale_targets, float scale_outputs);

  static void LocalUp(Matrix& input, Matrix& w, Matrix& output, int image_size,
                      int num_modules_y, int num_modules_x, int padding,
                      int stride, int num_input_channels, float scale_targets);

  static void LocalDown(Matrix& deriv_output, Matrix& w, Matrix& deriv_input,
                        int image_size_y, int image_size_x, int num_modules_y,
                        int padding, int stride, int num_input_channels,
                        float scale_targets);

  static void LocalOutp(Matrix& input, Matrix& deriv_output, Matrix& dw,
                        int image_size_y, int num_modules_y, int num_modules_x,
                        int kernel_size, int padding, int stride,
                        int num_input_channels,
                        float scale_targets, float scale_outputs);

  static void ConvMaxPool(Matrix& input, Matrix& output, int num_input_channels,
                     int kernel_size, int padding, int stride, int num_modules);

  static void ConvMaxPoolUndo(Matrix& input, Matrix& deriv_output, Matrix& output,
                          Matrix& deriv_input, int kernel_size, int padding,
                          int stride, int num_modules);

  static void ConvResponseNormCrossMap(
      Matrix& input, Matrix& output, int numFilters, int sizeF, float addScale,
      float powScale, bool blocked);

  static void ConvResponseNormCrossMapUndo(
    Matrix& outGrads, Matrix& inputs, Matrix& acts, Matrix& targets, int numFilters,
    int sizeF, float addScale, float powScale, bool blocked);

  static void ConvUpSample(Matrix& input, Matrix& output, int factor,
                           int input_image_size, float scaleTargets);
  
  static void ConvDownSample(Matrix& input, Matrix& output, int factor,
                             int input_image_size);

  static void ConvRGBToYUV(Matrix& input, Matrix& output);

  static void ExtractPatches(Matrix& source, Matrix& dest, Matrix& width_offset,
                             Matrix& height_offset, Matrix& flip_bit,
                             int image_width, int image_height, int patch_width,
                             int patch_height);

  static void GetOnes(int rows, int cols, Matrix& ones);
  static void RegisterTempMemory(int size);
  static void RegisterTempMemory(int size, const std::string& why);
  static void RegisterOnes(int size);
  static void GetTemp(int rows, int cols, Matrix& temp);
  static void InitRandom(int seed);
  static void SetupCUDADevice(int gpu_id);
  static void SetupCUDADevices(const std::vector<int>& boards);
  static void SetDevice(int gpu_id);
  static void SyncAllDevices();
  static int GetDevice();
  static int GetNumBoards() {return num_boards_;}

  static std::vector<Matrix> ones_, temp_;
  static std::vector<rnd_struct> rnd_;

 private:
  cudamat mat_, mat_t_;
  cudaEvent_t ready_;
  int gpu_id_;
  std::string name_;
  static int num_boards_;
  static int current_gpu_id_;
  static std::vector<int> boards_, temp_size_, ones_size_;
};

}
}

#endif
