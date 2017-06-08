#ifndef DATAWRITER_H_
#define DATAWRITER_H_
#include "layer.hpp"

#include "opencv2/core.hpp"

namespace cv
{
namespace graphmodels
{

/** Writes data into an HDF5 file.
 * Handles multiple output streams.
 */ 
class CV_EXPORTS DataWriter {
 public:
  DataWriter(const config::FeatureExtractorConfig config);
  virtual ~DataWriter();
  void SetNumDims(const string& name, const int num_dims);
  void SetDataSetSize(int dataset_size);
  virtual void Write(vector<Layer*>& layers, int numcases);
  void WriteHDF5(Matrix& m, const string& dataset, int numcases, bool transpose);
  void WriteHDF5SeqBuf(Matrix& m, const string& dataset, int numcases);

 private:
  int dataset_size_;
  typedef struct {
    int num_dims, current_row, average_batches, average_online, counter, consumed;
    hid_t dataset, dataspace;
    Matrix buf, seq_buf;
  } stream;
  map<string, stream> streams_;
  hid_t file_;
};

/** Buffers a specified number of batches, averages them and then writes
 * the average into an HDF5 file.
 */ 
/*
class AveragedDataWriter : public DataWriter {
 public:
  AveragedDataWriter(const string& output_file, const int dataset_size,
                     const int avg_after, int max_batchsize);
  ~AveragedDataWriter();
  virtual void AddStream(const string& name, const int numdims);
  virtual void Write(Matrix& mat, const int data_id, const int rows);
 private:
  const int avg_after_, max_batchsize_;
  vector<Matrix*> buf_;
  vector<int> counter_;
};
*/
/** Averages a specified number of consecutive entries and writes the
 * average into an HDF5 file.
 */
/*
class SequentialAveragedDataWriter : public DataWriter {
 public:
  SequentialAveragedDataWriter(const string& output_file, const int dataset_size,
                               const int avg_after);
  ~SequentialAveragedDataWriter();
  virtual void AddStream(const string& name, const int numdims);
  virtual void Write(Matrix& mat, const int data_id, const int rows);

 private:
  const int avg_after_, dataset_size_;
  vector<Matrix*> buf_;
  int consumed_, num_rows_written_;
};
*/

}
}

#endif
