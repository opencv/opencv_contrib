#include "opencv2/graphmodels/util.hpp"
#include <stdlib.h>
#include <string.h>
#include <iostream>
#include <fstream>
#include <ctime>
#include <sstream>
#include <limits>

using namespace std;

namespace cv
{
namespace graphmodels
{

void WaitForEnter() {
  cout << "Press ENTER to continue...";
  cin.ignore(numeric_limits<streamsize>::max(), '\n');
}

int Bound(int val, int lb, int ub) {
  val = val > ub ? ub : val;
  val = val < lb ? lb : val;
  return val;
}

void TimestampModelFile(const string& src_file, const string& dest_file, const string& timestamp) {
  ifstream src(src_file, ios::binary);
  ofstream dst(dest_file, ios::binary);
  if (!dst) {
    cerr << "Error: Could not write to " << dest_file << endl;
    exit(1);
  } else {
    cout << "Timestamped model : " << dest_file << endl;
  }
  dst << src.rdbuf();
  dst << endl << "timestamp : \"" << timestamp << "\"" << endl;
  dst.close();
  src.close();
}

// Year-Month-Day-Hour-Minute-Second
string GetTimeStamp() {
  time_t rawtime;
  struct tm *timeinfo;
  time(&rawtime);
  timeinfo = localtime(&rawtime);
  char timestr[30];
  strftime(timestr, sizeof(timestr), "%Y%m%d%H%M%S", timeinfo);
  stringstream ss;
  ss << timestr;
  return ss.str();
}

void ReadModel(const string& model_file, config::Model& model) {
  string ext = model_file.substr(model_file.find_last_of('.'));
  if (ext.compare(".pb") == 0) {
    ReadModelBinary(model_file, model);
  } else {
    ReadModelText(model_file, model);
  }
}

void ReadModelText(const string& model_file, config::Model& model) {
  stringstream ss;
  ifstream file(model_file.c_str());
  ss << file.rdbuf();
  if (!google::protobuf::TextFormat::ParseFromString(ss.str(), &model)) {
    cerr << "Could not read text proto buffer : " << model_file << endl;
    exit(1);
  }
}
void ReadFeatureExtractorConfig(const string& config_file, config::FeatureExtractorConfig& config) {
  stringstream ss;
  ifstream file(config_file.c_str());
  ss << file.rdbuf();
  if (!google::protobuf::TextFormat::ParseFromString(ss.str(), &config)) {
    cerr << "Could not read text proto buffer : " << config_file << endl;
    exit(1);
  }
}
void ReadDataConfig(const string& data_config_file, config::DatasetConfig& data_config) {
  stringstream ss;
  ifstream file(data_config_file.c_str());
  ss << file.rdbuf();
  if (!google::protobuf::TextFormat::ParseFromString(ss.str(), &data_config)) {
    cerr << "Could not read text proto buffer : " << data_config_file << endl;
    exit(1);
  }
}

void ReadLayerConfig(const string& layer_config_file, config::Layer& layer_config) {
  stringstream ss;
  ifstream file(layer_config_file.c_str());
  ss << file.rdbuf();
  if (!google::protobuf::TextFormat::ParseFromString(ss.str(), &layer_config)) {
    cerr << "Could not read text proto buffer : " << layer_config_file << endl;
    exit(1);
  }
}

void WriteModelBinary(const string& output_file, const config::Model& model) {
  ofstream out(output_file.c_str());
  model.SerializeToOstream(&out);
  out.close();
}

void ReadModelBinary(const string& input_file, config::Model& model) {
  ifstream in(input_file.c_str());
  model.ParseFromIstream(&in);
  in.close();
}


void WriteHDF5CPU(hid_t file, float* mat, int rows, int cols, const string& name) {
  hid_t dataset, dataspace;
  hsize_t dimsf[2];
  dimsf[0] = rows;
  dimsf[1] = cols;
  dataspace = H5Screate_simple(2, dimsf, NULL);
  dataset = H5Dcreate(file, name.c_str(), H5T_NATIVE_FLOAT, dataspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  H5Dwrite(dataset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, mat);
  H5Sclose(dataspace);
  H5Dclose(dataset);
}

void ReadHDF5ShapeFromFile(const string& file_name, const string& dataset_name, int* rows, int* cols) {
  hid_t file = H5Fopen(file_name.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
  ReadHDF5Shape(file, dataset_name, rows, cols);
  H5Fclose(file);
}

void ReadHDF5Shape(hid_t file, const string& name, int* rows, int* cols) {
  hid_t dataset, dataspace;
  dataset = H5Dopen(file, name.c_str(), H5P_DEFAULT);
  dataspace = H5Dget_space(dataset);
  const int ndims = H5Sget_simple_extent_ndims(dataspace);
  hsize_t dims_out[2];
  H5Sget_simple_extent_dims(dataspace, dims_out, NULL);
  *cols = dims_out[0];
  *rows = (ndims == 1) ? 1 :dims_out[1];
  H5Dclose(dataset);
}

void WriteHDF5IntAttr(hid_t file, const string& name, const int* val) {
  hid_t aid, attr;
  aid  = H5Screate(H5S_SCALAR);
  attr = H5Acreate2(file, name.c_str(), H5T_NATIVE_INT, aid, H5P_DEFAULT,
                    H5P_DEFAULT);
  H5Awrite(attr, H5T_NATIVE_INT, val);
  H5Sclose(aid);
  H5Aclose(attr);
}

void ReadHDF5IntAttr(hid_t file, const string& name, int* val) {
  hid_t attr = H5Aopen(file, name.c_str(), H5P_DEFAULT);
  H5Aread(attr, H5T_NATIVE_INT, val);
  H5Aclose(attr);
}

void ReadHDF5CPU(hid_t file, float* mat, int size, const string& name) {
  hid_t dataset, dataspace;
  dataset = H5Dopen(file, name.c_str(), H5P_DEFAULT);
  dataspace = H5Dget_space(dataset);
  const int ndims = H5Sget_simple_extent_ndims(dataspace);
  hsize_t dims_out[2];
  H5Sget_simple_extent_dims(dataspace, dims_out, NULL);
  int rows = (ndims == 1) ? 1 : dims_out[1];
  int datasize = dims_out[0] * rows;
  if (size != datasize) {
    cerr << "Dimension mismatch: Expected "
         << size << " Got " << rows << "-" << dims_out[0] << endl;
    exit(1);
  }
  H5Dread(dataset, H5T_NATIVE_FLOAT, dataspace, dataspace, H5P_DEFAULT, mat);
  H5Dclose(dataset);
}

bool ReadLines(const string& filename, vector<string>& lines) {
  ifstream f(filename, ios::in);
  if (!f.is_open()) {
    cerr << "Could not open data file : " << filename << endl;
    return false;
  }

  while (!f.eof()) {
    string str;
    f >> str;
    if (!f.eof()) lines.push_back(str);
  }
  f.close();
  return true;
}

string GetStringError(int err_code) {
  if (err_code == -1)
    return "Incompatible matrix dimensions.";
  if (err_code == -2)
    return "CUBLAS error.";
  if (err_code == -3)
    return "CUDA error ";
  if (err_code == -4)
    return "Operation not supported on views.";
  if (err_code == -5)
    return "Operation not supported on transposed matrices.";
  if (err_code == -6)
    return "";
  if (err_code == -7)
    return "Incompatible transposedness.";
  if (err_code == -8)
    return "Matrix is not in device memory.";
  if (err_code == -9)
    return "Operation not supported.";
  return "Some error";
}

}
}
