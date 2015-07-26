#include "precomp.hpp"
using std::string;
using namespace std;

namespace cv
{
namespace cnn_3dobj
{
	DataTrans::DataTrans()
	{
	};
	void DataTrans::list_dir(const char *path,vector<string>& files,bool r)
	{
	  DIR *pDir;
	  struct dirent *ent;
	  char childpath[512];
	  pDir = opendir(path);
	  memset(childpath, 0, sizeof(childpath));
	  while ((ent = readdir(pDir)) != NULL)
	  {
	    if (ent->d_type & DT_DIR)
	    {

	      if (strcmp(ent->d_name, ".") == 0 || strcmp(ent->d_name, "..") == 0)
	      {
		continue;
	      }
	      if(r)
	      {
		sprintf(childpath, "%s/%s", path, ent->d_name);
		DataTrans::list_dir(childpath,files,false);
	      }
	    }
	    else
	    {
	      files.push_back(ent->d_name);
	    }
	  }
	  sort(files.begin(),files.end());

	};

	string DataTrans::get_classname(string path)
	{
	  int index = path.find_last_of('_');
	  return path.substr(0, index);
	}


	int DataTrans::get_labelid(string fileName)
	{
	  string class_name_tmp = get_classname(fileName);
	  all_class_name.insert(class_name_tmp);
	  map<string,int>::iterator name_iter_tmp = class2id.find(class_name_tmp);
	  if (name_iter_tmp == class2id.end())
	  {
	    int id = class2id.size();
	    class2id.insert(name_iter_tmp, std::make_pair(class_name_tmp, id));
	    return id;
	  }
	  else
	  {
	    return name_iter_tmp->second;
	  }
	}

	void DataTrans::loadimg(string path,char* buffer,const bool is_color)
	{
	  cv::Mat img = cv::imread(path, is_color);
	  string val;
	  int rows = img.rows;
	  int cols = img.cols;
	  int pos=0;
	  int channel;
	  if (is_color == 0)
	  {
		  channel = 1;
	  }else{
		  channel = 3;
	  }
	  for (int c = 0; c < channel; c++)
	  {
	    for (int row = 0; row < rows; row++)
	    {
	      for (int col = 0; col < cols; col++)
	      {
		buffer[pos++]=img.at<cv::Vec3b>(row,col)[c];
	      }
	    }
	  }

	};

	void DataTrans::convert(string imgdir,string outputdb,string attachdir,int channel,int width,int height)
	{
	  leveldb::DB* db;
	  leveldb::Options options;
	  options.create_if_missing = true;
	  // options.error_if_exists = true;
	  caffe::Datum datum;
	  datum.set_channels(channel);
	  datum.set_height(height);
	  datum.set_width(width);
	  int image_size = channel*width*height;
	  char buffer[image_size];

	  string value;
	  CHECK(leveldb::DB::Open(options, outputdb, &db).ok());
	  vector<string> filenames;
	  list_dir(imgdir.c_str(),filenames, false);
	  string img_log = attachdir+"image_filename";
	  ofstream writefile(img_log.c_str());
	  for(int i=0;i<(int)filenames.size();i++)
	  {
	    string path= imgdir;
	    path.append(filenames[i]);

	    loadimg(path,buffer,false);

	    int labelid = get_labelid(filenames[i]);

	    datum.set_label(labelid);
	    datum.set_data(buffer,image_size);
	    datum.SerializeToString(&value);
	    snprintf(buffer, image_size, "%05d", i);
	    printf("\nclassid:%d classname:%s abspath:%s",labelid,get_classname(filenames[i]).c_str(),path.c_str());
	    db->Put(leveldb::WriteOptions(),string(buffer),value);
	    //printf("%d %s\n",i,fileNames[i].c_str());

	    assert(writefile.is_open());
	    writefile<<i<<" "<<filenames[i]<<"\n";

	  }
	  delete db;
	  writefile.close();

	  img_log = attachdir+"image_classname";
	  writefile.open(img_log.c_str());
	  set<string>::iterator iter = all_class_name.begin();
	  while(iter != all_class_name.end())
	  {
	    assert(writefile.is_open());
	    writefile<<(*iter)<<"\n";
	    //printf("%s\n",(*iter).c_str());
	    iter++;
	  }
	  writefile.close();

	};

	std::vector<cv::Mat> DataTrans::feature_extraction_pipeline(std::string pretrained_binary_proto, std::string feature_extraction_proto, std::string save_feature_dataset_names, std::string extract_feature_blob_names, int num_mini_batches, std::string device, int dev_id) {
	  if (strcmp(device.c_str(), "GPU") == 0) {
	    LOG(ERROR)<< "Using GPU";
	    int device_id = 0;
	    if (strcmp(device.c_str(), "GPU") == 0) {
	      device_id = dev_id;
	      CHECK_GE(device_id, 0);
	    }
	    LOG(ERROR) << "Using Device_id=" << device_id;
	    Caffe::SetDevice(device_id);
	    Caffe::set_mode(Caffe::GPU);
	  } else {
	    LOG(ERROR) << "Using CPU";
	    Caffe::set_mode(Caffe::CPU);
	  }
	  boost::shared_ptr<Net<float> > feature_extraction_net(
	      new Net<float>(feature_extraction_proto, caffe::TEST));
	  feature_extraction_net->CopyTrainedLayersFrom(pretrained_binary_proto);
	  std::vector<std::string> blob_names;
	  blob_names.push_back(extract_feature_blob_names);
	  std::vector<std::string> dataset_names;
	  dataset_names.push_back(save_feature_dataset_names);
	  CHECK_EQ(blob_names.size(), dataset_names.size()) <<
	      " the number of blob names and dataset names must be equal";
	  size_t num_features = blob_names.size();

	  for (size_t i = 0; i < num_features; i++) {
	    CHECK(feature_extraction_net->has_blob(blob_names[i]))
		<< "Unknown feature blob name " << blob_names[i]
		<< " in the network " << feature_extraction_proto;
	  }
	  std::vector<FILE*> files;
	  for (size_t i = 0; i < num_features; ++i)
	  {
		  LOG(INFO) << "Opening file " << dataset_names[i];
		  FILE * temp = fopen(dataset_names[i].c_str(), "wb");
		  files.push_back(temp);
	  }


	  LOG(ERROR)<< "Extacting Features";

	  Datum datum;
	  std::vector<cv::Mat> featureVec;
	  std::vector<Blob<float>*> input_vec;
	  std::vector<int> image_indices(num_features, 0);
	  for (int batch_index = 0; batch_index < num_mini_batches; ++batch_index) {
	    feature_extraction_net->Forward(input_vec);
	    for (size_t i = 0; i < num_features; ++i) {
			const boost::shared_ptr<Blob<float> > feature_blob = feature_extraction_net
		  ->blob_by_name(blob_names[i]);
	      int batch_size = feature_blob->num();
	      int dim_features = feature_blob->count() / batch_size;
		  if (batch_index == 0)
		  {
			  int fea_num = batch_size*num_mini_batches;
			  fwrite(&dim_features, sizeof(int), 1, files[i]);
			  fwrite(&fea_num, sizeof(int), 1, files[i]);
		  }
	      const float* feature_blob_data;
	      for (int n = 0; n < batch_size; ++n) {

		feature_blob_data = feature_blob->cpu_data() +
		    feature_blob->offset(n);
			fwrite(feature_blob_data, sizeof(float), dim_features, files[i]);
			cv::Mat tempfeat = cv::Mat(1, dim_features, CV_32FC1);
			for (int dim = 0; dim < dim_features; dim++) {
				tempfeat.at<float>(0,dim) = *(feature_blob_data++);
			}
			featureVec.push_back(tempfeat);
		++image_indices[i];
		if (image_indices[i] % 1000 == 0) {
		  LOG(ERROR)<< "Extracted features of " << image_indices[i] <<
		      " query images for feature blob " << blob_names[i];
		}
	      }  // for (int n = 0; n < batch_size; ++n)
	    }  // for (int i = 0; i < num_features; ++i)
	  }  // for (int batch_index = 0; batch_index < num_mini_batches; ++batch_index)
	  // write the last batch
	  for (size_t i = 0; i < num_features; ++i) {
		  fclose(files[i]);
	  }

	  LOG(ERROR)<< "Successfully extracted the features!";
	  return featureVec;
	};
}}
