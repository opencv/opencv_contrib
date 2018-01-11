#include <opencv2/dnn_modern.hpp>
#include <opencv2/imgcodecs.hpp>

#include <iostream>
#include <fstream>


using namespace std;
using namespace cv;
using namespace cv::dnn2;

static void help() {
  cout
      << "\n----------------------------------------------------------------------------\n"
      << " This program shows how to import a Caffe model using the \n"
      << " OpenCV Modern Deep Learning module (DNN2).\n"
      << " Usage:\n"
      << "        example_dnn_modern_simple_test <model_file> <trained_file> <mean_file>\n"
      << "                                       <label_file> <image_file>\n"
      << " where: model_file   is the path to the *.prototxt\n"
      << "        trained_file is the path to the *.caffemodel\n"
      << "        mean_file    is the path to the *.binaryproto\n"
      << "        label_file   is the path to the labels file\n"
      << "        image_file   is the path to the image to evaluate\n"
      << "----------------------------------------------------------------------------\n\n"
      << endl;
}

vector<string> get_label_list(const string& label_file);
void print_n_labels(const vector<string>& labels,
                    const vector<float_t>& result,
                    const int top_n);

vector<string> get_label_list(const string& label_file) {
    string line;
    ifstream ifs(label_file.c_str());

    if (ifs.fail() || ifs.bad()) {
        throw runtime_error("failed to open:" + label_file);
    }

    vector<string> lines;
    while (getline(ifs, line)) lines.push_back(line);

    return lines;
}

void print_n_labels(const vector<string>& labels,
                    const vector<float_t>& result,
                    const int top_n) {
    vector<float_t> sorted(result.begin(), result.end());

    partial_sort(sorted.begin(), sorted.begin()+top_n, sorted.end(), greater<float_t>());

    for (int i = 0; i < top_n; i++) {
        size_t idx = distance(result.begin(), find(result.begin(), result.end(), sorted[i]));
        cout << labels[idx] << "," << sorted[i] << endl;
    }
}

int main(int argc, char* argv[]) {

    if (argc < 6) {
        help();
        exit(0);
    }

    int    arg_channel  = 1;
    string model_file   = argv[arg_channel++];
    string trained_file = argv[arg_channel++];
    string mean_file    = argv[arg_channel++];
    string label_file   = argv[arg_channel++];
    string img_file     = argv[arg_channel++];

    // load Caffe model
    Ptr<CaffeConverter> caffe_ptr = CaffeConverter::create(
        model_file, trained_file, mean_file);

    // load input image
    cv::Mat img = cv::imread(img_file, -1);

    // inference !
    vector<float_t> scores;
    caffe_ptr->eval(img, scores);

    // retrieve n labels
    const int n = 5;
    vector<string> labels = get_label_list(label_file);

    print_n_labels(labels, scores, n);

    return 0;
}
