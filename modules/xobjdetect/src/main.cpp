#include <opencv2/opencv.hpp>
#include <vector>
#include <sstream>
#include <cassert>
#include <iostream>
#include <fstream>
#include <iomanip>

#include "lbpfeatures.h"

using namespace std;
using namespace cv;

static void compute_cdf(const Mat1b& data,
                        const Mat1f& weights,
                        Mat1f& cdf)
{
    for (int i = 0; i < cdf.cols; ++i)
        cdf(0, i) = 0;

    for (int i = 0; i < weights.cols; ++i) {
        cdf(0, data(0, i)) += weights(0, i);
    }

    for (int i = 1; i < cdf.cols; ++i) {
        cdf(0, i) += cdf(0, i - 1);
    }
}

void compute_min_step(const Mat &data_pos, const Mat &data_neg, size_t n_bins,
                      Mat &data_min, Mat &data_step)
{
    // Check that quantized data will fit in unsigned char
    assert(n_bins <= 256);

    assert(data_pos.rows == data_neg.rows);

    Mat reduced_pos, reduced_neg;

    reduce(data_pos, reduced_pos, 1, CV_REDUCE_MIN);
    reduce(data_neg, reduced_neg, 1, CV_REDUCE_MIN);
    min(reduced_pos, reduced_neg, data_min);
    data_min -= 0.01;

    Mat data_max;
    reduce(data_pos, reduced_pos, 1, CV_REDUCE_MAX);
    reduce(data_neg, reduced_neg, 1, CV_REDUCE_MAX);
    max(reduced_pos, reduced_neg, data_max);
    data_max += 0.01;

    data_step = (data_max - data_min) / (n_bins - 1);
}

void quantize_data(Mat &data, Mat1f &data_min, Mat1f &data_step)
{
#pragma omp parallel for
    for (int col = 0; col < data.cols; ++col) {
        data.col(col) -= data_min;
        data.col(col) /= data_step;
    }
    data.convertTo(data, CV_8U);
}

class WaldBoost
{
public:
    WaldBoost(int weak_count):
        weak_count_(weak_count),
        thresholds_(),
        alphas_(),
        feature_indices_(),
        polarities_(),
        cascade_thresholds_() {}

    WaldBoost():
        weak_count_(),
        thresholds_(),
        alphas_(),
        feature_indices_(),
        polarities_(),
        cascade_thresholds_() {}

    vector<int> get_feature_indices()
    {
        return feature_indices_;
    }

    void detect(Ptr<CvFeatureEvaluator> eval,
                const Mat& img, const vector<float>& scales,
                vector<Rect>& bboxes, Mat1f& confidences)
    {
        bboxes.clear();
        confidences.release();

        Mat resized_img;
        int step = 4;
        float h;
        for (size_t i = 0; i < scales.size(); ++i) {
            float scale = scales[i];
            resize(img, resized_img, Size(), scale, scale);
            eval->setImage(resized_img, 0, 0, feature_indices_);
            int n_rows = 24 / scale;
            int n_cols = 24 / scale;
            for (int r = 0; r + 24 < resized_img.rows; r += step) {
                for (int c = 0; c + 24 < resized_img.cols; c += step) {
                    //eval->setImage(resized_img(Rect(c, r, 24, 24)), 0, 0);
                    eval->setWindow(Point(c, r));
                    if (predict(eval, &h) == +1) {
                        int row = r / scale;
                        int col = c / scale;
                        bboxes.push_back(Rect(col, row, n_cols, n_rows));
                        confidences.push_back(h);
                    }
                }
            }
        }
        groupRectangles(bboxes, 3, 0.7);
    }

    void fit(Mat& data_pos, Mat& data_neg)
    {
        // data_pos: F x N_pos
        // data_neg: F x N_neg
        // every feature corresponds to row
        // every sample corresponds to column
        assert(data_pos.rows >= weak_count_);
        assert(data_pos.rows == data_neg.rows);

        vector<bool> feature_ignore;
        for (int i = 0; i < data_pos.rows; ++i) {
            feature_ignore.push_back(false);
        }

        Mat1f pos_weights(1, data_pos.cols, 1.0f / (2 * data_pos.cols));
        Mat1f neg_weights(1, data_neg.cols, 1.0f / (2 * data_neg.cols));
        Mat1f pos_trace(1, data_pos.cols, 0.0f);
        Mat1f neg_trace(1, data_neg.cols, 0.0f);

        bool quantize = false;
        if (data_pos.type() != CV_8U) {
            cerr << "quantize" << endl;
            quantize = true;
        }

        Mat1f data_min, data_step;
        int n_bins = 256;
        if (quantize) {
            compute_min_step(data_pos, data_neg, n_bins, data_min, data_step);
            quantize_data(data_pos, data_min, data_step);
            quantize_data(data_neg, data_min, data_step);
        }

        cerr << "pos=" << data_pos.cols << " neg=" << data_neg.cols << endl;
        for (int i = 0; i < weak_count_; ++i) {
            // Train weak learner with lowest error using weights
            double min_err = DBL_MAX;
            int min_feature_ind = -1;
            int min_polarity = 0;
            int threshold_q = 0;
            float min_threshold = 0;
#pragma omp parallel for
            for (int feat_i = 0; feat_i < data_pos.rows; ++feat_i) {
                if (feature_ignore[feat_i])
                    continue;

                // Construct cdf
                Mat1f pos_cdf(1, n_bins), neg_cdf(1, n_bins);
                compute_cdf(data_pos.row(feat_i), pos_weights, pos_cdf);
                compute_cdf(data_neg.row(feat_i), neg_weights, neg_cdf);

                float neg_total = sum(neg_weights)[0];
                Mat1f err_direct = pos_cdf + neg_total - neg_cdf;
                Mat1f err_backward = 1.0f - err_direct;

                int idx1[2], idx2[2];
                double err1, err2;
                minMaxIdx(err_direct, &err1, NULL, idx1);
                minMaxIdx(err_backward, &err2, NULL, idx2);
#pragma omp critical
                {
                if (min(err1, err2) < min_err) {
                    if (err1 < err2) {
                        min_err = err1;
                        min_polarity = +1;
                        threshold_q = idx1[1];
                    } else {
                        min_err = err2;
                        min_polarity = -1;
                        threshold_q = idx2[1];
                    }
                    min_feature_ind = feat_i;
                    if (quantize) {
                        min_threshold = data_min(feat_i, 0) + data_step(feat_i, 0) *
                            (threshold_q + .5f);
                    } else {
                        min_threshold = threshold_q + .5f;
                    }
                }
                }
            }


            float alpha = .5f * log((1 - min_err) / min_err);
            alphas_.push_back(alpha);
            feature_indices_.push_back(min_feature_ind);
            thresholds_.push_back(min_threshold);
            polarities_.push_back(min_polarity);
            feature_ignore[min_feature_ind] = true;


            double loss = 0;
            // Update positive weights
            for (int j = 0; j < data_pos.cols; ++j) {
                int val = data_pos.at<unsigned char>(min_feature_ind, j);
                int label = min_polarity * (val - threshold_q) >= 0 ? +1 : -1;
                pos_weights(0, j) *= exp(-alpha * label);
                pos_trace(0, j) += alpha * label;
                loss += exp(-pos_trace(0, j)) / (2.0f * data_pos.cols);
            }
            double min_pos, max_neg = -100000;
            minMaxIdx(pos_trace, &min_pos, NULL);

            // Update negative weights
            for (int j = 0; j < data_neg.cols; ++j) {
                int val = data_neg.at<unsigned char>(min_feature_ind, j);
                int label = min_polarity * (val - threshold_q) >= 0 ? +1 : -1;
                neg_weights(0, j) *= exp(alpha * label);
                neg_trace(0, j) += alpha * label;
                loss += exp(+neg_trace(0, j)) / (2.0f * data_neg.cols);
            }

            // Compute threshold
            double a = 0.02;
            int col_pos = 0, col_neg = 0;
            int cur_pos = 0, cur_neg = 0;
            int max_col = -1;
            bool max_pos = false;
            for (;
                 col_pos < data_pos.cols &&
                 col_neg < data_neg.cols; ) {
                bool pos = false;
                if (data_pos.at<uchar>(min_feature_ind, col_pos) <
                    data_pos.at<uchar>(min_feature_ind, col_neg)) {
                    col_pos += 1;
                    cur_pos += 1;
                    pos = true;
                } else {
                    col_neg += 1;
                    cur_neg += 1;
                }
                float p_neg = cur_neg / (float)data_neg.cols;
                float p_pos = cur_pos / (float)data_pos.cols;
                if (a * p_neg > p_pos) {
                    if (pos)
                        max_col = col_pos;
                    else
                        max_col = col_neg;
                    max_pos = pos;
                }
            }

            if (max_pos) {
                cascade_thresholds_.push_back(pos_trace(0, max_col));
            } else {
                cascade_thresholds_.push_back(neg_trace(0, max_col));
            }

            cerr << "i=" << setw(4) << i;
            cerr << " feat=" << setw(5) << min_feature_ind;
            cerr << " thr=" << setw(3) << threshold_q;
            cerr <<  " alpha=" << fixed << setprecision(3)
                 << alpha << " err=" << fixed << setprecision(3) << min_err
                 << " loss=" << scientific << loss << endl;

            if (loss < 1e-50 || min_err > 0.5) {
                cerr << "Stopping early" << endl;
                weak_count_ = i + 1;
                break;
            }

            // Normalize weights
            double z = (sum(pos_weights) + sum(neg_weights))[0];
            pos_weights /= z;
            neg_weights /= z;
        }
    }

    int predict(Ptr<CvFeatureEvaluator> eval, float *h) const
    {
        const float thr = -2.5;
        assert(feature_indices_.size() == size_t(weak_count_));
        float res = 0;
        for (int i = 0; i < weak_count_; ++i) {
            float val = (*eval)(feature_indices_[i], 0);
            int label = polarities_[i] * (val - thresholds_[i]) > 0 ? +1: -1;
            res += alphas_[i] * label;
            if (res < cascade_thresholds_[i]) {
                return -1;
            }
        }
        *h = res;
        return res > thr ? +1 : -1;
    }

    void save(const string& filename)
    {
        ofstream f(filename.c_str());
        f << weak_count_ << endl;
        for (size_t i = 0; i < thresholds_.size(); ++i) {
            f << thresholds_[i] << " ";
        }
        f << endl;
        for (size_t i = 0; i < alphas_.size(); ++i) {
            f << alphas_[i] << " ";
        }
        f << endl;
        for (size_t i = 0; i < polarities_.size(); ++i) {
            f << polarities_[i] << " ";
        }
        f << endl;
        for (size_t i = 0; i < cascade_thresholds_.size(); ++i) {
            f << cascade_thresholds_[i] << " ";
        }
        f << endl;
        for (size_t i = 0; i < feature_indices_.size(); ++i) {
            f << feature_indices_[i] << " ";
        }
        f << endl;
    }

    void load(const string& filename)
    {
        ifstream f(filename.c_str());
        f >> weak_count_;
        thresholds_.resize(weak_count_);
        alphas_.resize(weak_count_);
        polarities_.resize(weak_count_);
        cascade_thresholds_.resize(weak_count_);
        feature_indices_.resize(weak_count_);

        for (int i = 0; i < weak_count_; ++i) {
            f >> thresholds_[i];
        }
        for (int i = 0; i < weak_count_; ++i) {
            f >> alphas_[i];
        }
        for (int i = 0; i < weak_count_; ++i) {
            f >> polarities_[i];
        }
        for (int i = 0; i < weak_count_; ++i) {
            f >> cascade_thresholds_[i];
        }
        for (int i = 0; i < weak_count_; ++i) {
            f >> feature_indices_[i];
        }
    }

    void reset(int weak_count)
    {
        weak_count_ = weak_count;
        thresholds_.clear();
        alphas_.clear();
        feature_indices_.clear();
        polarities_.clear();
        cascade_thresholds_.clear();
    }

    ~WaldBoost()
    {
    }
private:
    int weak_count_;
    vector<float> thresholds_;
    vector<float> alphas_;
    vector<int> feature_indices_;
    vector<int> polarities_;
    vector<float> cascade_thresholds_;
};


void test_boosting();
void train_boosting();


int main(int argc, char **argv)
{
    string stage = argv[1];
    if (stage == "train") {
        train_boosting();
    } else if (stage == "detect") {
        test_boosting();
    }
}


vector<Mat>
read_imgs(const string& path)
{
    vector<String> filenames;
    glob(path, filenames);
    vector<Mat> imgs;
    for (size_t i = 0; i < filenames.size(); ++i) {
        imgs.push_back(imread(filenames[i], CV_LOAD_IMAGE_GRAYSCALE));
    }
    return imgs;
}
vector<Mat>
sample_patches(const string& path, int n_rows, int n_cols, int n_patches)
{
    vector<String> filenames;
    glob(path, filenames);
    vector<Mat> patches;
    int patch_count = 0;
    for (size_t i = 0; i < filenames.size(); ++i) {
        Mat img = imread(filenames[i], CV_LOAD_IMAGE_GRAYSCALE);
        for (int row = 0; row + n_rows < img.rows; row += n_rows) {
            for (int col = 0; col + n_cols < img.cols; col += n_cols) {
                patches.push_back(img(Rect(col, row, n_cols, n_rows)).clone());
                patch_count += 1;
                if (patch_count == n_patches) {
                    goto sampling_finished;
                }
            }
        }
    }
sampling_finished:
    return patches;
}

void train_boosting()
{
    cerr << "read imgs" << endl;
    vector<Mat> pos_imgs = read_imgs("/home/vlad/gsoc/lfw_faces");
    assert(pos_imgs.size());

    const char neg_path[] = "/home/vlad/gsoc/rtsd_bg";
    vector<Mat> neg_imgs = sample_patches(neg_path, 24, 24, pos_imgs.size());
    assert(neg_imgs.size());

    int n_features;
    Mat pos_data, neg_data;

    Ptr<CvFeatureEvaluator> eval = CvFeatureEvaluator::create(CvFeatureParams::LBP);
    eval->init(CvFeatureParams::create(CvFeatureParams::LBP), 1, Size(24, 24));
    n_features = eval->getNumFeatures();

    const int stages[] = {32, 64, 128, 256, 512, 1024, 2048, 4096};
    const int stage_count = sizeof(stages) / sizeof(*stages);
    const int stage_neg = 5000;
    const int max_per_image = 25;

    const float scales_arr[] = {.3, .4, .5, .6, .7, .8, .9, 1};
    const vector<float> scales(scales_arr,
            scales_arr + sizeof(scales_arr) / sizeof(*scales_arr));

    WaldBoost boost;
    vector<String> neg_filenames;
    glob(neg_path, neg_filenames);


    for (int i = 0; i < stage_count; ++i) {

        cerr << "compute features" << endl;

        pos_data = Mat1b(n_features, pos_imgs.size());
        neg_data = Mat1b(n_features, neg_imgs.size());

        for (size_t k = 0; k < pos_imgs.size(); ++k) {
            eval->setImage(pos_imgs[k], +1, 0, boost.get_feature_indices());
            for (int j = 0; j < n_features; ++j) {
                pos_data.at<uchar>(j, k) = (*eval)(j, 0);
            }
        }

        for (size_t k = 0; k < neg_imgs.size(); ++k) {
            eval->setImage(neg_imgs[k], 0, 0, boost.get_feature_indices());
            for (int j = 0; j < n_features; ++j) {
                neg_data.at<uchar>(j, k) = (*eval)(j, 0);
            }
        }


        boost.reset(stages[i]);
        boost.fit(pos_data, neg_data);

        if (i + 1 == stage_count) {
            break;
        }

        int bootstrap_count = 0;
        int img_i = 0;
        for (; img_i < neg_filenames.size(); ++img_i) {
            cerr << "win " << bootstrap_count << "/" << stage_neg
                 << " img " << (img_i + 1) << "/" << neg_filenames.size() << "\r";
            Mat img = imread(neg_filenames[img_i], CV_LOAD_IMAGE_GRAYSCALE);
            vector<Rect> bboxes;
            Mat1f confidences;
            boost.detect(eval, img, scales, bboxes, confidences);

            if (confidences.rows > 0) {
                Mat1i indices;
                sortIdx(confidences, indices,
                        CV_SORT_EVERY_COLUMN + CV_SORT_DESCENDING);

                int win_count = min(max_per_image, confidences.rows);
                win_count = min(win_count, stage_neg - bootstrap_count);
                Mat window;
                for (int k = 0; k < win_count; ++k) {
                    resize(img(bboxes[indices(k, 0)]), window, Size(24, 24));
                    neg_imgs.push_back(window.clone());
                    bootstrap_count += 1;
                }
                if (bootstrap_count >= stage_neg) {
                    break;
                }
            }
        }
        cerr << "bootstrapped " << bootstrap_count << " windows from "
             << (img_i + 1) << " images" << endl;
    }

    boost.save("models/model.txt");
    vector<int> feature_indices = boost.get_feature_indices();
    Mat1i feature_map(1, feature_indices.size());
    for (size_t i = 0; i < feature_indices.size(); ++i) {
        feature_map(0, i) = feature_indices[i];
    }
    FileStorage fs("models/features.yaml", FileStorage::WRITE);
    eval->writeFeatures(fs, feature_map);
}

void test_boosting()
{
    WaldBoost boost;
    const char model_filename[] = "models/model.txt";
    boost.load(model_filename);

    Mat test_img = imread("imgs/test4.png", CV_LOAD_IMAGE_GRAYSCALE);
    vector<Rect> bboxes;
    Mat1f confidences;
    vector<float> scales;
    for (float scale = 0.2f; scale < 1.2f; scale *= 1.2) {
        scales.push_back(scale);
    }
    Ptr<CvFeatureParams> params = CvFeatureParams::create(CvFeatureParams::LBP);
    Ptr<CvFeatureEvaluator> eval = CvFeatureEvaluator::create(CvFeatureParams::LBP);
    eval->init(params, 1, Size(24, 24));
    boost.detect(eval, test_img, scales, bboxes, confidences);
    cerr << "detected " << bboxes.size() << " objects" << endl;

    for (size_t i = 0; i < bboxes.size(); ++i) {
        rectangle(test_img, bboxes[i], Scalar(255, 0, 0));
    }
    imwrite("imgs/out.png", test_img);
}

