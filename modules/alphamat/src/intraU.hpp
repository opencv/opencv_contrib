// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

namespace cv{
  namespace alphamat{

using namespace Eigen;
using namespace nanoflann;

typedef std::vector<std::vector<double>> my_vector_of_vectors_t;
typedef std::vector<std::set<int, std::greater<int>>> my_vector_of_set_t;
std::vector<int> orig_ind;

void generateFVectorIntraU(my_vector_of_vectors_t &samples, Mat &img, Mat& tmap);

void kdtree_intraU(Mat &img, Mat& tmap, my_vector_of_vectors_t& indm, my_vector_of_set_t& inds, my_vector_of_vectors_t& samples);

double l1norm(std::vector<double>& x, std::vector<double>& y);

void intraU(my_vector_of_vectors_t& indm, my_vector_of_set_t& inds, my_vector_of_vectors_t& samples, SparseMatrix<double>& Wuu, SparseMatrix<double>& Duu);

void UU(Mat& image, Mat& tmap, SparseMatrix<double>& Wuu, SparseMatrix<double>& Duu);

}
}

/*

int main()
{
  Mat image,tmap;
  string img_path = "../../data/input_lowres/plasticbag.png";
  image = imread(img_path, CV_LOAD_IMAGE_COLOR);   // Read the file

  string tmap_path = "../../data/trimap_lowres/Trimap1/plasticbag.png";
    tmap = imread(tmap_path, CV_LOAD_IMAGE_GRAYSCALE);
    // SparseMatrix<double> Wuu = UU(image, tmap);

}

*/
