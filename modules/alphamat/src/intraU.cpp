// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "precomp.hpp"
#include "intraU.hpp"

namespace cv { namespace alphamat {

int findColMajorInd(int rowMajorInd, int nRows, int nCols)
{
    int iInd = rowMajorInd / nCols;
    int jInd = rowMajorInd % nCols;
    return (jInd * nRows + iInd);
}

static
void generateFVectorIntraU(my_vector_of_vectors_t& samples, Mat& img, Mat& tmap, std::vector<int>& orig_ind)
{
    int nRows = img.rows;
    int nCols = img.cols;
    int unk_count = 0;
    int i, j;
    for (i = 0; i < nRows; ++i)
    {
        for (j = 0; j < nCols; ++j)
        {
            uchar pix = tmap.at<uchar>(i, j);
            if (pix == 128)
                unk_count++;
        }
    }
    samples.resize(unk_count);
    orig_ind.resize(unk_count);

    int c1 = 0;
    for (i = 0; i < nRows; ++i)
    {
        for (j = 0; j < nCols; ++j)
        {
            uchar pix = tmap.at<uchar>(i, j);
            if (pix == 128)  // collection of unknown pixels samples
            {
                samples[c1].resize(ALPHAMAT_DIM);
                samples[c1][0] = img.at<cv::Vec3b>(i, j)[0] / 255.0;
                samples[c1][1] = img.at<cv::Vec3b>(i, j)[1] / 255.0;
                samples[c1][2] = img.at<cv::Vec3b>(i, j)[2] / 255.0;
                samples[c1][3] = (double(i + 1) / nRows) / 20;
                samples[c1][4] = (double(j + 1) / nCols) / 20;
                orig_ind[c1] = i * nCols + j;
                c1++;
            }
        }
    }

    CV_LOG_INFO(NULL, "ALPHAMAT: Total number of unknown pixels : " << c1);
}

static
void kdtree_intraU(Mat& img, Mat& tmap, my_vector_of_vectors_t& indm, my_vector_of_vectors_t& samples, std::vector<int>& orig_ind)
{
    // Generate feature vectors for intra U:
    generateFVectorIntraU(samples, img, tmap, orig_ind);

    typedef KDTreeVectorOfVectorsAdaptor<my_vector_of_vectors_t, double> my_kd_tree_t;
    my_kd_tree_t mat_index(ALPHAMAT_DIM /*dim*/, samples, 10 /* max leaf */);
    mat_index.index->buildIndex();
    // do a knn search with ku  = 5
    const size_t num_results = 5 + 1;

    int N = samples.size();  // no. of unknown samples

    std::vector<size_t> ret_indexes(num_results);
    std::vector<double> out_dists_sqr(num_results);
    nanoflann::KNNResultSet<double> resultSet(num_results);

    indm.resize(N);
    for (int i = 0; i < N; i++)
    {
        resultSet.init(&ret_indexes[0], &out_dists_sqr[0]);
        mat_index.index->findNeighbors(resultSet, &samples[i][0], nanoflann::SearchParams(10));

        indm[i].resize(num_results - 1);
        for (std::size_t j = 1; j < num_results; j++)
        {
            indm[i][j - 1] = ret_indexes[j];
        }
    }
}

static
double l1norm(std::vector<double>& x, std::vector<double>& y)
{
    double sum = 0;
    for (int i = 0; i < ALPHAMAT_DIM; i++)
        sum += abs(x[i] - y[i]);
    return sum / ALPHAMAT_DIM;
}

static
void intraU(Mat& img, my_vector_of_vectors_t& indm, my_vector_of_vectors_t& samples,
        std::vector<int>& orig_ind, SparseMatrix<double>& Wuu, SparseMatrix<double>& Duu)
{
    // input: indm, samples
    int n = indm.size();  // num of unknown samples
    CV_LOG_INFO(NULL, "ALPHAMAT: num of unknown samples, n : " << n);

    int i, j, nbr_ind;
    for (i = 0; i < n; i++)
    {
        samples[i][3] *= 1 / 100;
        samples[i][4] *= 1 / 100;
    }

    my_vector_of_vectors_t weights;
    typedef Triplet<double> T;
    std::vector<T> triplets, td;

    double weight;
    for (i = 0; i < n; i++)
    {
        int num_nbr = indm[i].size();
        int cMaj_i = findColMajorInd(orig_ind[i], img.rows, img.cols);
        for (j = 0; j < num_nbr; j++)
        {
            nbr_ind = indm[i][j];
            int cMaj_nbr_j = findColMajorInd(orig_ind[nbr_ind], img.rows, img.cols);
            weight = max(1 - l1norm(samples[i], samples[j]), 0.0);

            triplets.push_back(T(cMaj_i, cMaj_nbr_j, weight / 2));
            td.push_back(T(cMaj_i, cMaj_i, weight / 2));

            triplets.push_back(T(cMaj_nbr_j, cMaj_i, weight / 2));
            td.push_back(T(cMaj_nbr_j, cMaj_nbr_j, weight / 2));
        }
    }

    Wuu.setFromTriplets(triplets.begin(), triplets.end());
    Duu.setFromTriplets(td.begin(), td.end());
}

void UU(Mat& image, Mat& tmap, SparseMatrix<double>& Wuu, SparseMatrix<double>& Duu)
{
    my_vector_of_vectors_t samples, indm;
    std::vector<int> orig_ind;

    kdtree_intraU(image, tmap, indm, samples, orig_ind);
    intraU(image, indm, samples, orig_ind, Wuu, Duu);
    CV_LOG_INFO(NULL, "ALPHAMAT: Intra U Done");
}

}}  // namespace cv::alphamat
