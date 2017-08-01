/*
 *  By downloading, copying, installing or using the software you agree to this license.
 *  If you do not agree to this license, do not download, install,
 *  copy or use the software.
 *
 *
 *  License Agreement
 *  For Open Source Computer Vision Library
 *  (3 - clause BSD License)
 *
 *  Redistribution and use in source and binary forms, with or without modification,
 *  are permitted provided that the following conditions are met :
 *
 *  * Redistributions of source code must retain the above copyright notice,
 *  this list of conditions and the following disclaimer.
 *
 *  * Redistributions in binary form must reproduce the above copyright notice,
 *  this list of conditions and the following disclaimer in the documentation
 *  and / or other materials provided with the distribution.
 *
 *  * Neither the names of the copyright holders nor the names of the contributors
 *  may be used to endorse or promote products derived from this software
 *  without specific prior written permission.
 *
 *  This software is provided by the copyright holders and contributors "as is" and
 *  any express or implied warranties, including, but not limited to, the implied
 *  warranties of merchantability and fitness for a particular purpose are disclaimed.
 *  In no event shall copyright holders or contributors be liable for any direct,
 *  indirect, incidental, special, exemplary, or consequential damages
 *  (including, but not limited to, procurement of substitute goods or services;
 *  loss of use, data, or profits; or business interruption) however caused
 *  and on any theory of liability, whether in contract, strict liability,
 *  or tort(including negligence or otherwise) arising in any way out of
 *  the use of this software, even if advised of the possibility of such damage.
 */


#include <Eigen/Dense>
#include <Eigen/SparseCore>
#include <Eigen/SparseCholesky>
#include <Eigen/IterativeLinearSolvers>
#include <Eigen/Sparse>




#include "precomp.hpp"
// #include<opencv2/core/core.hpp>
// #include<opencv2/core/eigen.hpp>
// #include<opencv2/highgui.hpp>
// #include<opencv2/opencv.hpp>
// #include <opencv2/ximgproc.hpp>



#include <cmath>
#include <chrono>
#include <vector>
#include <memory>
#include <stdlib.h>
#include <iostream>
#include <iterator>
#include <algorithm>
#include <unordered_map>

namespace cv
{
namespace ximgproc
{

    class FastBilateralSolverFilterImpl : public FastBilateralSolverFilter
    {
    public:

        static Ptr<FastBilateralSolverFilterImpl> create(InputArray guide, double sigma_spatial, double sigma_luma, double sigma_chroma)
        {
            CV_Assert( guide.type() == CV_8UC3 );
            FastBilateralSolverFilterImpl *fbs = new FastBilateralSolverFilterImpl();
            Mat gui = guide.getMat();
            fbs->init(gui,sigma_spatial,sigma_luma,sigma_chroma);
            return Ptr<FastBilateralSolverFilterImpl>(fbs);
        }

        // FastBilateralSolverFilterImpl(){}

        void filter(InputArray& src, InputArray& confidence, OutputArray& dst)
        {

            CV_Assert( src.type() == CV_8UC1 && confidence.type() == CV_8UC1 && src.size() == confidence.size() );
            if (src.rows() != rows || src.cols() != cols)
            {
                CV_Error(Error::StsBadSize, "Size of the filtered image must be equal to the size of the guide image");
                return;
            }

            dst.create(src.size(), src.type());
            Mat tar = src.getMat();
            Mat con = confidence.getMat();
            Mat out = dst.getMat();

            solve(tar,con,out);
        }

    // protected:
        void solve(cv::Mat& src, cv::Mat& confidence, cv::Mat& dst);
        void init(cv::Mat& reference_bgr, double sigma_spatial, double sigma_luma, double sigma_chroma);

        void Splat(Eigen::VectorXf& input, Eigen::VectorXf& dst);
        void Blur(Eigen::VectorXf& input, Eigen::VectorXf& dst);
        void Slice(Eigen::VectorXf& input, Eigen::VectorXf& dst);

    private:

        int npixels;
        int nvertices;
        int dim;
        int cols;
        int rows;
        std::vector<Eigen::SparseMatrix<float, Eigen::ColMajor> > blurs;
        std::vector<int> splat_idx;
        std::vector<std::pair<int, int>> blur_idx;
        Eigen::VectorXf m;
        Eigen::VectorXf n;
        Eigen::SparseMatrix<float, Eigen::ColMajor> blurs_test;
        Eigen::SparseMatrix<float, Eigen::ColMajor> S;
        Eigen::SparseMatrix<float, Eigen::ColMajor> Dn;
        Eigen::SparseMatrix<float, Eigen::ColMajor> Dm;

          struct grid_params
          {
              float spatialSigma;
              float lumaSigma;
              float chromaSigma;
              grid_params()
              {
                  spatialSigma = 8.0;
                  lumaSigma = 4.0;
                  chromaSigma = 4.0;
              }
          };

          struct bs_params
          {
              float lam;
              float A_diag_min;
              float cg_tol;
              int cg_maxiter;
              bs_params()
              {
                  lam = 128.0;
                  A_diag_min = 1e-5;
                  cg_tol = 1e-5;
                  cg_maxiter = 25;
              }
          };

        grid_params grid_param;
        bs_params bs_param;

    };



    void FastBilateralSolverFilterImpl::init(cv::Mat& reference_bgr, double sigma_spatial, double sigma_luma, double sigma_chroma)
    {

	      cv::Mat reference_yuv;
	      cv::cvtColor(reference_bgr, reference_yuv, COLOR_BGR2YCrCb);

	      std::chrono::steady_clock::time_point begin_grid_construction = std::chrono::steady_clock::now();

	      cols = reference_yuv.cols;
	      rows = reference_yuv.rows;
        npixels = cols*rows;
	      std::int64_t hash_vec[5];
	      for (int i = 0; i < 5; ++i)
		        hash_vec[i] = static_cast<std::int64_t>(std::pow(255, i));

	      std::unordered_map<std::int64_t /* hash */, int /* vert id */> hashed_coords;
	      hashed_coords.reserve(cols*rows);

	      const unsigned char* pref = (const unsigned char*)reference_yuv.data;
	      int vert_idx = 0;
	      int pix_idx = 0;


      	// construct Splat(Slice) matrices
        splat_idx.resize(npixels);
    	  for (int y = 0; y < rows; ++y)
      	{
    	    for (int x = 0; x < cols; ++x)
      		{
      			std::int64_t coord[5];
      			coord[0] = int(x / sigma_spatial);
      			coord[1] = int(y / sigma_spatial);
      			coord[2] = int(pref[0] / sigma_luma);
      			coord[3] = int(pref[1] / sigma_chroma);
      			coord[4] = int(pref[2] / sigma_chroma);

      			// convert the coordinate to a hash value
      			std::int64_t hash_coord = 0;
      			for (int i = 0; i < 5; ++i)
      				  hash_coord += coord[i] * hash_vec[i];

      			// pixels whom are alike will have the same hash value.
      			// We only want to keep a unique list of hash values, therefore make sure we only insert
      			// unique hash values.
      			auto it = hashed_coords.find(hash_coord);
      			if (it == hashed_coords.end())
      			{
      				hashed_coords.insert(std::pair<std::int64_t, int>(hash_coord, vert_idx));
              splat_idx[pix_idx] = vert_idx;
      				++vert_idx;
      			}
      			else
      			{
              splat_idx[pix_idx] = it->second;
      			}

      			pref += 3; // skip 3 bytes (y u v)
      			++pix_idx;
      		}
      	}
        nvertices = hashed_coords.size();


      	// construct Blur matrices
      	std::chrono::steady_clock::time_point begin_blur_construction = std::chrono::steady_clock::now();
        Eigen::VectorXf ones_nvertices = Eigen::VectorXf::Ones(nvertices);
        Eigen::VectorXf ones_npixels = Eigen::VectorXf::Ones(npixels);
        blurs_test = ones_nvertices.asDiagonal();
        blurs_test *= 10;
        for(int offset = -1; offset <= 1;++offset)
        {
            if(offset == 0) continue;
          	for (int i = 0; i < 5; ++i)
          	{
          	     Eigen::SparseMatrix<float, Eigen::ColMajor> blur_temp(hashed_coords.size(), hashed_coords.size());
                 blur_temp.reserve(Eigen::VectorXi::Constant(nvertices,6));
          		   std::int64_t offset_hash_coord = offset * hash_vec[i];
        		     for (auto it = hashed_coords.begin(); it != hashed_coords.end(); ++it)
      		       {
      			         std::int64_t neighb_coord = it->first + offset_hash_coord;
      			         auto it_neighb = hashed_coords.find(neighb_coord);
      			         if (it_neighb != hashed_coords.end())
          			     {
                         blur_temp.insert(it->second,it_neighb->second) = 1.0f;
                         blur_idx.push_back(std::pair<int,int>(it->second, it_neighb->second));
          			     }
          		   }
                 blurs_test += blur_temp;
              }
        }
        blurs_test.finalize();


        //bistochastize
        int maxiter = 10;
        n = ones_nvertices;
        m = Eigen::VectorXf::Zero(nvertices);
        for (int i = 0; i < splat_idx.size(); i++) {
            m(splat_idx[i]) += 1.0f;
        }

        Eigen::VectorXf bluredn(nvertices);

        for (int i = 0; i < maxiter; i++) {
            Blur(n,bluredn);
            n = ((n.array()*m.array()).array()/bluredn.array()).array().sqrt();
        }
        Blur(n,bluredn);

        m = n.array() * (bluredn).array();
        Dm = m.asDiagonal();
        Dn = n.asDiagonal();

        int debugn = blurs_test.nonZeros(); //FIXME: if don't call nonZeros(), the result will be destroy
        // std::cout << "Splat"<< splat_idx.size() << '\n';
        // std::cout << "Blur"<< blurs_test.nonZeros() << '\n';
        // std::cout << "Dn"<< Dn.nonZeros() << '\n';
        // std::cout << "Dm"<< Dm.nonZeros() << '\n';
        // std::cout << "test" << '\n';
    }

    void FastBilateralSolverFilterImpl::Splat(Eigen::VectorXf& input, Eigen::VectorXf& output)
    {
        output.setZero();
        for (int i = 0; i < splat_idx.size(); i++) {
            output(splat_idx[i]) += input(i);
        }

    }

    void FastBilateralSolverFilterImpl::Blur(Eigen::VectorXf& input, Eigen::VectorXf& output)
    {
        output.setZero();
        output = input * 10;
        for (int i = 0; i < blur_idx.size(); i++)
        {
            output(blur_idx[i].first) += input(blur_idx[i].second);
        }
    }


    void FastBilateralSolverFilterImpl::Slice(Eigen::VectorXf& input, Eigen::VectorXf& output)
    {
        output.setZero();
        for (int i = 0; i < splat_idx.size(); i++) {
            output(i) = input(splat_idx[i]);
        }
    }


    void FastBilateralSolverFilterImpl::solve(cv::Mat& target,
               cv::Mat& confidence,
               cv::Mat& output)
    {

        Eigen::VectorXf x(npixels);
        Eigen::VectorXf w(npixels);

      	const uchar *pft = reinterpret_cast<const uchar*>(target.data);
      	for (int i = 0; i < npixels; i++)
      	{
      		  x(i) = float(pft[i])/255.0f;
        }
      	const uchar *pfc = reinterpret_cast<const uchar*>(confidence.data);
      	for (int i = 0; i < npixels; i++)
      	{
      		  w(i) = float(pfc[i])/255.0f;
        }

        Eigen::SparseMatrix<float, Eigen::ColMajor> M(nvertices,nvertices);
        Eigen::SparseMatrix<float, Eigen::ColMajor> A_data(nvertices,nvertices);
        Eigen::SparseMatrix<float, Eigen::ColMajor> A(nvertices,nvertices);
        Eigen::VectorXf b(nvertices);
        Eigen::VectorXf y(nvertices);
        Eigen::VectorXf w_splat(nvertices);
        Eigen::VectorXf xw(x.size());



        //construct A
        Splat(w,w_splat);
        A_data = (w_splat).asDiagonal();
        A = bs_param.lam * (Dm - Dn * (blurs_test*Dn)) + A_data ;

        //construct b
        b.setZero();
        for (int i = 0; i < splat_idx.size(); i++) {
            b(splat_idx[i]) += x(i) * w(i);
        }

        // solve Ay = b
        Eigen::ConjugateGradient<Eigen::SparseMatrix<float>, Eigen::Lower|Eigen::Upper> cg;
        cg.compute(A);
        cg.setMaxIterations(bs_param.cg_maxiter);
        cg.setTolerance(bs_param.cg_tol);
        y = cg.solve(b);
        // std::cout << "#iterations:     " << cg.iterations() << std::endl;
        // std::cout << "estimated error: " << cg.error()      << std::endl;

        //slice
      	uchar *pftar = (uchar*)(output.data);
      	for (int i = 0; i < splat_idx.size(); i++)
      	{
      		  pftar[i] = y(splat_idx[i]) * 255;
        }


    }


////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////
CV_EXPORTS_W
Ptr<FastBilateralSolverFilter> createFastBilateralSolverFilter(InputArray guide, double sigma_spatial, double sigma_luma, double sigma_chroma)
{
    return Ptr<FastBilateralSolverFilter>(FastBilateralSolverFilterImpl::create(guide, sigma_spatial, sigma_luma, sigma_chroma));
}

CV_EXPORTS_W
void fastBilateralSolverFilter(InputArray guide, InputArray src, InputArray confidence, OutputArray dst, double sigma_spatial, double sigma_luma, double sigma_chroma)
{
    Ptr<FastBilateralSolverFilter> fbs = createFastBilateralSolverFilter(guide, sigma_spatial, sigma_luma, sigma_chroma);
    fbs->filter(src, confidence, dst);
}



}

}
