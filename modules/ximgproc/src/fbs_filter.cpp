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

#include "precomp.hpp"

#include <cmath>
#include <vector>
#include <memory>
#include <stdlib.h>
#include <iostream>
#include <iterator>
#include <algorithm>


#ifdef HAVE_EIGEN
#  if defined __GNUC__ && defined __APPLE__
#    pragma GCC diagnostic ignored "-Wshadow"
#  endif
#  include <Eigen/Dense>
#  include <Eigen/SparseCore>
#  include <Eigen/SparseCholesky>
#  include <Eigen/IterativeLinearSolvers>
#  include <Eigen/Sparse>




#if __cplusplus <= 199711L
    #include <map>
    typedef std::map<long long /* hash */, int /* vert id */>  mapId;
#else
    #include <unordered_map>
    typedef std::unordered_map<long long /* hash */, int /* vert id */>  mapId;
#endif

namespace cv
{
namespace ximgproc
{

    class FastBilateralSolverFilterImpl : public FastBilateralSolverFilter
    {
    public:

        static Ptr<FastBilateralSolverFilterImpl> create(InputArray guide, double sigma_spatial, double sigma_luma, double sigma_chroma, double lambda, int num_iter, double max_tol)
        {
            CV_Assert(guide.type() == CV_8UC1 || guide.type() == CV_8UC3);
            FastBilateralSolverFilterImpl *fbs = new FastBilateralSolverFilterImpl();
            Mat gui = guide.getMat();
            fbs->init(gui,sigma_spatial,sigma_luma,sigma_chroma,lambda,num_iter,max_tol);
            return Ptr<FastBilateralSolverFilterImpl>(fbs);
        }

        void filter(InputArray src, InputArray confidence, OutputArray dst) CV_OVERRIDE
        {

            CV_Assert(!src.empty() && (src.depth() == CV_8U || src.depth() == CV_16S || src.depth() == CV_16U || src.depth() == CV_32F) && src.channels()<=4);
            CV_Assert(!confidence.empty() && (confidence.depth() == CV_8U || confidence.depth() == CV_32F) && confidence.channels()==1);
            if (src.rows() != rows || src.cols() != cols)
            {
                CV_Error(Error::StsBadSize, "Size of the filtered image must be equal to the size of the guide image");
                return;
            }
            if (confidence.rows() != rows || confidence.cols() != cols)
            {
                CV_Error(Error::StsBadSize, "Size of the confidence image must be equal to the size of the guide image");
                return;
            }

            std::vector<Mat> src_channels;
            std::vector<Mat> dst_channels;
            if(src.channels()==1)
                src_channels.push_back(src.getMat());
            else
                split(src,src_channels);

            Mat conf = confidence.getMat();

            for(int i=0;i<src.channels();i++)
            {
                Mat cur_res = src_channels[i].clone();

                solve(cur_res,conf,cur_res);
                cur_res.convertTo(cur_res, src.type());
                dst_channels.push_back(cur_res);
            }

            dst.create(src.size(),src_channels[0].type());
            if(src.channels()==1)
            {
                Mat& dstMat = dst.getMatRef();
                dstMat = dst_channels[0];
            }
            else
            merge(dst_channels,dst);
            CV_Assert(src.type() == dst.type() && src.size() == dst.size());
        }

    // protected:
        void solve(cv::Mat& src, cv::Mat& confidence, cv::Mat& dst);
        void init(cv::Mat& reference, double sigma_spatial, double sigma_luma, double sigma_chroma, double lambda, int num_iter, double max_tol);

        void Splat(Eigen::VectorXf& input, Eigen::VectorXf& dst);
        void Blur(Eigen::VectorXf& input, Eigen::VectorXf& dst);
        void Slice(Eigen::VectorXf& input, Eigen::VectorXf& dst);

        void diagonal(Eigen::VectorXf& v,Eigen::SparseMatrix<float>& mat)
        {
            mat = Eigen::SparseMatrix<float>(v.size(),v.size());
            for (int i = 0; i < int(v.size()); i++)
            {
                mat.insert(i,i) = v(i);
            }
        }



    private:

        int npixels;
        int nvertices;
        int dim;
        int cols;
        int rows;
        std::vector<int> splat_idx;
        std::vector<std::pair<int, int> > blur_idx;
        Eigen::VectorXf m;
        Eigen::VectorXf n;
        Eigen::SparseMatrix<float, Eigen::ColMajor> blurs;
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
                lumaSigma = 8.0;
                chromaSigma = 8.0;
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
                A_diag_min = 1e-5f;
                cg_tol = 1e-5f;
                cg_maxiter = 25;
            }
        };

        grid_params grid_param;
        bs_params bs_param;

    };



    void FastBilateralSolverFilterImpl::init(cv::Mat& reference, double sigma_spatial, double sigma_luma, double sigma_chroma, double lambda, int num_iter, double max_tol)
    {

        bs_param.lam = lambda;
        bs_param.cg_maxiter = num_iter;
        bs_param.cg_tol = max_tol;

        if(reference.channels()==1)
        {
            dim = 3;
            cols = reference.cols;
            rows = reference.rows;
            npixels = cols*rows;
            long long hash_vec[3];
            for (int i = 0; i < 3; ++i)
            hash_vec[i] = static_cast<long long>(std::pow(255, i));

            mapId hashed_coords;
#if __cplusplus <= 199711L
#else
            hashed_coords.reserve(cols*rows);
#endif

            const unsigned char* pref = (const unsigned char*)reference.data;
            int vert_idx = 0;
            int pix_idx = 0;

            // construct Splat(Slice) matrices
            splat_idx.resize(npixels);
            for (int y = 0; y < rows; ++y)
            {
                for (int x = 0; x < cols; ++x)
                {
                    long long coord[3];
                    coord[0] = int(x / sigma_spatial);
                    coord[1] = int(y / sigma_spatial);
                    coord[2] = int(pref[0] / sigma_luma);

                    // convert the coordinate to a hash value
                    long long hash_coord = 0;
                    for (int i = 0; i < 3; ++i)
                        hash_coord += coord[i] * hash_vec[i];

                    // pixels whom are alike will have the same hash value.
                    // We only want to keep a unique list of hash values, therefore make sure we only insert
                    // unique hash values.
                    mapId::iterator it = hashed_coords.find(hash_coord);
                    if (it == hashed_coords.end())
                    {
                        hashed_coords.insert(std::pair<long long, int>(hash_coord, vert_idx));
                        splat_idx[pix_idx] = vert_idx;
                        ++vert_idx;
                    }
                    else
                    {
                        splat_idx[pix_idx] = it->second;
                    }

                    pref += 1; // skip 1 bytes (y)
                    ++pix_idx;
                }
            }
            nvertices = static_cast<int>(hashed_coords.size());

            // construct Blur matrices
            Eigen::VectorXf ones_nvertices = Eigen::VectorXf::Ones(nvertices);
            diagonal(ones_nvertices,blurs);
            blurs *= 10;
            for(int offset = -1; offset <= 1;++offset)
            {
                if(offset == 0) continue;
                for (int i = 0; i < dim; ++i)
                {
                    Eigen::SparseMatrix<float, Eigen::ColMajor> blur_temp(hashed_coords.size(), hashed_coords.size());
                    blur_temp.reserve(Eigen::VectorXi::Constant(nvertices,6));
                    long long offset_hash_coord = offset * hash_vec[i];
                    for (mapId::iterator it = hashed_coords.begin(); it != hashed_coords.end(); ++it)
                    {
                        long long neighb_coord = it->first + offset_hash_coord;
                        mapId::iterator it_neighb = hashed_coords.find(neighb_coord);
                        if (it_neighb != hashed_coords.end())
                        {
                            blur_temp.insert(it->second,it_neighb->second) = 1.0f;
                            blur_idx.push_back(std::pair<int,int>(it->second, it_neighb->second));
                        }
                    }
                    blurs += blur_temp;
                }
            }
            blurs.finalize();

            //bistochastize
            int maxiter = 10;
            n = ones_nvertices;
            m = Eigen::VectorXf::Zero(nvertices);
            for (int i = 0; i < int(splat_idx.size()); i++)
            {
                m(splat_idx[i]) += 1.0f;
            }

            Eigen::VectorXf bluredn(nvertices);

            for (int i = 0; i < maxiter; i++)
            {
                Blur(n,bluredn);
                n = ((n.array()*m.array()).array()/bluredn.array()).array().sqrt();
            }
            Blur(n,bluredn);

            m = n.array() * (bluredn).array();
            diagonal(m,Dm);
            diagonal(n,Dn);

        }
        else
        {
            dim = 5;
            cv::Mat reference_yuv;
            cv::cvtColor(reference, reference_yuv, COLOR_BGR2YCrCb);

            cols = reference_yuv.cols;
            rows = reference_yuv.rows;
            npixels = cols*rows;
            long long hash_vec[5];
            for (int i = 0; i < 5; ++i)
            hash_vec[i] = static_cast<long long>(std::pow(255, i));

            mapId hashed_coords;
#if __cplusplus <= 199711L
#else
            hashed_coords.reserve(cols*rows);
#endif

            const unsigned char* pref = (const unsigned char*)reference_yuv.data;
            int vert_idx = 0;
            int pix_idx = 0;

            // construct Splat(Slice) matrices
            splat_idx.resize(npixels);
            for (int y = 0; y < rows; ++y)
            {
                for (int x = 0; x < cols; ++x)
                {
                    long long coord[5];
                    coord[0] = int(x / sigma_spatial);
                    coord[1] = int(y / sigma_spatial);
                    coord[2] = int(pref[0] / sigma_luma);
                    coord[3] = int(pref[1] / sigma_chroma);
                    coord[4] = int(pref[2] / sigma_chroma);

                    // convert the coordinate to a hash value
                    long long hash_coord = 0;
                    for (int i = 0; i < 5; ++i)
                        hash_coord += coord[i] * hash_vec[i];

                    // pixels whom are alike will have the same hash value.
                    // We only want to keep a unique list of hash values, therefore make sure we only insert
                    // unique hash values.
                    mapId::iterator it = hashed_coords.find(hash_coord);
                    if (it == hashed_coords.end())
                    {
                        hashed_coords.insert(std::pair<long long, int>(hash_coord, vert_idx));
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
            nvertices = static_cast<int>(hashed_coords.size());

            // construct Blur matrices
            Eigen::VectorXf ones_nvertices = Eigen::VectorXf::Ones(nvertices);
            diagonal(ones_nvertices,blurs);
            blurs *= 10;
            for(int offset = -1; offset <= 1;++offset)
            {
                if(offset == 0) continue;
                for (int i = 0; i < dim; ++i)
                {
                    Eigen::SparseMatrix<float, Eigen::ColMajor> blur_temp(hashed_coords.size(), hashed_coords.size());
                    blur_temp.reserve(Eigen::VectorXi::Constant(nvertices,6));
                    long long offset_hash_coord = offset * hash_vec[i];
                    for (mapId::iterator it = hashed_coords.begin(); it != hashed_coords.end(); ++it)
                    {
                        long long neighb_coord = it->first + offset_hash_coord;
                        mapId::iterator it_neighb = hashed_coords.find(neighb_coord);
                        if (it_neighb != hashed_coords.end())
                        {
                            blur_temp.insert(it->second,it_neighb->second) = 1.0f;
                            blur_idx.push_back(std::pair<int,int>(it->second, it_neighb->second));
                        }
                    }
                    blurs += blur_temp;
                }
            }
            blurs.finalize();


            //bistochastize
            int maxiter = 10;
            n = ones_nvertices;
            m = Eigen::VectorXf::Zero(nvertices);
            for (int i = 0; i < int(splat_idx.size()); i++)
            {
                m(splat_idx[i]) += 1.0f;
            }

            Eigen::VectorXf bluredn(nvertices);

            for (int i = 0; i < maxiter; i++)
            {
                Blur(n,bluredn);
                n = ((n.array()*m.array()).array()/bluredn.array()).array().sqrt();
            }
            Blur(n,bluredn);

            m = n.array() * (bluredn).array();
            diagonal(m,Dm);
            diagonal(n,Dn);
        }
    }

    void FastBilateralSolverFilterImpl::Splat(Eigen::VectorXf& input, Eigen::VectorXf& output)
    {
        output.setZero();
        for (int i = 0; i < int(splat_idx.size()); i++)
        {
            output(splat_idx[i]) += input(i);
        }

    }

    void FastBilateralSolverFilterImpl::Blur(Eigen::VectorXf& input, Eigen::VectorXf& output)
    {
        output.setZero();
        output = input * 10;
        for (int i = 0; i < int(blur_idx.size()); i++)
        {
            output(blur_idx[i].first) += input(blur_idx[i].second);
        }
    }


    void FastBilateralSolverFilterImpl::Slice(Eigen::VectorXf& input, Eigen::VectorXf& output)
    {
        output.setZero();
        for (int i = 0; i < int(splat_idx.size()); i++)
        {
            output(i) = input(splat_idx[i]);
        }
    }


    void FastBilateralSolverFilterImpl::solve(cv::Mat& target,
               cv::Mat& confidence,
               cv::Mat& output)
    {

        Eigen::SparseMatrix<float, Eigen::ColMajor> M(nvertices,nvertices);
        Eigen::SparseMatrix<float, Eigen::ColMajor> A_data(nvertices,nvertices);
        Eigen::SparseMatrix<float, Eigen::ColMajor> A(nvertices,nvertices);
        Eigen::VectorXf b(nvertices);
        Eigen::VectorXf y(nvertices);
        Eigen::VectorXf y0(nvertices);
        Eigen::VectorXf y1(nvertices);
        Eigen::VectorXf w_splat(nvertices);

        Eigen::VectorXf x(npixels);
        Eigen::VectorXf w(npixels);

        if(target.depth() == CV_16S)
        {
            const int16_t *pft = reinterpret_cast<const int16_t*>(target.data);
            for (int i = 0; i < npixels; i++)
            {
                x(i) = (cv::saturate_cast<float>(pft[i])+32768.0f)/65535.0f;
            }
        }
        else if(target.depth() == CV_16U)
        {
            const uint16_t *pft = reinterpret_cast<const uint16_t*>(target.data);
            for (int i = 0; i < npixels; i++)
            {
                x(i) = cv::saturate_cast<float>(pft[i])/65535.0f;
            }
        }
        else if(target.depth() == CV_8U)
        {
            const uchar *pft = reinterpret_cast<const uchar*>(target.data);
            for (int i = 0; i < npixels; i++)
            {
                x(i) = cv::saturate_cast<float>(pft[i])/255.0f;
            }
        }
        else if(target.depth() == CV_32F)
        {
            const float *pft = reinterpret_cast<const float*>(target.data);
            for (int i = 0; i < npixels; i++)
            {
                x(i) = pft[i];
            }
        }

        if(confidence.depth() == CV_8U)
        {
            const uchar *pfc = reinterpret_cast<const uchar*>(confidence.data);
            for (int i = 0; i < npixels; i++)
            {
                w(i) = cv::saturate_cast<float>(pfc[i])/255.0f;
            }
        }
        else if(confidence.depth() == CV_32F)
        {
            const float *pfc = reinterpret_cast<const float*>(confidence.data);
            for (int i = 0; i < npixels; i++)
            {
                w(i) = pfc[i];
            }
        }

        //construct A
        Splat(w,w_splat);

        diagonal(w_splat,A_data);
        A = bs_param.lam * (Dm - Dn * (blurs*Dn)) + A_data ;

        //construct b
        b.setZero();
        for (int i = 0; i < int(splat_idx.size()); i++)
        {
            b(splat_idx[i]) += x(i) * w(i);
        }

        //construct guess for y
        y0.setZero();
        for (int i = 0; i < int(splat_idx.size()); i++)
        {
            y0(splat_idx[i]) += x(i);
        }
        y1.setZero();
        for (int i = 0; i < int(splat_idx.size()); i++)
        {
            y1(splat_idx[i]) += 1.0f;
        }
        for (int i = 0; i < nvertices; i++)
        {
            y0(i) = y0(i)/y1(i);
        }


        // solve Ay = b
        Eigen::ConjugateGradient<Eigen::SparseMatrix<float>, Eigen::Lower|Eigen::Upper> cg;
        cg.compute(A);
        cg.setMaxIterations(bs_param.cg_maxiter);
        cg.setTolerance(bs_param.cg_tol);
        // y = cg.solve(b);
        y = cg.solveWithGuess(b,y0);
        std::cout << "#iterations:     " << cg.iterations() << std::endl;
        std::cout << "estimated error: " << cg.error()      << std::endl;

        //slice
        if(target.depth() == CV_16S)
        {
            int16_t *pftar = (int16_t*) output.data;
            for (int i = 0; i < int(splat_idx.size()); i++)
            {
                pftar[i] = cv::saturate_cast<short>(y(splat_idx[i]) * 65535.0f - 32768.0f);
            }
        }
        else if(target.depth() == CV_16U)
        {
            uint16_t *pftar = (uint16_t*) output.data;
            for (int i = 0; i < int(splat_idx.size()); i++)
            {
                pftar[i] = cv::saturate_cast<ushort>(y(splat_idx[i]) * 65535.0f);
            }
        }
        else if (target.depth() == CV_8U)
        {
            uchar *pftar = (uchar*) output.data;
            for (int i = 0; i < int(splat_idx.size()); i++)
            {
                pftar[i] = cv::saturate_cast<uchar>(y(splat_idx[i]) * 255.0f);
            }
        }
        else
        {
            float *pftar = (float*)(output.data);
            for (int i = 0; i < int(splat_idx.size()); i++)
            {
                pftar[i] = y(splat_idx[i]);
            }
        }


    }


////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////
Ptr<FastBilateralSolverFilter> createFastBilateralSolverFilter(InputArray guide, double sigma_spatial, double sigma_luma, double sigma_chroma, double lambda, int num_iter, double max_tol)
{
    return Ptr<FastBilateralSolverFilter>(FastBilateralSolverFilterImpl::create(guide, sigma_spatial, sigma_luma, sigma_chroma, lambda, num_iter, max_tol));
}

void fastBilateralSolverFilter(InputArray guide, InputArray src, InputArray confidence, OutputArray dst, double sigma_spatial, double sigma_luma, double sigma_chroma, double lambda, int num_iter, double max_tol)
{
    Ptr<FastBilateralSolverFilter> fbs = createFastBilateralSolverFilter(guide, sigma_spatial, sigma_luma, sigma_chroma, lambda, num_iter, max_tol);
    fbs->filter(src, confidence, dst);
}

}

}

#else

namespace cv
{
namespace ximgproc
{

Ptr<FastBilateralSolverFilter> createFastBilateralSolverFilter(InputArray, double, double, double, double, int, double)
{
    CV_Error(Error::StsNotImplemented, "createFastBilateralSolverFilter : needs to be compiled with EIGEN");
}

void fastBilateralSolverFilter(InputArray, InputArray, InputArray, OutputArray, double, double, double, double, int, double)
{
    CV_Error(Error::StsNotImplemented, "fastBilateralSolverFilter : needs to be compiled with EIGEN");
}


}

}

#endif // HAVE_EIGEN
