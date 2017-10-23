// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef __OPENCV_FOURIERDESCRIPTORS_HPP__
#define __OPENCV_FOURIERDESCRIPTORS_HPP__

#include <opencv2/core.hpp>

namespace cv {
    namespace ximgproc {

        //! @addtogroup ximgproc_shape
        //! @{

        /** @brief Class for ContourFitting algorithms.
        ContourFitting match two contours \f$ z_a \f$ and \f$ z_b \f$ minimizing distance
        \f[ d(z_a,z_b)=\sum (a_n - s  b_n e^{j(n \alpha +\phi )})^2 \f] where \f$ a_n \f$ and \f$ b_n \f$ are Fourier descriptors of \f$ z_a \f$ and \f$ z_b \f$ and s is a scaling factor and  \f$ \phi \f$ is angle rotation and \f$ \alpha \f$ is starting point factor adjustement
        */
        class CV_EXPORTS_W ContourFitting : public Algorithm
        {
            int ctrSize;
            int fdSize;
            std::vector<std::complex<double> > b;
            std::vector<std::complex<double> > a;
            std::vector<double> frequence;
            std::vector<double> rho, psi;
            void frequencyInit();
            void fAlpha(double x, double &fn, double &df);
            double distance(std::complex<double> r, double alpha);
            double  newtonRaphson(double x1, double x2);
        public:
            /** @brief Fit two closed curves using fourier descriptors. More details in @cite PersoonFu1977 and @cite BergerRaghunathan1998

            * @param ctr number of Fourier descriptors equal to number of contour points after resampling.
            * @param fd Contour defining second shape (Target).
            */
            ContourFitting(int ctr=1024,int fd=16):ctrSize(ctr),fdSize(fd){};
            /** @brief Fit two closed curves using fourier descriptors. More details in @cite PersoonFu1977 and @cite BergerRaghunathan1998

            @param src Contour defining first shape.
            @param dst Contour defining second shape (Target).
            @param _alphaPhiST : \f$ \alpha \f$=_alphaPhiST(0,0), \f$ \phi \f$=_alphaPhiST(0,1) (in radian), s=_alphaPhiST(0,2), Tx=_alphaPhiST(0,3), Ty=_alphaPhiST(0,4) rotation center
            @param dist distance between src and dst after matching.
            @param fdContour false then src and dst are contours and true src and dst are fourier descriptors.
            */
            CV_WRAP void estimateTransformation(InputArray src, InputArray dst, OutputArray _alphaPhiST, double *dist = 0, bool fdContour = false);
            /** @brief Fit two closed curves using fourier descriptors. More details in @cite PersoonFu1977 and @cite BergerRaghunathan1998

            @param src Contour defining first shape.
            @param dst Contour defining second shape (Target).
            @param _alphaPhiST : \f$ \alpha \f$=_alphaPhiST(0,0), \f$ \phi \f$=_alphaPhiST(0,1) (in radian), s=_alphaPhiST(0,2), Tx=_alphaPhiST(0,3), Ty=_alphaPhiST(0,4) rotation center
            @param dist distance between src and dst after matching.
            @param fdContour false then src and dst are contours and true src and dst are fourier descriptors.
            */
            CV_WRAP void estimateTransformation(InputArray src, InputArray dst, OutputArray _alphaPhiST, double &dist , bool fdContour = false);
            /** @brief set number of Fourier descriptors used in estimateTransformation

            @param n number of Fourier descriptors equal to number of contour points after resampling.
            */
            CV_WRAP void setCtrSize(int n);
            /** @brief set number of Fourier descriptors when estimateTransformation used vector<Point>

            @param n number of fourier descriptors used for optimal curve matching.
            */
            CV_WRAP void setFDSize(int n);
            /**
            @returns number of fourier descriptors
            */
            CV_WRAP int getCtrSize() { return ctrSize; };
            /**
            @returns number of fourier descriptors used for optimal curve matching
            */
            CV_WRAP int getFDSize() { return fdSize; };
        };
        /**
        * @brief   Fourier descriptors for planed closed curves
        *
        * For more details about this implementation, please see @cite PersoonFu1977
        *
        * @param   _src   contour type vector<Point> , vector<Point2f>  or vector<Point2d>
        * @param   _dst   Mat of type CV_64FC2 and nbElt rows A VERIFIER
        * @param   nbElt number of rows in _dst or getOptimalDFTSize rows if nbElt=-1
        * @param   nbFD number of FD return in _dst _dst = [FD(1...nbFD/2) FD(nbFD/2-nbElt+1...:nbElt)]
        *
        */
        CV_EXPORTS_W void fourierDescriptor(InputArray _src, OutputArray _dst, int nbElt=-1,int nbFD=-1);
        /**
        * @brief   transform a contour
        *
        * @param   _src   contour or Fourier Descriptors if fd is true
        * @param   _t   transform Mat given by estimateTransformation
        * @param   _dst   Mat of type CV_64FC2 and nbElt rows
        * @param   fdContour true _src are Fourier Descriptors. fdContour false _src is a contour
        *
        */
        CV_EXPORTS_W void transform(InputArray _src, InputArray _t,OutputArray _dst, bool fdContour=true);
        /**
        * @brief   Contour sampling .
        *
        * @param   _src   contour type vector<Point> , vector<Point2f>  or vector<Point2d>
        * @param   _out   Mat of type CV_64FC2 and nbElt rows
        * @param   nbElt number of points in _out contour
        *
        */
        CV_EXPORTS_W void contourSampling(InputArray _src, OutputArray _out, int nbElt);

        /**
        * @brief create

        * @param ctr number of Fourier descriptors equal to number of contour points after resampling.
        * @param fd Contour defining second shape (Target).
        */
        CV_EXPORTS_W Ptr<ContourFitting> create(int ctr = 1024, int fd = 16);

        //! @} ximgproc_shape
    }
}
#endif
