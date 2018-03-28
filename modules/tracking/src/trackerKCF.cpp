/*M///////////////////////////////////////////////////////////////////////////////////////
 //
 //  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
 //
 //  By downloading, copying, installing or using the software you agree to this license.
 //  If you do not agree to this license, do not download, install,
 //  copy or use the software.
 //
 //
 //                           License Agreement
 //                For Open Source Computer Vision Library
 //
 // Copyright (C) 2013, OpenCV Foundation, all rights reserved.
 // Third party copyrights are property of their respective owners.
 //
 // Redistribution and use in source and binary forms, with or without modification,
 // are permitted provided that the following conditions are met:
 //
 //   * Redistribution's of source code must retain the above copyright notice,
 //     this list of conditions and the following disclaimer.
 //
 //   * Redistribution's in binary form must reproduce the above copyright notice,
 //     this list of conditions and the following disclaimer in the documentation
 //     and/or other materials provided with the distribution.
 //
 //   * The name of the copyright holders may not be used to endorse or promote products
 //     derived from this software without specific prior written permission.
 //
 // This software is provided by the copyright holders and contributors "as is" and
 // any express or implied warranties, including, but not limited to, the implied
 // warranties of merchantability and fitness for a particular purpose are disclaimed.
 // In no event shall the Intel Corporation or contributors be liable for any direct,
 // indirect, incidental, special, exemplary, or consequential damages
 // (including, but not limited to, procurement of substitute goods or services;
 // loss of use, data, or profits; or business interruption) however caused
 // and on any theory of liability, whether in contract, strict liability,
 // or tort (including negligence or otherwise) arising in any way out of
 // the use of this software, even if advised of the possibility of such damage.
 //
 //M*/

#include "precomp.hpp"
#include "opencl_kernels_tracking.hpp"
#include <complex>
#include <cmath>

/*---------------------------
|  TrackerKCFModel
|---------------------------*/
namespace cv{
   /**
  * \brief Implementation of TrackerModel for KCF algorithm
  */
  class TrackerKCFModel : public TrackerModel{
  public:
    TrackerKCFModel(TrackerKCF::Params /*params*/){}
    ~TrackerKCFModel(){}
  protected:
    void modelEstimationImpl( const std::vector<Mat>& /*responses*/ ) CV_OVERRIDE {}
    void modelUpdateImpl() CV_OVERRIDE {}
  };
} /* namespace cv */


/*---------------------------
|  TrackerKCF
|---------------------------*/
namespace cv{

  /*
 * Prototype
 */
  class TrackerKCFImpl : public TrackerKCF {
  public:
    TrackerKCFImpl( const TrackerKCF::Params &parameters = TrackerKCF::Params() );
    void read( const FileNode& /*fn*/ ) CV_OVERRIDE;
    void write( FileStorage& /*fs*/ ) const CV_OVERRIDE;
    void setFeatureExtractor(void (*f)(const Mat, const Rect, Mat&), bool pca_func = false) CV_OVERRIDE;

  protected:
     /*
    * basic functions and vars
    */
    bool initImpl( const Mat& /*image*/, const Rect2d& boundingBox ) CV_OVERRIDE;
    bool updateImpl( const Mat& image, Rect2d& boundingBox ) CV_OVERRIDE;

    TrackerKCF::Params params;

    /*
    * KCF functions and vars
    */
    void createHanningWindow(OutputArray dest, const cv::Size winSize, const int type) const;
    void inline fft2(const Mat src, std::vector<Mat> & dest, std::vector<Mat> & layers_data) const;
    void inline fft2(const Mat src, Mat & dest) const;
    void inline ifft2(const Mat src, Mat & dest) const;
    void inline pixelWiseMult(const std::vector<Mat> src1, const std::vector<Mat>  src2, std::vector<Mat>  & dest, const int flags, const bool conjB=false) const;
    void inline sumChannels(std::vector<Mat> src, Mat & dest) const;
    void inline updateProjectionMatrix(const Mat src, Mat & old_cov,Mat &  proj_matrix,float pca_rate, int compressed_sz,
                                       std::vector<Mat> & layers_pca,std::vector<Scalar> & average, Mat pca_data, Mat new_cov, Mat w, Mat u, Mat v);
    void inline compress(const Mat proj_matrix, const Mat src, Mat & dest, Mat & data, Mat & compressed) const;
    bool getSubWindow(const Mat img, const Rect roi, Mat& feat, Mat& patch, TrackerKCF::MODE desc = GRAY) const;
    bool getSubWindow(const Mat img, const Rect roi, Mat& feat, void (*f)(const Mat, const Rect, Mat& )) const;
    void extractCN(Mat patch_data, Mat & cnFeatures) const;
    void denseGaussKernel(const float sigma, const Mat , const Mat y_data, Mat & k_data,
                          std::vector<Mat> & layers_data,std::vector<Mat> & xf_data,std::vector<Mat> & yf_data, std::vector<Mat> xyf_v, Mat xy, Mat xyf ) const;
    void calcResponse(const Mat alphaf_data, const Mat kf_data, Mat & response_data, Mat & spec_data) const;
    void calcResponse(const Mat alphaf_data, const Mat alphaf_den_data, const Mat kf_data, Mat & response_data, Mat & spec_data, Mat & spec2_data) const;

    void shiftRows(Mat& mat) const;
    void shiftRows(Mat& mat, int n) const;
    void shiftCols(Mat& mat, int n) const;
#ifdef HAVE_OPENCL
    bool inline oclTransposeMM(const Mat src, float alpha, UMat &dst);
#endif

  private:
    float output_sigma;
    Rect2d roi;
    Mat hann; 	//hann window filter
    Mat hann_cn; //10 dimensional hann-window filter for CN features,

    Mat y,yf; 	// training response and its FFT
    Mat x; 	// observation and its FFT
    Mat k,kf;	// dense gaussian kernel and its FFT
    Mat kf_lambda; // kf+lambda
    Mat new_alphaf, alphaf;	// training coefficients
    Mat new_alphaf_den, alphaf_den; // for splitted training coefficients
    Mat z; // model
    Mat response; // detection result
    Mat old_cov_mtx, proj_mtx; // for feature compression

    // pre-defined Mat variables for optimization of private functions
    Mat spec, spec2;
    std::vector<Mat> layers;
    std::vector<Mat> vxf,vyf,vxyf;
    Mat xy_data,xyf_data;
    Mat data_temp, compress_data;
    std::vector<Mat> layers_pca_data;
    std::vector<Scalar> average_data;
    Mat img_Patch;

    // storage for the extracted features, KRLS model, KRLS compressed model
    Mat X[2],Z[2],Zc[2];

    // storage of the extracted features
    std::vector<Mat> features_pca;
    std::vector<Mat> features_npca;
    std::vector<MODE> descriptors_pca;
    std::vector<MODE> descriptors_npca;

    // optimization variables for updateProjectionMatrix
    Mat data_pca, new_covar,w_data,u_data,vt_data;

    // custom feature extractor
    bool use_custom_extractor_pca;
    bool use_custom_extractor_npca;
    std::vector<void(*)(const Mat img, const Rect roi, Mat& output)> extractor_pca;
    std::vector<void(*)(const Mat img, const Rect roi, Mat& output)> extractor_npca;

    bool resizeImage; // resize the image whenever needed and the patch size is large

#ifdef HAVE_OPENCL
    ocl::Kernel transpose_mm_ker; // OCL kernel to compute transpose matrix multiply matrix.
#endif

    int frame;
  };

  /*
 * Constructor
 */
  Ptr<TrackerKCF> TrackerKCF::create(const TrackerKCF::Params &parameters){
      return Ptr<TrackerKCFImpl>(new TrackerKCFImpl(parameters));
  }
  Ptr<TrackerKCF> TrackerKCF::create(){
      return Ptr<TrackerKCFImpl>(new TrackerKCFImpl());
  }
  TrackerKCFImpl::TrackerKCFImpl( const TrackerKCF::Params &parameters ) :
      params( parameters )
  {
    isInit = false;
    resizeImage = false;
    use_custom_extractor_pca = false;
    use_custom_extractor_npca = false;

#ifdef HAVE_OPENCL
    // For update proj matrix's multiplication
    if(ocl::useOpenCL())
    {
        cv::String err;
        ocl::ProgramSource tmmSrc = ocl::tracking::tmm_oclsrc;
        ocl::Program tmmProg(tmmSrc, String(), err);
        transpose_mm_ker.create("tmm", tmmProg);
    }
#endif
  }

  void TrackerKCFImpl::read( const cv::FileNode& fn ){
    params.read( fn );
  }

  void TrackerKCFImpl::write( cv::FileStorage& fs ) const {
    params.write( fs );
  }

  /*
   * Initialization:
   * - creating hann window filter
   * - ROI padding
   * - creating a gaussian response for the training ground-truth
   * - perform FFT to the gaussian response
   */
  bool TrackerKCFImpl::initImpl( const Mat& image, const Rect2d& boundingBox ){
    frame=0;
    roi.x = cvRound(boundingBox.x);
    roi.y = cvRound(boundingBox.y);
    roi.width = cvRound(boundingBox.width);
    roi.height = cvRound(boundingBox.height);

    //calclulate output sigma
    output_sigma=std::sqrt(static_cast<float>(roi.width*roi.height))*params.output_sigma_factor;
    output_sigma=-0.5f/(output_sigma*output_sigma);

    //resize the ROI whenever needed
    if(params.resize && roi.width*roi.height>params.max_patch_size){
      resizeImage=true;
      roi.x/=2.0;
      roi.y/=2.0;
      roi.width/=2.0;
      roi.height/=2.0;
    }

    // add padding to the roi
    roi.x-=roi.width/2;
    roi.y-=roi.height/2;
    roi.width*=2;
    roi.height*=2;

    // initialize the hann window filter
    createHanningWindow(hann, roi.size(), CV_32F);

    // hann window filter for CN feature
    Mat _layer[] = {hann, hann, hann, hann, hann, hann, hann, hann, hann, hann};
    merge(_layer, 10, hann_cn);

    // create gaussian response
    y=Mat::zeros((int)roi.height,(int)roi.width,CV_32F);
    for(int i=0;i<int(roi.height);i++){
      for(int j=0;j<int(roi.width);j++){
        y.at<float>(i,j) =
                static_cast<float>((i-roi.height/2+1)*(i-roi.height/2+1)+(j-roi.width/2+1)*(j-roi.width/2+1));
      }
    }

    y*=(float)output_sigma;
    cv::exp(y,y);

    // perform fourier transfor to the gaussian response
    fft2(y,yf);

    if (image.channels() == 1) { // disable CN for grayscale images
      params.desc_pca &= ~(CN);
      params.desc_npca &= ~(CN);
    }
    model=Ptr<TrackerKCFModel>(new TrackerKCFModel(params));

    // record the non-compressed descriptors
    if((params.desc_npca & GRAY) == GRAY)descriptors_npca.push_back(GRAY);
    if((params.desc_npca & CN) == CN)descriptors_npca.push_back(CN);
    if(use_custom_extractor_npca)descriptors_npca.push_back(CUSTOM);
    features_npca.resize(descriptors_npca.size());

    // record the compressed descriptors
    if((params.desc_pca & GRAY) == GRAY)descriptors_pca.push_back(GRAY);
    if((params.desc_pca & CN) == CN)descriptors_pca.push_back(CN);
    if(use_custom_extractor_pca)descriptors_pca.push_back(CUSTOM);
    features_pca.resize(descriptors_pca.size());

    // accept only the available descriptor modes
    CV_Assert(
      (params.desc_pca & GRAY) == GRAY
      || (params.desc_npca & GRAY) == GRAY
      || (params.desc_pca & CN) == CN
      || (params.desc_npca & CN) == CN
      || use_custom_extractor_pca
      || use_custom_extractor_npca
    );

  //return true only if roi has intersection with the image
  if((roi & Rect2d(0,0, resizeImage ? image.cols / 2 : image.cols,
                   resizeImage ? image.rows / 2 : image.rows)) == Rect2d())
      return false;

    return true;
  }

  /*
   * Main part of the KCF algorithm
   */
  bool TrackerKCFImpl::updateImpl( const Mat& image, Rect2d& boundingBox ){
    double minVal, maxVal;	// min-max response
    Point minLoc,maxLoc;	// min-max location

    Mat img=image.clone();
    // check the channels of the input image, grayscale is preferred
    CV_Assert(img.channels() == 1 || img.channels() == 3);

    // resize the image whenever needed
    if(resizeImage)resize(img,img,Size(img.cols/2,img.rows/2),0,0,INTER_LINEAR_EXACT);

    // detection part
    if(frame>0){

      // extract and pre-process the patch
      // get non compressed descriptors
      for(unsigned i=0;i<descriptors_npca.size()-extractor_npca.size();i++){
        if(!getSubWindow(img,roi, features_npca[i], img_Patch, descriptors_npca[i]))return false;
      }
      //get non-compressed custom descriptors
      for(unsigned i=0,j=(unsigned)(descriptors_npca.size()-extractor_npca.size());i<extractor_npca.size();i++,j++){
        if(!getSubWindow(img,roi, features_npca[j], extractor_npca[i]))return false;
      }
      if(features_npca.size()>0)merge(features_npca,X[1]);

      // get compressed descriptors
      for(unsigned i=0;i<descriptors_pca.size()-extractor_pca.size();i++){
        if(!getSubWindow(img,roi, features_pca[i], img_Patch, descriptors_pca[i]))return false;
      }
      //get compressed custom descriptors
      for(unsigned i=0,j=(unsigned)(descriptors_pca.size()-extractor_pca.size());i<extractor_pca.size();i++,j++){
        if(!getSubWindow(img,roi, features_pca[j], extractor_pca[i]))return false;
      }
      if(features_pca.size()>0)merge(features_pca,X[0]);

      //compress the features and the KRSL model
      if(params.desc_pca !=0){
        compress(proj_mtx,X[0],X[0],data_temp,compress_data);
        compress(proj_mtx,Z[0],Zc[0],data_temp,compress_data);
      }

      // copy the compressed KRLS model
      Zc[1] = Z[1];

      // merge all features
      if(features_npca.size()==0){
        x = X[0];
        z = Zc[0];
      }else if(features_pca.size()==0){
        x = X[1];
        z = Z[1];
      }else{
        merge(X,2,x);
        merge(Zc,2,z);
      }

      //compute the gaussian kernel
      denseGaussKernel(params.sigma,x,z,k,layers,vxf,vyf,vxyf,xy_data,xyf_data);

      // compute the fourier transform of the kernel
      fft2(k,kf);
      if(frame==1)spec2=Mat_<Vec2f >(kf.rows, kf.cols);

      // calculate filter response
      if(params.split_coeff)
        calcResponse(alphaf,alphaf_den,kf,response, spec, spec2);
      else
        calcResponse(alphaf,kf,response, spec);

      // extract the maximum response
      minMaxLoc( response, &minVal, &maxVal, &minLoc, &maxLoc );
      if (maxVal < params.detect_thresh)
      {
          return false;
      }
      roi.x+=(maxLoc.x-roi.width/2+1);
      roi.y+=(maxLoc.y-roi.height/2+1);
    }

    // update the bounding box
    boundingBox.x=(resizeImage?roi.x*2:roi.x)+(resizeImage?roi.width*2:roi.width)/4;
    boundingBox.y=(resizeImage?roi.y*2:roi.y)+(resizeImage?roi.height*2:roi.height)/4;
    boundingBox.width = (resizeImage?roi.width*2:roi.width)/2;
    boundingBox.height = (resizeImage?roi.height*2:roi.height)/2;

    // extract the patch for learning purpose
    // get non compressed descriptors
    for(unsigned i=0;i<descriptors_npca.size()-extractor_npca.size();i++){
      if(!getSubWindow(img,roi, features_npca[i], img_Patch, descriptors_npca[i]))return false;
    }
    //get non-compressed custom descriptors
    for(unsigned i=0,j=(unsigned)(descriptors_npca.size()-extractor_npca.size());i<extractor_npca.size();i++,j++){
      if(!getSubWindow(img,roi, features_npca[j], extractor_npca[i]))return false;
    }
    if(features_npca.size()>0)merge(features_npca,X[1]);

    // get compressed descriptors
    for(unsigned i=0;i<descriptors_pca.size()-extractor_pca.size();i++){
      if(!getSubWindow(img,roi, features_pca[i], img_Patch, descriptors_pca[i]))return false;
    }
    //get compressed custom descriptors
    for(unsigned i=0,j=(unsigned)(descriptors_pca.size()-extractor_pca.size());i<extractor_pca.size();i++,j++){
      if(!getSubWindow(img,roi, features_pca[j], extractor_pca[i]))return false;
    }
    if(features_pca.size()>0)merge(features_pca,X[0]);

    //update the training data
    if(frame==0){
      Z[0] = X[0].clone();
      Z[1] = X[1].clone();
    }else{
      Z[0]=(1.0-params.interp_factor)*Z[0]+params.interp_factor*X[0];
      Z[1]=(1.0-params.interp_factor)*Z[1]+params.interp_factor*X[1];
    }

    if(params.desc_pca !=0 || use_custom_extractor_pca){
      // initialize the vector of Mat variables
      if(frame==0){
        layers_pca_data.resize(Z[0].channels());
        average_data.resize(Z[0].channels());
      }

      // feature compression
      updateProjectionMatrix(Z[0],old_cov_mtx,proj_mtx,params.pca_learning_rate,params.compressed_size,layers_pca_data,average_data,data_pca, new_covar,w_data,u_data,vt_data);
      compress(proj_mtx,X[0],X[0],data_temp,compress_data);
    }

    // merge all features
    if(features_npca.size()==0)
      x = X[0];
    else if(features_pca.size()==0)
      x = X[1];
    else
      merge(X,2,x);

    // initialize some required Mat variables
    if(frame==0){
      layers.resize(x.channels());
      vxf.resize(x.channels());
      vyf.resize(x.channels());
      vxyf.resize(vyf.size());
      new_alphaf=Mat_<Vec2f >(yf.rows, yf.cols);
    }

    // Kernel Regularized Least-Squares, calculate alphas
    denseGaussKernel(params.sigma,x,x,k,layers,vxf,vyf,vxyf,xy_data,xyf_data);

    // compute the fourier transform of the kernel and add a small value
    fft2(k,kf);
    kf_lambda=kf+params.lambda;

    float den;
    if(params.split_coeff){
      mulSpectrums(yf,kf,new_alphaf,0);
      mulSpectrums(kf,kf_lambda,new_alphaf_den,0);
    }else{
      for(int i=0;i<yf.rows;i++){
        for(int j=0;j<yf.cols;j++){
          den = 1.0f/(kf_lambda.at<Vec2f>(i,j)[0]*kf_lambda.at<Vec2f>(i,j)[0]+kf_lambda.at<Vec2f>(i,j)[1]*kf_lambda.at<Vec2f>(i,j)[1]);

          new_alphaf.at<Vec2f>(i,j)[0]=
          (yf.at<Vec2f>(i,j)[0]*kf_lambda.at<Vec2f>(i,j)[0]+yf.at<Vec2f>(i,j)[1]*kf_lambda.at<Vec2f>(i,j)[1])*den;
          new_alphaf.at<Vec2f>(i,j)[1]=
          (yf.at<Vec2f>(i,j)[1]*kf_lambda.at<Vec2f>(i,j)[0]-yf.at<Vec2f>(i,j)[0]*kf_lambda.at<Vec2f>(i,j)[1])*den;
        }
      }
    }

    // update the RLS model
    if(frame==0){
      alphaf=new_alphaf.clone();
      if(params.split_coeff)alphaf_den=new_alphaf_den.clone();
    }else{
      alphaf=(1.0-params.interp_factor)*alphaf+params.interp_factor*new_alphaf;
      if(params.split_coeff)alphaf_den=(1.0-params.interp_factor)*alphaf_den+params.interp_factor*new_alphaf_den;
    }

    frame++;
    return true;
  }


  /*-------------------------------------
  |  implementation of the KCF functions
  |-------------------------------------*/

  /*
   * hann window filter
   */
  void TrackerKCFImpl::createHanningWindow(OutputArray dest, const cv::Size winSize, const int type) const {
      CV_Assert( type == CV_32FC1 || type == CV_64FC1 );

      dest.create(winSize, type);
      Mat dst = dest.getMat();

      int rows = dst.rows, cols = dst.cols;

      AutoBuffer<float> _wc(cols);
      float * const wc = (float *)_wc;

      const float coeff0 = 2.0f * (float)CV_PI / (cols - 1);
      const float coeff1 = 2.0f * (float)CV_PI / (rows - 1);
      for(int j = 0; j < cols; j++)
        wc[j] = 0.5f * (1.0f - cos(coeff0 * j));

      if(dst.depth() == CV_32F){
        for(int i = 0; i < rows; i++){
          float* dstData = dst.ptr<float>(i);
          float wr = 0.5f * (1.0f - cos(coeff1 * i));
          for(int j = 0; j < cols; j++)
            dstData[j] = (float)(wr * wc[j]);
        }
      }else{
        for(int i = 0; i < rows; i++){
          double* dstData = dst.ptr<double>(i);
          double wr = 0.5f * (1.0f - cos(coeff1 * i));
          for(int j = 0; j < cols; j++)
            dstData[j] = wr * wc[j];
        }
      }

      // perform batch sqrt for SSE performance gains
      //cv::sqrt(dst, dst); //matlab do not use the square rooted version
  }

  /*
   * simplification of fourier transform function in opencv
   */
  void inline TrackerKCFImpl::fft2(const Mat src, Mat & dest) const {
    dft(src,dest,DFT_COMPLEX_OUTPUT);
  }

  void inline TrackerKCFImpl::fft2(const Mat src, std::vector<Mat> & dest, std::vector<Mat> & layers_data) const {
    split(src, layers_data);

    for(int i=0;i<src.channels();i++){
      dft(layers_data[i],dest[i],DFT_COMPLEX_OUTPUT);
    }
  }

  /*
   * simplification of inverse fourier transform function in opencv
   */
  void inline TrackerKCFImpl::ifft2(const Mat src, Mat & dest) const {
    idft(src,dest,DFT_SCALE+DFT_REAL_OUTPUT);
  }

  /*
   * Point-wise multiplication of two Multichannel Mat data
   */
  void inline TrackerKCFImpl::pixelWiseMult(const std::vector<Mat> src1, const std::vector<Mat>  src2, std::vector<Mat>  & dest, const int flags, const bool conjB) const {
    for(unsigned i=0;i<src1.size();i++){
      mulSpectrums(src1[i], src2[i], dest[i],flags,conjB);
    }
  }

  /*
   * Combines all channels in a multi-channels Mat data into a single channel
   */
  void inline TrackerKCFImpl::sumChannels(std::vector<Mat> src, Mat & dest) const {
    dest=src[0].clone();
    for(unsigned i=1;i<src.size();i++){
      dest+=src[i];
    }
  }

#ifdef HAVE_OPENCL
  bool inline TrackerKCFImpl::oclTransposeMM(const Mat src, float alpha, UMat &dst){
    // Current kernel only support matrix's rows is multiple of 4.
    // And if one line is less than 512KB, CPU will likely be faster.
    if (transpose_mm_ker.empty() ||
        src.rows % 4 != 0 ||
        (src.rows * 10) < (1024 * 1024 / 4))
      return false;

    Size s(src.rows, src.cols);
    const Mat tmp = src.t();
    const UMat uSrc = tmp.getUMat(ACCESS_READ);
    transpose_mm_ker.args(
        ocl::KernelArg::PtrReadOnly(uSrc),
        (int)uSrc.rows,
        (int)uSrc.cols,
        alpha,
        ocl::KernelArg::PtrWriteOnly(dst));
    size_t globSize[2] = {static_cast<size_t>(src.cols * 64), static_cast<size_t>(src.cols)};
    size_t localSize[2] = {64, 1};
    if (!transpose_mm_ker.run(2, globSize, localSize, true))
      return false;
    return true;
  }
#endif

  /*
   * obtains the projection matrix using PCA
   */
  void inline TrackerKCFImpl::updateProjectionMatrix(const Mat src, Mat & old_cov,Mat &  proj_matrix, float pca_rate, int compressed_sz,
                                                     std::vector<Mat> & layers_pca,std::vector<Scalar> & average, Mat pca_data, Mat new_cov, Mat w, Mat u, Mat vt) {
    CV_Assert(compressed_sz<=src.channels());

    split(src,layers_pca);

    for (int i=0;i<src.channels();i++){
      average[i]=mean(layers_pca[i]);
      layers_pca[i]-=average[i];
    }

    // calc covariance matrix
    merge(layers_pca,pca_data);
    pca_data=pca_data.reshape(1,src.rows*src.cols);

#ifdef HAVE_OPENCL
    bool oclSucceed = false;
    Size s(pca_data.cols, pca_data.cols);
    UMat result(s, pca_data.type());
    if (oclTransposeMM(pca_data, 1.0f/(float)(src.rows*src.cols-1), result)) {
      if(old_cov.rows==0) old_cov=result.getMat(ACCESS_READ).clone();
      SVD::compute((1.0-pca_rate)*old_cov + pca_rate * result.getMat(ACCESS_READ), w, u, vt);
      oclSucceed = true;
    }
#define TMM_VERIFICATION 0

    if (oclSucceed == false || TMM_VERIFICATION) {
      new_cov=1.0f/(float)(src.rows*src.cols-1)*(pca_data.t()*pca_data);
#if TMM_VERIFICATION
      for(int i = 0; i < new_cov.rows; i++)
        for(int j = 0; j < new_cov.cols; j++)
          if (abs(new_cov.at<float>(i, j) - result.getMat(ACCESS_RW).at<float>(i , j)) > abs(new_cov.at<float>(i, j)) * 1e-3)
            printf("error @ i %d j %d got %G expected %G \n", i, j, result.getMat(ACCESS_RW).at<float>(i , j), new_cov.at<float>(i, j));
#endif
      if(old_cov.rows==0)old_cov=new_cov.clone();
      SVD::compute((1.0f - pca_rate) * old_cov + pca_rate * new_cov, w, u, vt);
    }
#else
    new_cov=1.0/(float)(src.rows*src.cols-1)*(pca_data.t()*pca_data);
    if(old_cov.rows==0)old_cov=new_cov.clone();

    // calc PCA
    SVD::compute((1.0-pca_rate)*old_cov+pca_rate*new_cov, w, u, vt);
#endif
    // extract the projection matrix
    proj_matrix=u(Rect(0,0,compressed_sz,src.channels())).clone();
    Mat proj_vars=Mat::eye(compressed_sz,compressed_sz,proj_matrix.type());
    for(int i=0;i<compressed_sz;i++){
      proj_vars.at<float>(i,i)=w.at<float>(i);
    }

    // update the covariance matrix
    old_cov=(1.0-pca_rate)*old_cov+pca_rate*proj_matrix*proj_vars*proj_matrix.t();
  }

  /*
   * compress the features
   */
  void inline TrackerKCFImpl::compress(const Mat proj_matrix, const Mat src, Mat & dest, Mat & data, Mat & compressed) const {
    data=src.reshape(1,src.rows*src.cols);
    compressed=data*proj_matrix;
    dest=compressed.reshape(proj_matrix.cols,src.rows).clone();
  }

  /*
   * obtain the patch and apply hann window filter to it
   */
  bool TrackerKCFImpl::getSubWindow(const Mat img, const Rect _roi, Mat& feat, Mat& patch, TrackerKCF::MODE desc) const {

    Rect region=_roi;

    // return false if roi is outside the image
    if((roi & Rect2d(0,0, img.cols, img.rows)) == Rect2d() )
        return false;

    // extract patch inside the image
    if(_roi.x<0){region.x=0;region.width+=_roi.x;}
    if(_roi.y<0){region.y=0;region.height+=_roi.y;}
    if(_roi.x+_roi.width>img.cols)region.width=img.cols-_roi.x;
    if(_roi.y+_roi.height>img.rows)region.height=img.rows-_roi.y;
    if(region.width>img.cols)region.width=img.cols;
    if(region.height>img.rows)region.height=img.rows;

    patch=img(region).clone();

    // add some padding to compensate when the patch is outside image border
    int addTop,addBottom, addLeft, addRight;
    addTop=region.y-_roi.y;
    addBottom=(_roi.height+_roi.y>img.rows?_roi.height+_roi.y-img.rows:0);
    addLeft=region.x-_roi.x;
    addRight=(_roi.width+_roi.x>img.cols?_roi.width+_roi.x-img.cols:0);

    copyMakeBorder(patch,patch,addTop,addBottom,addLeft,addRight,BORDER_REPLICATE);
    if(patch.rows==0 || patch.cols==0)return false;

    // extract the desired descriptors
    switch(desc){
      case CN:
        CV_Assert(img.channels() == 3);
        extractCN(patch,feat);
        feat=feat.mul(hann_cn); // hann window filter
        break;
      default: // GRAY
        if(img.channels()>1)
          cvtColor(patch,feat, CV_BGR2GRAY);
        else
          feat=patch;
        //feat.convertTo(feat,CV_32F);
        feat.convertTo(feat,CV_32F, 1.0/255.0, -0.5);
        //feat=feat/255.0-0.5; // normalize to range -0.5 .. 0.5
        feat=feat.mul(hann); // hann window filter
        break;
    }

    return true;

  }

  /*
   * get feature using external function
   */
  bool TrackerKCFImpl::getSubWindow(const Mat img, const Rect _roi, Mat& feat, void (*f)(const Mat, const Rect, Mat& )) const{

    // return false if roi is outside the image
    if((_roi.x+_roi.width<0)
      ||(_roi.y+_roi.height<0)
      ||(_roi.x>=img.cols)
      ||(_roi.y>=img.rows)
    )return false;

    f(img, _roi, feat);

    if(_roi.width != feat.cols || _roi.height != feat.rows){
      printf("error in customized function of features extractor!\n");
      printf("Rules: roi.width==feat.cols && roi.height = feat.rows \n");
    }

    Mat hann_win;
    std::vector<Mat> _layers;

    for(int i=0;i<feat.channels();i++)
      _layers.push_back(hann);

    merge(_layers, hann_win);

    feat=feat.mul(hann_win); // hann window filter

    return true;
  }

  /* Convert BGR to ColorNames
   */
  void TrackerKCFImpl::extractCN(Mat patch_data, Mat & cnFeatures) const {
    Vec3b & pixel = patch_data.at<Vec3b>(0,0);
    unsigned index;

    if(cnFeatures.type() != CV_32FC(10))
      cnFeatures = Mat::zeros(patch_data.rows,patch_data.cols,CV_32FC(10));

    for(int i=0;i<patch_data.rows;i++){
      for(int j=0;j<patch_data.cols;j++){
        pixel=patch_data.at<Vec3b>(i,j);
        index=(unsigned)(floor((float)pixel[2]/8)+32*floor((float)pixel[1]/8)+32*32*floor((float)pixel[0]/8));

        //copy the values
        for(int _k=0;_k<10;_k++){
          cnFeatures.at<Vec<float,10> >(i,j)[_k]=ColorNames[index][_k];
        }
      }
    }

  }

  /*
   *  dense gauss kernel function
   */
  void TrackerKCFImpl::denseGaussKernel(const float sigma, const Mat x_data, const Mat y_data, Mat & k_data,
                                        std::vector<Mat> & layers_data,std::vector<Mat> & xf_data,std::vector<Mat> & yf_data, std::vector<Mat> xyf_v, Mat xy, Mat xyf ) const {
    double normX, normY;

    fft2(x_data,xf_data,layers_data);
    fft2(y_data,yf_data,layers_data);

    normX=norm(x_data);
    normX*=normX;
    normY=norm(y_data);
    normY*=normY;

    pixelWiseMult(xf_data,yf_data,xyf_v,0,true);
    sumChannels(xyf_v,xyf);
    ifft2(xyf,xyf);

    if(params.wrap_kernel){
      shiftRows(xyf, x_data.rows/2);
      shiftCols(xyf, x_data.cols/2);
    }

    //(xx + yy - 2 * xy) / numel(x)
    xy=(normX+normY-2*xyf)/(x_data.rows*x_data.cols*x_data.channels());

    // TODO: check wether we really need thresholding or not
    //threshold(xy,xy,0.0,0.0,THRESH_TOZERO);//max(0, (xx + yy - 2 * xy) / numel(x))
    for(int i=0;i<xy.rows;i++){
      for(int j=0;j<xy.cols;j++){
        if(xy.at<float>(i,j)<0.0)xy.at<float>(i,j)=0.0;
      }
    }

    float sig=-1.0f/(sigma*sigma);
    xy=sig*xy;
    exp(xy,k_data);

  }

  /* CIRCULAR SHIFT Function
   * http://stackoverflow.com/questions/10420454/shift-like-matlab-function-rows-or-columns-of-a-matrix-in-opencv
   */
  // circular shift one row from up to down
  void TrackerKCFImpl::shiftRows(Mat& mat) const {

      Mat temp;
      Mat m;
      int _k = (mat.rows-1);
      mat.row(_k).copyTo(temp);
      for(; _k > 0 ; _k-- ) {
        m = mat.row(_k);
        mat.row(_k-1).copyTo(m);
      }
      m = mat.row(0);
      temp.copyTo(m);

  }

  // circular shift n rows from up to down if n > 0, -n rows from down to up if n < 0
  void TrackerKCFImpl::shiftRows(Mat& mat, int n) const {
      if( n < 0 ) {
        n = -n;
        flip(mat,mat,0);
        for(int _k=0; _k < n;_k++) {
          shiftRows(mat);
        }
        flip(mat,mat,0);
      }else{
        for(int _k=0; _k < n;_k++) {
          shiftRows(mat);
        }
      }
  }

  //circular shift n columns from left to right if n > 0, -n columns from right to left if n < 0
  void TrackerKCFImpl::shiftCols(Mat& mat, int n) const {
      if(n < 0){
        n = -n;
        flip(mat,mat,1);
        transpose(mat,mat);
        shiftRows(mat,n);
        transpose(mat,mat);
        flip(mat,mat,1);
      }else{
        transpose(mat,mat);
        shiftRows(mat,n);
        transpose(mat,mat);
      }
  }

  /*
   * calculate the detection response
   */
  void TrackerKCFImpl::calcResponse(const Mat alphaf_data, const Mat kf_data, Mat & response_data, Mat & spec_data) const {
    //alpha f--> 2channels ; k --> 1 channel;
    mulSpectrums(alphaf_data,kf_data,spec_data,0,false);
    ifft2(spec_data,response_data);
  }

  /*
   * calculate the detection response for splitted form
   */
  void TrackerKCFImpl::calcResponse(const Mat alphaf_data, const Mat _alphaf_den, const Mat kf_data, Mat & response_data, Mat & spec_data, Mat & spec2_data) const {

    mulSpectrums(alphaf_data,kf_data,spec_data,0,false);

    //z=(a+bi)/(c+di)=[(ac+bd)+i(bc-ad)]/(c^2+d^2)
    float den;
    for(int i=0;i<kf_data.rows;i++){
      for(int j=0;j<kf_data.cols;j++){
        den=1.0f/(_alphaf_den.at<Vec2f>(i,j)[0]*_alphaf_den.at<Vec2f>(i,j)[0]+_alphaf_den.at<Vec2f>(i,j)[1]*_alphaf_den.at<Vec2f>(i,j)[1]);
        spec2_data.at<Vec2f>(i,j)[0]=
          (spec_data.at<Vec2f>(i,j)[0]*_alphaf_den.at<Vec2f>(i,j)[0]+spec_data.at<Vec2f>(i,j)[1]*_alphaf_den.at<Vec2f>(i,j)[1])*den;
        spec2_data.at<Vec2f>(i,j)[1]=
          (spec_data.at<Vec2f>(i,j)[1]*_alphaf_den.at<Vec2f>(i,j)[0]-spec_data.at<Vec2f>(i,j)[0]*_alphaf_den.at<Vec2f>(i,j)[1])*den;
      }
    }

    ifft2(spec2_data,response_data);
  }

  void TrackerKCFImpl::setFeatureExtractor(void (*f)(const Mat, const Rect, Mat&), bool pca_func){
    if(pca_func){
      extractor_pca.push_back(f);
      use_custom_extractor_pca = true;
    }else{
      extractor_npca.push_back(f);
      use_custom_extractor_npca = true;
    }
  }
  /*----------------------------------------------------------------------*/

  /*
 * Parameters
 */
  TrackerKCF::Params::Params(){
      detect_thresh = 0.5f;
      sigma=0.2f;
      lambda=0.0001f;
      interp_factor=0.075f;
      output_sigma_factor=1.0f / 16.0f;
      resize=true;
      max_patch_size=80*80;
      split_coeff=true;
      wrap_kernel=false;
      desc_npca = GRAY;
      desc_pca = CN;

      //feature compression
      compress_feature=true;
      compressed_size=2;
      pca_learning_rate=0.15f;
  }

  void TrackerKCF::Params::read( const cv::FileNode& fn ){
      *this = TrackerKCF::Params();

      if (!fn["detect_thresh"].empty())
          fn["detect_thresh"] >> detect_thresh;

      if (!fn["sigma"].empty())
          fn["sigma"] >> sigma;

      if (!fn["lambda"].empty())
          fn["lambda"] >> lambda;

      if (!fn["interp_factor"].empty())
          fn["interp_factor"] >> interp_factor;

      if (!fn["output_sigma_factor"].empty())
          fn["output_sigma_factor"] >> output_sigma_factor;

      if (!fn["resize"].empty())
          fn["resize"] >> resize;

      if (!fn["max_patch_size"].empty())
          fn["max_patch_size"] >> max_patch_size;

      if (!fn["split_coeff"].empty())
          fn["split_coeff"] >> split_coeff;

      if (!fn["wrap_kernel"].empty())
          fn["wrap_kernel"] >> wrap_kernel;


      if (!fn["desc_npca"].empty())
          fn["desc_npca"] >> desc_npca;

      if (!fn["desc_pca"].empty())
          fn["desc_pca"] >> desc_pca;

      if (!fn["compress_feature"].empty())
          fn["compress_feature"] >> compress_feature;

      if (!fn["compressed_size"].empty())
          fn["compressed_size"] >> compressed_size;

      if (!fn["pca_learning_rate"].empty())
          fn["pca_learning_rate"] >> pca_learning_rate;
  }

  void TrackerKCF::Params::write( cv::FileStorage& fs ) const{
    fs << "detect_thresh" << detect_thresh;
    fs << "sigma" << sigma;
    fs << "lambda" << lambda;
    fs << "interp_factor" << interp_factor;
    fs << "output_sigma_factor" << output_sigma_factor;
    fs << "resize" << resize;
    fs << "max_patch_size" << max_patch_size;
    fs << "split_coeff" << split_coeff;
    fs << "wrap_kernel" << wrap_kernel;
    fs << "desc_npca" << desc_npca;
    fs << "desc_pca" << desc_pca;
    fs << "compress_feature" << compress_feature;
    fs << "compressed_size" << compressed_size;
    fs << "pca_learning_rate" << pca_learning_rate;
  }
} /* namespace cv */
