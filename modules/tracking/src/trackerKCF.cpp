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
#include <complex>

/*---------------------------
|  TrackerKCFModel
|---------------------------*/
namespace cv{
   /**
  * \brief Implementation of TrackerModel for MIL algorithm
  */
  class TrackerKCFModel : public TrackerModel{
  public:
    TrackerKCFModel(TrackerKCF::Params /*params*/){}
    ~TrackerKCFModel(){}
  protected:
    void modelEstimationImpl( const std::vector<Mat>& /*responses*/ ){}
    void modelUpdateImpl(){}
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
    void read( const FileNode& /*fn*/ );
    void write( FileStorage& /*fs*/ ) const;

  protected:
     /*
    * basic functions and vars
    */
    bool initImpl( const Mat& /*image*/, const Rect2d& boundingBox );
    bool updateImpl( const Mat& image, Rect2d& boundingBox );

    TrackerKCF::Params params;

    /*
    * KCF functions and vars
    */
    void createHanningWindow(OutputArray _dst, const cv::Size winSize, const int type) const;
    void inline fft2(const Mat src, std::vector<Mat> & dest) const;
    void inline fft2(const Mat src, Mat & dest) const;
    void inline ifft2(const Mat src, Mat & dest) const;
    void inline pixelWiseMult(const std::vector<Mat> src1, const std::vector<Mat>  src2, std::vector<Mat>  & dest, const int flags, const bool conjB=false) const;
    void inline sumChannels(std::vector<Mat> src, Mat & dest) const;
    void inline updateProjectionMatrix(const Mat src, Mat & old_cov,Mat &  _proj_mtx,double pca_rate, int compressed_sz) const;
    void inline compress(const Mat _proj_mtx, const Mat src, Mat & dest) const;
    bool getSubWindow(const Mat img, const Rect roi, Mat& patch) const;
    void extractCN(Mat _patch, Mat & cnFeatures) const;
    void denseGaussKernel(const double sigma, const Mat _x, const Mat _y, Mat & _k) const;
    void calcResponse(const Mat _alphaf, const Mat _k, Mat & _response) const;
    void calcResponse(const Mat _alphaf, const Mat _alphaf_den, const Mat _k, Mat & _response) const;

    void shiftRows(Mat& mat) const;
    void shiftRows(Mat& mat, int n) const;
    void shiftCols(Mat& mat, int n) const;

  private:
    double output_sigma;
    Rect2d roi;
    Mat hann; 	//hann window filter

    Mat y,yf; 	// training response and its FFT
    Mat x,xf; 	// observation and its FFT
    Mat k,kf;	// dense gaussian kernel and its FFT
    Mat kf_lambda; // kf+lambda
    Mat new_alphaf, alphaf;	// training coefficients
    Mat new_alphaf_den, alphaf_den; // for splitted training coefficients
    Mat z, new_z; // model
    Mat response; // detection result
    Mat old_cov_mtx, proj_mtx; // for feature compression

    bool resizeImage; // resize the image whenever needed and the patch size is large

    int frame;
  };

  /*
 * Constructor
 */
  Ptr<TrackerKCF> TrackerKCF::createTracker(const TrackerKCF::Params &parameters){
      return Ptr<TrackerKCFImpl>(new TrackerKCFImpl(parameters));
  }
  TrackerKCFImpl::TrackerKCFImpl( const TrackerKCF::Params &parameters ) :
      params( parameters )
  {
    isInit = false;
    resizeImage = false;

    CV_Assert(params.descriptor == GRAY || params.descriptor == CN /*|| params.descriptor == CN2*/);
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
  bool TrackerKCFImpl::initImpl( const Mat& /*image*/, const Rect2d& boundingBox ){
    frame=0;
    roi = boundingBox;

    //calclulate output sigma
    output_sigma=sqrt(roi.width*roi.height)*params.output_sigma_factor;
    output_sigma=-0.5/(output_sigma*output_sigma);

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
    createHanningWindow(hann, roi.size(), CV_64F);
    if(params.descriptor==CN){
      Mat layers[] = {hann, hann, hann, hann, hann, hann, hann, hann, hann, hann};
      merge(layers, 10, hann);
    }

    // create gaussian response
    y=Mat::zeros((int)roi.height,(int)roi.width,CV_64F);
    for(unsigned i=0;i<roi.height;i++){
      for(unsigned j=0;j<roi.width;j++){
        y.at<double>(i,j)=(i-roi.height/2+1)*(i-roi.height/2+1)+(j-roi.width/2+1)*(j-roi.width/2+1);
      }
    }

    y*=(double)output_sigma;
    cv::exp(y,y);

    // perform fourier transfor to the gaussian response
    fft2(y,yf);

    model=Ptr<TrackerKCFModel>(new TrackerKCFModel(params));

    // TODO: return true only if roi inside the image
    return true;
  }

  /*
   * Main part of the KCF algorithm
   */
  bool TrackerKCFImpl::updateImpl( const Mat& image, Rect2d& boundingBox ){
    double minVal, maxVal;	// min-max response
    Point minLoc,maxLoc;	// min-max location
    Mat zc;

    Mat img=image.clone();
    // check the channels of the input image, grayscale is preferred
    CV_Assert(image.channels() == 1 || image.channels() == 3);

    // resize the image whenever needed
    if(resizeImage)resize(img,img,Size(img.cols/2,img.rows/2));

    // extract and pre-process the patch
    if(!getSubWindow(img,roi, x))return false;

    // detection part
    if(frame>0){
      //compute the gaussian kernel
      if(params.compress_feature){
        compress(proj_mtx,x,x);
        compress(proj_mtx,z,zc);
        denseGaussKernel(params.sigma,x,zc,k);
      }else
        denseGaussKernel(params.sigma,x,z,k);

      // calculate filter response
      if(params.split_coeff)
        calcResponse(alphaf,alphaf_den,k,response);
      else
        calcResponse(alphaf,k,response);

      // extract the maximum response
      minMaxLoc( response, &minVal, &maxVal, &minLoc, &maxLoc );
      roi.x+=(maxLoc.x-roi.width/2+1);
      roi.y+=(maxLoc.y-roi.height/2+1);

      // update the bounding box
      boundingBox.x=(resizeImage?roi.x*2:roi.x)+boundingBox.width/2;
      boundingBox.y=(resizeImage?roi.y*2:roi.y)+boundingBox.height/2;
    }

    // extract the patch for learning purpose
    if(!getSubWindow(img,roi, x))return false;

    //update the training data
    new_z=x.clone();
    if(frame==0)
      z=x.clone();
    else
      z=(1.0-params.interp_factor)*z+params.interp_factor*new_z;

    if(params.compress_feature){
      // feature compression
      updateProjectionMatrix(z,old_cov_mtx,proj_mtx,params.pca_learning_rate,params.compressed_size);
      compress(proj_mtx,x,x);
    }

    // Kernel Regularized Least-Squares, calculate alphas
    denseGaussKernel(params.sigma,x,x,k);

    fft2(k,kf);
    kf_lambda=kf+params.lambda;

    /* TODO: optimize this element-wise division
     * new_alphaf=yf./kf
     * z=(a+bi)/(c+di)[(ac+bd)+i(bc-ad)]/(c^2+d^2)
     */
    new_alphaf=Mat_<Vec2d >(yf.rows, yf.cols);
    std::complex<double> temp;

    if(params.split_coeff){
      mulSpectrums(yf,kf,new_alphaf,0);
      mulSpectrums(kf,kf_lambda,new_alphaf_den,0);
    }else{
      for(int i=0;i<yf.rows;i++){
        for(int j=0;j<yf.cols;j++){
          temp=std::complex<double>(yf.at<Vec2d>(i,j)[0],yf.at<Vec2d>(i,j)[1])/(std::complex<double>(kf_lambda.at<Vec2d>(i,j)[0],kf_lambda.at<Vec2d>(i,j)[1])/*+std::complex<double>(0.0000000001,0.0000000001)*/);
          new_alphaf.at<Vec2d >(i,j)[0]=temp.real();
          new_alphaf.at<Vec2d >(i,j)[1]=temp.imag();
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
  void TrackerKCFImpl::createHanningWindow(OutputArray _dst, const cv::Size winSize, const int type) const {
      CV_Assert( type == CV_32FC1 || type == CV_64FC1 );

      _dst.create(winSize, type);
      Mat dst = _dst.getMat();

      int rows = dst.rows, cols = dst.cols;

      AutoBuffer<double> _wc(cols);
      double * const wc = (double *)_wc;

      double coeff0 = 2.0 * CV_PI / (double)(cols - 1), coeff1 = 2.0f * CV_PI / (double)(rows - 1);
      for(int j = 0; j < cols; j++)
        wc[j] = 0.5 * (1.0 - cos(coeff0 * j));

      if(dst.depth() == CV_32F){
        for(int i = 0; i < rows; i++){
          float* dstData = dst.ptr<float>(i);
          double wr = 0.5 * (1.0 - cos(coeff1 * i));
          for(int j = 0; j < cols; j++)
            dstData[j] = (float)(wr * wc[j]);
        }
      }else{
        for(int i = 0; i < rows; i++){
          double* dstData = dst.ptr<double>(i);
          double wr = 0.5 * (1.0 - cos(coeff1 * i));
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
    std::vector<Mat> layers(src.channels());
    std::vector<Mat> outputs(src.channels());

    split(src, layers);

    for(int i=0;i<src.channels();i++){
      dft(layers[i],outputs[i],DFT_COMPLEX_OUTPUT);
    }

    merge(outputs,dest);
  }

  void inline TrackerKCFImpl::fft2(const Mat src, std::vector<Mat> & dest) const {
    std::vector<Mat> layers(src.channels());
    dest.clear();
    dest.resize(src.channels());

    split(src, layers);

    for(int i=0;i<src.channels();i++){
      dft(layers[i],dest[i],DFT_COMPLEX_OUTPUT);
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
    dest.clear();
    dest.resize(src1.size());

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

  /*
   * obtains the projection matrix using PCA
   */
  void inline TrackerKCFImpl::updateProjectionMatrix(const Mat src, Mat & old_cov,Mat &  _proj_mtx, double pca_rate, int compressed_sz) const {
    CV_Assert(compressed_sz<=src.channels());

    // compute average
    std::vector<Mat> layers(src.channels());
    std::vector<Scalar> average(src.channels());
    split(src,layers);

    for (int i=0;i<src.channels();i++){
      average[i]=mean(layers[i]);
      layers[i]-=average[i];
    }

    // calc covariance matrix
    Mat data,new_cov;
    merge(layers,data);
    data=data.reshape(1,src.rows*src.cols);

    new_cov=1.0/(double)(src.rows*src.cols-1)*(data.t()*data);
    if(old_cov.rows==0)old_cov=new_cov.clone();

    // calc PCA
    Mat w, u, vt;
    SVD::compute((1.0-pca_rate)*old_cov+pca_rate*new_cov, w, u, vt);

    // extract the projection matrix
    _proj_mtx=u(Rect(0,0,compressed_sz,src.channels())).clone();
    Mat proj_vars=Mat::eye(compressed_sz,compressed_sz,_proj_mtx.type());
    for(int i=0;i<compressed_sz;i++){
      proj_vars.at<double>(i,i)=w.at<double>(i);
    }

    // update the covariance matrix
    old_cov=(1.0-pca_rate)*old_cov+pca_rate*_proj_mtx*proj_vars*_proj_mtx.t();
  }

  /*
   * compress the features
   */
  void inline TrackerKCFImpl::compress(const Mat _proj_mtx, const Mat src, Mat & dest) const {
    Mat data=src.reshape(1,src.rows*src.cols);
    Mat compressed=data*_proj_mtx;
    dest=compressed.reshape(_proj_mtx.cols,src.rows).clone();
  }

  /*
   * obtain the patch and apply hann window filter to it
   */
  bool TrackerKCFImpl::getSubWindow(const Mat img, const Rect _roi, Mat& patch) const {

    Rect region=_roi;

    // return false if roi is outside the image
    if((_roi.x+_roi.width<0)
      ||(_roi.y+_roi.height<0)
      ||(_roi.x>=img.cols)
      ||(_roi.y>=img.rows)
    )return false;

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
    switch(params.descriptor){
      case GRAY:
        if(img.channels()>1)cvtColor(patch,patch, CV_BGR2GRAY);
        patch.convertTo(patch,CV_64F);
        patch=patch/255.0-0.5; // normalize to range -0.5 .. 0.5
        break;
      case CN:
        CV_Assert(img.channels() == 3);
        extractCN(patch,patch);
        break;
      case CN2:
        if(patch.channels()>1)cvtColor(patch,patch, CV_BGR2GRAY);
        break;
    }

    patch=patch.mul(hann); // hann window filter

    return true;

  }

  /* Convert BGR to ColorNames
   */
  void TrackerKCFImpl::extractCN(Mat _patch, Mat & cnFeatures) const {
    Vec3b & pixel = _patch.at<Vec3b>(0,0);
    unsigned index;

    Mat temp = Mat::zeros(_patch.rows,_patch.cols,CV_64FC(10));

    for(int i=0;i<_patch.rows;i++){
      for(int j=0;j<_patch.cols;j++){
        pixel=_patch.at<Vec3b>(i,j);
        index=(unsigned)(floor(pixel[2]/8)+32*floor(pixel[1]/8)+32*32*floor(pixel[0]/8));

        //copy the values
        for(int _k=0;_k<10;_k++){
          temp.at<Vec<double,10> >(i,j)[_k]=ColorNames[index][_k];
        }
      }
    }

    cnFeatures=temp.clone();
  }

  /*
   *  dense gauss kernel function
   */
  void TrackerKCFImpl::denseGaussKernel(const double sigma, const Mat _x, const Mat _y, Mat & _k) const {
    std::vector<Mat> _xf,_yf,xyf_v;
    Mat xy,xyf;
    double normX, normY;

    fft2(_x,_xf);
    fft2(_y,_yf);

    normX=norm(_x);
    normX*=normX;
    normY=norm(_y);
    normY*=normY;

    pixelWiseMult(_xf,_yf,xyf_v,0,true);
    sumChannels(xyf_v,xyf);
    ifft2(xyf,xyf);

    if(params.wrap_kernel){
      shiftRows(xyf, _x.rows/2);
      shiftCols(xyf, _x.cols/2);
    }

    //(xx + yy - 2 * xy) / numel(x)
    xy=(normX+normY-2*xyf)/(_x.rows*_x.cols*_x.channels());

    // TODO: check wether we really need thresholding or not
    //threshold(xy,xy,0.0,0.0,THRESH_TOZERO);//max(0, (xx + yy - 2 * xy) / numel(x))
    for(int i=0;i<xy.rows;i++){
      for(int j=0;j<xy.cols;j++){
        if(xy.at<double>(i,j)<0.0)xy.at<double>(i,j)=0.0;
      }
    }

    double sig=-1.0/(sigma*sigma);
    xy=sig*xy;
    exp(xy,_k);

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
  void TrackerKCFImpl::calcResponse(const Mat _alphaf, const Mat _k, Mat & _response) const {
    //alpha f--> 2channels ; k --> 1 channel;
    Mat _kf;
    fft2(_k,_kf);
    Mat spec;
    mulSpectrums(_alphaf,_kf,spec,0,false);
    ifft2(spec,_response);
  }

  /*
   * calculate the detection response for splitted form
   */
  void TrackerKCFImpl::calcResponse(const Mat _alphaf, const Mat _alphaf_den, const Mat _k, Mat & _response) const {
    Mat _kf;
    fft2(_k,_kf);
    Mat spec;
    Mat spec2=Mat_<Vec2d >(_k.rows, _k.cols);
    std::complex<double> temp;

    mulSpectrums(_alphaf,_kf,spec,0,false);

    for(int i=0;i<_k.rows;i++){
      for(int j=0;j<_k.cols;j++){
        temp=std::complex<double>(spec.at<Vec2d>(i,j)[0],spec.at<Vec2d>(i,j)[1])/(std::complex<double>(_alphaf_den.at<Vec2d>(i,j)[0],_alphaf_den.at<Vec2d>(i,j)[1])/*+std::complex<double>(0.0000000001,0.0000000001)*/);
        spec2.at<Vec2d >(i,j)[0]=temp.real();
        spec2.at<Vec2d >(i,j)[1]=temp.imag();
      }
    }

    ifft2(spec2,_response);
  }
  /*----------------------------------------------------------------------*/

  /*
 * Parameters
 */
  TrackerKCF::Params::Params(){
      sigma=0.2;
      lambda=0.01;
      interp_factor=0.075;
      output_sigma_factor=1.0/16.0;
      resize=true;
      max_patch_size=80*80;
      descriptor=CN;
      split_coeff=true;
      wrap_kernel=false;

      //feature compression
      compress_feature=true;
      compressed_size=2;
      pca_learning_rate=0.15;
  }

  void TrackerKCF::Params::read( const cv::FileNode& /*fn*/ ){}

  void TrackerKCF::Params::write( cv::FileStorage& /*fs*/ ) const{}

} /* namespace cv */
