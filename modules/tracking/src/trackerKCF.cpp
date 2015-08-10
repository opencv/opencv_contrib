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
    void inline fft2(const Mat src, std::vector<Mat> & _dest, std::vector<Mat> & _layers) const;
    void inline fft2(const Mat src, Mat & _dest) const;
    void inline ifft2(const Mat src, Mat & dest) const;
    void inline pixelWiseMult(const std::vector<Mat> src1, const std::vector<Mat>  src2, std::vector<Mat>  & dest, const int flags, const bool conjB=false) const;
    void inline sumChannels(std::vector<Mat> src, Mat & dest) const;
    void inline updateProjectionMatrix(const Mat src, Mat & old_cov,Mat &  _proj_mtx,double pca_rate, int compressed_sz,
                                       std::vector<Mat> & layers_pca,std::vector<Scalar> & average, Mat _data_pca, Mat _new_cov, Mat w, Mat u, Mat v) const;
    void inline compress(const Mat _proj_mtx, const Mat src, Mat & dest, Mat & data, Mat & compressed) const;
    bool getSubWindow(const Mat img, const Rect roi, Mat& feat, Mat& patch, TrackerKCF::MODE desc = GRAY) const;
    void extractCN(Mat _patch, Mat & cnFeatures) const;
    void denseGaussKernel(const double sigma, const Mat _x, const Mat _y, Mat & _k,
                          std::vector<Mat> & _layers,std::vector<Mat> & _xf,std::vector<Mat> & _yf, std::vector<Mat> xyf_v, Mat xy, Mat xyf ) const;
    void calcResponse(const Mat _alphaf, const Mat _kf, Mat & _response, Mat & _spec) const;
    void calcResponse(const Mat _alphaf, const Mat _alphaf_den, const Mat _kf, Mat & _response, Mat & _spec, Mat & _spec2) const;

    void shiftRows(Mat& mat) const;
    void shiftRows(Mat& mat, int n) const;
    void shiftCols(Mat& mat, int n) const;

  private:
    double output_sigma;
    Rect2d roi;
    Mat hann; 	//hann window filter
    Mat hann_cn;

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
    Mat _xy,_xyf;
    Mat _data, _compress;
    std::vector<Mat> _layers_pca;
    std::vector<Scalar> _average;
    Mat img_Patch;

    // data type for the extracted features
    struct _features{
      Mat vec[2];
      Mat & pca;
      Mat & npca;
      _features():pca(vec[0]),npca(vec[1]){};
    };

    // storage for the extracted features, KRLS model, KRLS compressed model
    _features X,Z,Zc;

    // storage of the extracted features
    std::vector<Mat> features_pca;
    std::vector<Mat> features_npca;
    std::vector<MODE> descriptors_pca;
    std::vector<MODE> descriptors_npca;

    // optimization variables for updateProjectionMatrix
    Mat data_pca, new_covar,_w,_u,_vt;

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

    // accept only the available descriptor modes
    CV_Assert(
      (params.desc_pca & GRAY) == GRAY
      || (params.desc_npca & GRAY) == GRAY
      || (params.desc_pca & CN) == CN
      || (params.desc_npca & CN) == CN
    );
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

    // hann window filter for CN feature
    Mat _layer[] = {hann, hann, hann, hann, hann, hann, hann, hann, hann, hann};
    merge(_layer, 10, hann_cn);

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

    // record the non-compressed descriptors
    if((params.desc_npca & GRAY) == GRAY)descriptors_npca.push_back(GRAY);
    if((params.desc_npca & CN) == CN)descriptors_npca.push_back(CN);
    features_npca.resize(descriptors_npca.size());

    // record the compressed descriptors
    if((params.desc_pca & GRAY) == GRAY)descriptors_pca.push_back(GRAY);
    if((params.desc_pca & CN) == CN)descriptors_pca.push_back(CN);
    features_pca.resize(descriptors_pca.size());

    // TODO: return true only if roi inside the image
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
    if(resizeImage)resize(img,img,Size(img.cols/2,img.rows/2));

    // detection part
    if(frame>0){

      // extract and pre-process the patch
      // get non compressed descriptors
      for(unsigned i=0;i<descriptors_npca.size();i++){
        if(!getSubWindow(img,roi, features_npca[i], img_Patch, descriptors_npca[i]))return false;
      }
      if(features_npca.size()>0)merge(features_npca,X.npca);

      // get compressed descriptors
      for(unsigned i=0;i<descriptors_pca.size();i++){
        if(!getSubWindow(img,roi, features_pca[i], img_Patch, descriptors_pca[i]))return false;
      }
      if(features_pca.size()>0)merge(features_pca,X.pca);

      //compress the features and the KRSL model
      if(params.desc_pca !=0){
        compress(proj_mtx,X.pca,X.pca,_data,_compress);
        compress(proj_mtx,Z.pca,Zc.pca,_data,_compress);
      }

      // copy the compressed KRLS model
      Zc.npca = Z.npca;

      // merge all features
      if(params.desc_npca==0){
        x = X.pca;
        z = Zc.pca;
      }else if(params.desc_pca==0){
        x = X.npca;
        z = Z.npca;
      }else{
        merge(X.vec,2,x);
        merge(Zc.vec,2,z);
      }

      //compute the gaussian kernel
      denseGaussKernel(params.sigma,x,z,k,layers,vxf,vyf,vxyf,_xy,_xyf);

      // compute the fourier transform of the kernel
      fft2(k,kf);
      if(frame==1)spec2=Mat_<Vec2d >(kf.rows, kf.cols);

      // calculate filter response
      if(params.split_coeff)
        calcResponse(alphaf,alphaf_den,kf,response, spec, spec2);
      else
        calcResponse(alphaf,kf,response, spec);

      // extract the maximum response
      minMaxLoc( response, &minVal, &maxVal, &minLoc, &maxLoc );
      roi.x+=(maxLoc.x-roi.width/2+1);
      roi.y+=(maxLoc.y-roi.height/2+1);

      // update the bounding box
      boundingBox.x=(resizeImage?roi.x*2:roi.x)+boundingBox.width/2;
      boundingBox.y=(resizeImage?roi.y*2:roi.y)+boundingBox.height/2;
    }

    // extract the patch for learning purpose
    // get non compressed descriptors
    for(unsigned i=0;i<descriptors_npca.size();i++){
      if(!getSubWindow(img,roi, features_npca[i], img_Patch, descriptors_npca[i]))return false;
    }
    if(features_npca.size()>0)merge(features_npca,X.npca);

    // get compressed descriptors
    for(unsigned i=0;i<descriptors_pca.size();i++){
      if(!getSubWindow(img,roi, features_pca[i], img_Patch, descriptors_pca[i]))return false;
    }
    if(features_pca.size()>0)merge(features_pca,X.pca);

    //update the training data
    if(frame==0){
      Z.pca = X.pca.clone();
      Z.npca = X.npca.clone();
    }else{
      Z.pca=(1.0-params.interp_factor)*Z.pca+params.interp_factor*X.pca;
      Z.npca=(1.0-params.interp_factor)*Z.npca+params.interp_factor*X.npca;
    }

    if(params.desc_pca !=0/*params.compress_feature*/ /*&& params.descriptor != GRAY*/){
      // initialize the vector of Mat variables
      if(frame==0){
        _layers_pca.resize(Z.pca.channels());
        _average.resize(Z.pca.channels());
      }

      // feature compression
      updateProjectionMatrix(Z.pca,old_cov_mtx,proj_mtx,params.pca_learning_rate,params.compressed_size,_layers_pca,_average,data_pca, new_covar,_w,_u,_vt);
      compress(proj_mtx,X.pca,X.pca,_data,_compress);
    }

    // merge all features
    if(params.desc_npca==0)
      x = X.pca;
    else if(params.desc_pca==0)
      x = X.npca;
    else
      merge(X.vec,2,x);

    // initialize some required Mat variables
    if(frame==0){
      layers.resize(x.channels());
      vxf.resize(x.channels());
      vyf.resize(x.channels());
      vxyf.resize(vyf.size());
      new_alphaf=Mat_<Vec2d >(yf.rows, yf.cols);
    }

    // Kernel Regularized Least-Squares, calculate alphas
    denseGaussKernel(params.sigma,x,x,k,layers,vxf,vyf,vxyf,_xy,_xyf);

    // compute the fourier transform of the kernel and add a small value
    fft2(k,kf);
    kf_lambda=kf+params.lambda;

    double den;
    if(params.split_coeff){
      mulSpectrums(yf,kf,new_alphaf,0);
      mulSpectrums(kf,kf_lambda,new_alphaf_den,0);
    }else{
      for(int i=0;i<yf.rows;i++){
        for(int j=0;j<yf.cols;j++){
          den = 1.0/(kf_lambda.at<Vec2d>(i,j)[0]*kf_lambda.at<Vec2d>(i,j)[0]+kf_lambda.at<Vec2d>(i,j)[1]*kf_lambda.at<Vec2d>(i,j)[1]);

          new_alphaf.at<Vec2d>(i,j)[0]=
          (yf.at<Vec2d>(i,j)[0]*kf_lambda.at<Vec2d>(i,j)[0]+yf.at<Vec2d>(i,j)[1]*kf_lambda.at<Vec2d>(i,j)[1])*den;
          new_alphaf.at<Vec2d>(i,j)[1]=
          (yf.at<Vec2d>(i,j)[1]*kf_lambda.at<Vec2d>(i,j)[0]-yf.at<Vec2d>(i,j)[0]*kf_lambda.at<Vec2d>(i,j)[1])*den;
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
  void inline TrackerKCFImpl::fft2(const Mat src, Mat & _dest) const {
    dft(src,_dest,DFT_COMPLEX_OUTPUT);
  }

  void inline TrackerKCFImpl::fft2(const Mat src, std::vector<Mat> & _dest, std::vector<Mat> & _layers) const {
    split(src, _layers);

    for(int i=0;i<src.channels();i++){
      dft(_layers[i],_dest[i],DFT_COMPLEX_OUTPUT);
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

  /*
   * obtains the projection matrix using PCA
   */
  void inline TrackerKCFImpl::updateProjectionMatrix(const Mat src, Mat & old_cov,Mat &  _proj_mtx, double pca_rate, int compressed_sz,
                                                     std::vector<Mat> & layers_pca,std::vector<Scalar> & average, Mat _data_pca, Mat _new_cov, Mat w, Mat u, Mat vt) const {
    CV_Assert(compressed_sz<=src.channels());

    split(src,layers_pca);

    for (int i=0;i<src.channels();i++){
      average[i]=mean(layers_pca[i]);
      layers_pca[i]-=average[i];
    }

    // calc covariance matrix
    merge(layers_pca,_data_pca);
    _data_pca=_data_pca.reshape(1,src.rows*src.cols);

    _new_cov=1.0/(double)(src.rows*src.cols-1)*(_data_pca.t()*_data_pca);
    if(old_cov.rows==0)old_cov=_new_cov.clone();

    // calc PCA
    SVD::compute((1.0-pca_rate)*old_cov+pca_rate*_new_cov, w, u, vt);

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
  void inline TrackerKCFImpl::compress(const Mat _proj_mtx, const Mat src, Mat & dest, Mat & data, Mat & compressed) const {
    data=src.reshape(1,src.rows*src.cols);
    compressed=data*_proj_mtx;
    dest=compressed.reshape(_proj_mtx.cols,src.rows).clone();
  }

  /*
   * obtain the patch and apply hann window filter to it
   */
  bool TrackerKCFImpl::getSubWindow(const Mat img, const Rect _roi, Mat& feat, Mat& patch, TrackerKCF::MODE desc) const {

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
    switch(desc){
      default:
        if(img.channels()>1)
          cvtColor(patch,feat, CV_BGR2GRAY);
        else
          feat=patch;
        feat.convertTo(feat,CV_64F);
        feat=feat/255.0-0.5; // normalize to range -0.5 .. 0.5
        feat=feat.mul(hann); // hann window filter
        break;
      case CN:
        CV_Assert(img.channels() == 3);
        extractCN(patch,feat);
        feat=feat.mul(hann_cn); // hann window filter
        break;
    }

    return true;

  }

  /* Convert BGR to ColorNames
   */
  void TrackerKCFImpl::extractCN(Mat _patch, Mat & cnFeatures) const {
    Vec3b & pixel = _patch.at<Vec3b>(0,0);
    unsigned index;

    if(cnFeatures.type() != CV_64FC(10))
      cnFeatures = Mat::zeros(_patch.rows,_patch.cols,CV_64FC(10));

    for(int i=0;i<_patch.rows;i++){
      for(int j=0;j<_patch.cols;j++){
        pixel=_patch.at<Vec3b>(i,j);
        index=(unsigned)(floor(pixel[2]/8)+32*floor(pixel[1]/8)+32*32*floor(pixel[0]/8));

        //copy the values
        for(int _k=0;_k<10;_k++){
          cnFeatures.at<Vec<double,10> >(i,j)[_k]=ColorNames[index][_k];
        }
      }
    }

  }

  /*
   *  dense gauss kernel function
   */
  void TrackerKCFImpl::denseGaussKernel(const double sigma, const Mat _x, const Mat _y, Mat & _k,
                                        std::vector<Mat> & _layers,std::vector<Mat> & _xf,std::vector<Mat> & _yf, std::vector<Mat> xyf_v, Mat xy, Mat xyf ) const {
    double normX, normY;

    fft2(_x,_xf,_layers);
    fft2(_y,_yf,_layers);

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
  void TrackerKCFImpl::calcResponse(const Mat _alphaf, const Mat _kf, Mat & _response, Mat & _spec) const {
    //alpha f--> 2channels ; k --> 1 channel;
    mulSpectrums(_alphaf,_kf,_spec,0,false);
    ifft2(spec,_response);
  }

  /*
   * calculate the detection response for splitted form
   */
  void TrackerKCFImpl::calcResponse(const Mat _alphaf, const Mat _alphaf_den, const Mat _kf, Mat & _response, Mat & _spec, Mat & _spec2) const {

    mulSpectrums(_alphaf,_kf,_spec,0,false);

    //z=(a+bi)/(c+di)=[(ac+bd)+i(bc-ad)]/(c^2+d^2)
    double den;
    for(int i=0;i<_kf.rows;i++){
      for(int j=0;j<_kf.cols;j++){
        den=1.0/(_alphaf_den.at<Vec2d>(i,j)[0]*_alphaf_den.at<Vec2d>(i,j)[0]+_alphaf_den.at<Vec2d>(i,j)[1]*_alphaf_den.at<Vec2d>(i,j)[1]);
        _spec2.at<Vec2d>(i,j)[0]=
          (_spec.at<Vec2d>(i,j)[0]*_alphaf_den.at<Vec2d>(i,j)[0]+_spec.at<Vec2d>(i,j)[1]*_alphaf_den.at<Vec2d>(i,j)[1])*den;
        _spec2.at<Vec2d>(i,j)[1]=
          (_spec.at<Vec2d>(i,j)[1]*_alphaf_den.at<Vec2d>(i,j)[0]-_spec.at<Vec2d>(i,j)[0]*_alphaf_den.at<Vec2d>(i,j)[1])*den;
      }
    }

    ifft2(_spec2,_response);
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
      split_coeff=true;
      wrap_kernel=false;
      desc_npca = GRAY;
      desc_pca = CN;

      //feature compression
      compress_feature=true;
      compressed_size=2;
      pca_learning_rate=0.15;
  }

  void TrackerKCF::Params::read( const cv::FileNode& /*fn*/ ){}

  void TrackerKCF::Params::write( cv::FileStorage& /*fs*/ ) const{}

} /* namespace cv */
