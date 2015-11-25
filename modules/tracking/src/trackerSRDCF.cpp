#include "precomp.hpp"


namespace cv{

typedef Eigen::Triplet<float> trip;

struct TrackedRegion{
	TrackedRegion(){ }
	TrackedRegion(const cv::Point2i init_center, const cv::Size init_size) : center(init_center), size(init_size){ }
	TrackedRegion(const cv::Rect box) : center(box.x + round((float)box.size().width /2.0),
						   box.y + round((float)box.size().height/2.0)),
						   size(box.size()){ }

	cv::Rect Rect() const{
		return cv::Rect(center.x-floor((float)size.width /2.0),
						center.y-floor((float)size.height/2.0),
						size.width,size.height);
	}

	TrackedRegion resize(const float factor) const{
		TrackedRegion newRegion;
		newRegion.center = center;
		newRegion.size = cv::Size(round(size.width *factor),
						          round(size.height*factor));
		return newRegion;
	}

	cv::Point2i center;
	cv::Size size;
};

class SRDCF : public TrackerSRDCF{
  public:
   SRDCF(const TrackerSRDCF::Params parameters) : p(parameters) {}
   ~SRDCF() {}

   bool initImpl(const Mat& image, const Rect2d& boundingBox);
   bool updateImpl(const Mat& image, Rect2d& boundingBox);
   void read( const cv::FileNode& fn );
   void write( cv::FileStorage& fs ) const;

   /*
    * Initialize the tracker on the region specified in the image
    */
   void initialize(const cv::Mat& image, const cv::Rect region);
   /*
    *  Return the current bounding box of the target as estimated by the tracker
    */
   cv::Rect getBoundingBox() const;

   /*
    *  Update the current estimate of the targets position from the image with the current bounding box estimate
    */
   void detect(const cv::Mat& image);

   /*
    * Update the current tracker model, from the current best position estimated by the tracker in the image provided
    */
   void update(const cv::Mat& image);

  private:

   //internal functions
   int get_num_real_coeff(const cv::Size matrix_sz);
   std::vector<cv::Mat> assemble_filter(const Eigen::VectorXf& filter_vector, const int nDim, const cv::Size filter_sz);
   void split_spectrum(const cv::Mat fft_matrix, std::vector<float>& real, std::vector<float>& image);

   std::vector<cv::Mat> fft2(const cv::Mat featureData);

   std::vector<cv::Mat> construct_sample_data(std::vector<cv::Mat> feture_matrices);

   Eigen::SparseMatrix<float> make_sparse_data(const std::vector<cv::Mat>& data);

   Eigen::VectorXf make_rhs(const std::vector<cv::Mat>& data, const cv::Mat labels);
   
   int shift_index(const int index, const int length) const;
   cv::Mat make_labels(const cv::Size matrix_size, const cv::Size target_size, const float sigma_factor) const;

   Eigen::SparseMatrix<float> make_regularization_matrix(const cv::Size sample_size, const int feature_dims);

   Eigen::VectorXf compute_filter(const Eigen::SparseMatrix<float>& lhs_data, const Eigen::SparseMatrix<float>& reg_matrix, const Eigen::VectorXf& rhs);

   cv::Mat compute_response(const std::vector<cv::Mat>& filter, const std::vector<cv::Mat>& sample);

   std::vector<cv::Mat> compute_feature_vec(const cv::Mat& patch);
   void update_impl(const cv::Mat& image, const TrackedRegion& region, const float update_rate);
   cv::Mat detect_impl(const cv::Mat& image, const TrackedRegion& region);
   cv::Mat channelMultiply(std::vector<cv::Mat> a, std::vector<cv::Mat> b, int flags, bool conjb);
   
   cv::Mat extractTrackedRegion(const cv::Mat image, const TrackedRegion region, const cv::Size output_sz);
   //parameters set on construction
   TrackerSRDCF::Params p;

   //internal state variables
   cv::Mat labelsf; //label function
   cv::Mat window; //cos (hann) window
   Eigen::SparseMatrix<float> regularizer; //regularization marix
   std::vector<cv::Mat> filterf;

   float scale_factor; //downsampling factor from image coordinates to tracker model coordinates
   Eigen::VectorXf filter_solution;
   Eigen::VectorXf rhs_state;
   Eigen::SparseMatrix<float> lhs_state;


   //current estimated position and size in pixel coordinates
   TrackedRegion target;
   
}; //end definition



  Ptr<TrackerSRDCF> TrackerSRDCF::createTracker(const TrackerSRDCF::Params &parameters){
    return Ptr<SRDCF>(new SRDCF(parameters));
  }

  void SRDCF::read( const cv::FileNode& fn ){
    p.read( fn );
  }

  void SRDCF::write( cv::FileStorage& fs ) const {
    p.write( fs );
  }


bool SRDCF::initImpl(const Mat& image, const Rect2d& boundingBox){
    initialize(image,boundingBox);
	return true;
}
bool SRDCF::updateImpl(const Mat& image, Rect2d& boundingBox){
	detect(image);
	cv::Rect new_bounding_box = getBoundingBox();
	update(image);
	boundingBox = new_bounding_box;
	return true;
}

//public functions
void SRDCF::initialize(const cv::Mat& image, const cv::Rect region){
   //convert region into internal representation defined from center pixel and size
   //including the padding
   target = TrackedRegion(region).resize(p.target_padding);

   scale_factor = sqrt(target.size.area() / p.model_sz.area());

   //create labels
   labelsf = make_labels(p.model_sz, target.resize((1.0/scale_factor)*(1.0/p.target_padding)).size, p.sigma_factor);

   //create regularizer
   //FIXME remove magic constant
   regularizer = make_regularization_matrix(p.model_sz,3);

   //create window function
   cv::createHanningWindow(window,p.model_sz,CV_32FC1);

   //create the initial filter from the init patch
   update_impl(image,target,1.0);
}


void SRDCF::update(const cv::Mat& image){
 update_impl(image,target,p.update_rate);
}

//private functions
std::vector<cv::Mat> SRDCF::compute_feature_vec(const cv::Mat& patch){
  std::vector<cv::Mat> feature_vec = fft2(patch);

  return feature_vec;
}

cv::Rect SRDCF::getBoundingBox() const{
  TrackedRegion return_bb = target.resize(1.0 / p.target_padding);
  return return_bb.Rect();
}

cv::Mat SRDCF::extractTrackedRegion(const cv::Mat image, const TrackedRegion region, const cv::Size output_sz){

    int xMin = region.center.x - floor(((float)region.size.width)/2.0);
    int yMin = region.center.y - floor(((float)region.size.height)/2.0);

    //int xMax = region.center.x + ceil(((float)region.size.width)/2.0);
    //int yMax = region.center.y + ceil(((float)region.size.height)/2.0);

    int xMax = xMin + region.size.width;
    int yMax = yMin + region.size.height;

    int xMinPad,xMaxPad,yMinPad,yMaxPad;

    if(xMin < 0){
        xMinPad = -xMin;
    }else{
        xMinPad = 0;
    }

    if(xMax > image.size().width){
        xMaxPad = xMax - image.size().width;
    }else{
        xMaxPad = 0;
    }

    if(yMin < 0){
        yMinPad = -yMin;
    }else{
        yMinPad = 0;
    }

    if(yMax > image.size().height){
        yMaxPad = yMax - image.size().height;
    }else{
        yMaxPad = 0;
    }

    //compute the acual rectangle we will extract from the image
    cv::Rect extractionRegion = cv::Rect(xMin + xMinPad,
                                         yMin + yMinPad,
                                         (xMax-xMin) - xMaxPad - xMinPad,
                                         (yMax-yMin) - yMaxPad - yMinPad);


    //make sure the patch is not completely outside the image
    if(extractionRegion.x + extractionRegion.width > 0 && 
       extractionRegion.y + extractionRegion.height > 0 &&
       extractionRegion.x < image.cols &&
       extractionRegion.y < image.rows){


        cv::Mat real_patch(region.size,image.type());


        //replicate along borders if needed
        if(xMinPad > 0 || xMaxPad > 0 || yMinPad > 0 || yMaxPad > 0){
            cv::copyMakeBorder(image(extractionRegion), real_patch, yMinPad, 
                               yMaxPad, xMinPad, xMaxPad, cv::BORDER_REPLICATE);

        }else{
            real_patch = image(extractionRegion);
        }

        if(!(real_patch.size().width == region.size.width && real_patch.size().height == region.size.height)){
            //cout << "kasst" << endl;
        }


      cv::Mat ds_patch;
      cv::resize(real_patch,ds_patch,output_sz);

      return ds_patch;

    }else{
        cv::Mat dummyRegion = cv::Mat::zeros(region.size,image.type());
	cv::Mat ds_patch;
	cv::resize(dummyRegion,ds_patch,output_sz);

        return ds_patch;
    }

}
void SRDCF::update_impl(const cv::Mat& image, const TrackedRegion& region, const float update_rate){
  //extract pixels to use for update
  cv::Mat pixels = extractTrackedRegion(image,region,p.model_sz); 
  std::vector<cv::Mat> feature_vecf = compute_feature_vec(pixels);
  
  //make lhs
  Eigen::SparseMatrix<float> lhs = make_sparse_data(feature_vecf);
  Eigen::VectorXf rhs = make_rhs(feature_vecf,labelsf);

  if(update_rate < 1.0){
     float update_rate_inv = (1.0 - update_rate);
     rhs_state = rhs * update_rate + rhs_state * update_rate_inv; 
     lhs_state = lhs * update_rate + lhs_state * update_rate_inv;
     //TODO klura ut vart exakt autokorrelationen hamnade nu 
  }else{
     rhs_state = rhs;
     lhs_state = lhs;
  }

  Eigen::VectorXf hf = compute_filter(lhs_state,regularizer,rhs_state);
  filterf = assemble_filter(hf,feature_vecf.size(),p.model_sz); 

}

cv::Mat SRDCF::channelMultiply(std::vector<cv::Mat> a, std::vector<cv::Mat> b, int flags, bool conjb){
    CV_Assert(a.size() == b.size());

    cv::Mat prod;
    cv::Mat sum = cv::Mat::zeros(a[0].size(),a[0].type());
    for(unsigned int i = 0; i < a.size(); ++i){
        cv::Mat ca = a[i];
        cv::Mat cb = b[i];
        cv::mulSpectrums(a[i],b[i],prod,flags,conjb);
        sum += prod;
    }
    return sum;
}

void SRDCF::detect(const cv::Mat& image){
  cv::Mat response = detect_impl(image,target);

  cv::Point2i maxpos;
  cv::minMaxLoc(response,NULL,NULL,NULL,&maxpos);

  //TODO remove me
  
  cv::Point2i translation(round(shift_index(maxpos.x,response.cols)*scale_factor),
		          round(shift_index(maxpos.y,response.rows)*scale_factor));

  target.center = target.center + translation;
}

cv::Mat SRDCF::detect_impl(const cv::Mat& image, const TrackedRegion& region){
  cv::Mat pixels = extractTrackedRegion(image,region,p.model_sz);

  std::vector<cv::Mat> feature_vecf = compute_feature_vec(pixels);

  return compute_response(filterf,feature_vecf);
}

int SRDCF::get_num_real_coeff(const cv::Size matrix_sz){
    int num_sym_coeff = 1;

    if (matrix_sz.width % 2 == 0 && matrix_sz.height % 2 == 0)
        num_sym_coeff = 4;
    else if (matrix_sz.width % 2 == 0 || matrix_sz.height % 2 == 0)
        num_sym_coeff = 2;

    int num_pos_coeff = (matrix_sz.area() - num_sym_coeff) / 2;

    return num_pos_coeff + num_sym_coeff;
}

std::vector<cv::Mat> SRDCF::assemble_filter(const Eigen::VectorXf& filter_vector, const int nDim, const cv::Size filter_sz){
    std::vector<cv::Mat> filter_matrix(nDim);
    int filter_index = 0;
    const int nRows = filter_sz.height;
    const int nCols = filter_sz.width;
    const int num_real_coeff = get_num_real_coeff(filter_sz);

    for(int d=0; d < nDim; d++){
        filter_index = num_real_coeff*2*d;
        filter_matrix[d] = cv::Mat(filter_sz,CV_32FC1);
        filter_matrix[d].at<float>(0,0) = filter_vector(filter_index);
        filter_index += 1;
        for(int row = 1; row < nRows-1; row += 2){
            filter_matrix[d].at<float>(row,0) = filter_vector(filter_index);
            filter_matrix[d].at<float>(row+1,0) = filter_vector(filter_index+num_real_coeff);
            filter_index += 1;
        }
        if(nRows % 2 == 0){
            filter_matrix[d].at<float>(nRows-1,0) = filter_vector(filter_index);
            filter_index += 1;
        }

        for(int row = 0; row < nRows; row++){
            for(int col=1; col < nCols-1; col += 2){
                filter_matrix[d].at<float>(row,col) = filter_vector(filter_index);
                filter_matrix[d].at<float>(row,col+1) = filter_vector(filter_index + num_real_coeff);
                filter_index += 1;
            }
        }

        if(nCols % 2 == 0){
            filter_matrix[d].at<float>(0,nCols-1) = filter_vector(filter_index);
            filter_index += 1;
            for(int row = 1; row < nRows-1; row+= 2){
                filter_matrix[d].at<float>(row,nCols-1) = filter_vector(filter_index);
                filter_matrix[d].at<float>(row+1,nCols-1) = filter_vector(filter_index + num_real_coeff);
                filter_index += 1;
            }
            if(nRows % 2 == 0){
                filter_matrix[d].at<float>(nRows-1,nCols-1) = filter_vector(filter_index);
                filter_index += 1;
            }
        }
    }

    return filter_matrix;
}

void SRDCF::split_spectrum(const cv::Mat fft_matrix, std::vector<float>& real, std::vector<float>& imag){
    //exact number of elemts plus some extra
    real.clear();
    imag.clear();

    real.reserve(fft_matrix.rows*fft_matrix.cols/2+5);
    imag.reserve(fft_matrix.rows*fft_matrix.cols/2+5);

    int nRows = fft_matrix.rows;
    int nCols = fft_matrix.cols;

    real.push_back(fft_matrix.at<float>(0,0)); //DC component
    imag.push_back(0.0); //always zero imag for DC

    for(int row = 1; row < nRows-1; row = row + 2){
        real.push_back(fft_matrix.at<float>(row,0));
        imag.push_back(fft_matrix.at<float>(row+1,0));
    }
    if(nRows % 2 == 0){
        real.push_back(fft_matrix.at<float>(nRows-1,0));
        imag.push_back(0.0);
    }


    for(int row = 0; row < nRows; row++){
        for(int col=1; col < fft_matrix.cols-1; col = col + 2){
            real.push_back(fft_matrix.at<float>(row,col));
            imag.push_back(fft_matrix.at<float>(row,col+1));
        }
    }

    if(nCols % 2 == 0){
        real.push_back(fft_matrix.at<float>(0,nCols-1));
        imag.push_back(0);

        for(int row = 1; row < nRows-1; row = row + 2){
            real.push_back(fft_matrix.at<float>(row,nCols-1));
            imag.push_back(fft_matrix.at<float>(row+1,nCols-1));
        }

        if( nRows % 2 == 0){
            real.push_back(fft_matrix.at<float>(nRows-1,nCols-1));
            imag.push_back(0);
        }
    }
}

std::vector<cv::Mat> SRDCF::fft2(const cv::Mat featureData){
    std::vector<cv::Mat> channels(featureData.channels());
    std::vector<cv::Mat> channelsf(featureData.channels());
    cv::split(featureData,channels);

    for(size_t i=0; i < channels.size(); ++i){
	
	cv::Mat windowed;
	cv::multiply(channels[i],window,windowed);

        cv::dft(windowed,channelsf[i],0);
    }

    return channelsf;
}



std::vector<cv::Mat> SRDCF::construct_sample_data(std::vector<cv::Mat> feature_matrices){

    const int nFeat = feature_matrices.size();
    std::vector<cv::Mat> outer_products(nFeat*nFeat);
    //compute outer product of features
    for(size_t feat_row = 0; feat_row < feature_matrices.size(); feat_row++){
        for(size_t feat_col =0; feat_col < feature_matrices.size(); feat_col++){
            cv::mulSpectrums(feature_matrices[feat_col],feature_matrices[feat_row],outer_products[feat_col*nFeat+feat_row],0,true);
        }
    }
    return outer_products;
}

Eigen::SparseMatrix<float> SRDCF::make_sparse_data(const std::vector<cv::Mat>& data){
    const int feature_dim = data.size();

    std::vector<cv::Mat> outer_products = construct_sample_data(data);
    std::vector<trip> triplet_list;
    //Dont ask...
    triplet_list.reserve((outer_products.size()*(outer_products[0].total()+10))*2);
    std::vector<float> real,imag;
    for(size_t i=0; i < outer_products.size(); i++){

        int block_i = i % feature_dim;
        int block_j = i / feature_dim;
        split_spectrum(outer_products[i],real,imag);

        int block_start_i = real.size() * 2 * block_i;
        int block_start_j = real.size() * 2 * block_j;

        int diag_length = real.size();
        for(int ii=0; ii < diag_length; ii++){
            //real coefficents
            triplet_list.push_back(trip(block_start_i+ii, block_start_j+ii,real[ii]));
            triplet_list.push_back(trip(diag_length+block_start_i+ii, diag_length+block_start_j+ii,real[ii]));

            //imaginary coefficents
            triplet_list.push_back(trip(block_start_i+ii,block_start_j+diag_length+ii,-imag[ii]));
            triplet_list.push_back(trip(block_start_i+ii+diag_length,block_start_j+ii,imag[ii]));
        }
    }

    int matrix_size = 2 * real.size() * feature_dim;

    std::vector<clock_t> times;
    Eigen::SparseMatrix<float> sparse_data(matrix_size,matrix_size);
    sparse_data.reserve(triplet_list.size());
    sparse_data.setFromTriplets(triplet_list.begin(),triplet_list.end());

    return sparse_data;
}

Eigen::VectorXf SRDCF::make_rhs(const std::vector<cv::Mat>& data, const cv::Mat labels){

    std::vector<cv::Mat> products(data.size());

    for(size_t i=0; i < data.size(); i++){
        cv::mulSpectrums(labels,data[i],products[i],0,true);
    }

    std::vector<float> real,imag;
    Eigen::VectorXf rhs;
    int element_index = 0;
    for(size_t i=0; i < products.size(); i++){
        split_spectrum(products[i],real,imag);
        if(i == 0){
            rhs = Eigen::VectorXf(real.size()*2*data.size());
        }

        for(size_t jj = 0; jj < real.size(); jj++){
            rhs(element_index) = real[jj];
            element_index += 1;
        }
        for(size_t j =0; j < imag.size(); j++){
            rhs(element_index) = imag[j];
            element_index += 1;
        }
    }

    return rhs;
} 

/*
 * @params index to shift
 * @params length of the matrix side to shift
 * @return the shifted index
 */
int SRDCF::shift_index(const int index, const int length) const{
    int shifted_index;

    if(index > length/2){
        shifted_index = -length + index;
    }else{
        shifted_index = index;
    }

    return shifted_index;
}

/*
 *  @params matrix_size, size of the labling function, same as the filter
 *  @params target_size, size of the target in the feature representation
 *  @params sigma_factor, sigma scale factor (tracker parameter)
 */
cv::Mat SRDCF::make_labels(const cv::Size matrix_size, const cv::Size target_size ,const float sigma_factor) const{
    cv::Mat new_labels(matrix_size.height,matrix_size.width,CV_32F);

    std::cout << "matrix size " << matrix_size << " target_size " << target_size << std::endl;

    const float sigma = std::sqrt(target_size.area()) * sigma_factor;
    const float constant = -0.5 / pow(sigma,2);
    std::cout << "constant is" << constant << std::endl;
    for(int x = 0; x < matrix_size.width; x++){
        for(int y = 0; y < matrix_size.height; y++){
            int shift_x = shift_index(x,matrix_size.width);
            int shift_y = shift_index(y,matrix_size.height);
	    float value =  std::exp(constant*(std::pow(shift_x,2) + std::pow(shift_y,2)));
            new_labels.at<float>(y,x) = value;
	}
    }

    cv::Mat labels_dft;


    cv::dft(new_labels, labels_dft);

    return labels_dft;
}

Eigen::SparseMatrix<float> SRDCF::make_regularization_matrix(const cv::Size sample_size, const int feature_dims){
    //const float reg_power = 2;
    const float reg_min = 0.1;
    //const float sparsity_treshold = 0.05;

    const int mat_size = get_num_real_coeff(sample_size) * 2 * feature_dims;

    Eigen::SparseMatrix<float> reg_matrix(mat_size, mat_size);
    reg_matrix.setIdentity();

    return reg_matrix * (reg_min * reg_min);
}


Eigen::VectorXf SRDCF::compute_filter(const Eigen::SparseMatrix<float>& lhs_data, const Eigen::SparseMatrix<float>& reg_matrix, const Eigen::VectorXf& rhs){

    Eigen::SparseMatrix<float> lhs = lhs_data + reg_matrix;



    if(filter_solution.rows() > 0 && filter_solution.cols() > 0){
      Eigen::ConjugateGradient< Eigen::SparseMatrix<float> > solver;
      solver.setMaxIterations(10);
      solver.compute(lhs);
      //filter_solution = solver.solveWithGuess(rhs,filter_solution);
      filter_solution = solver.solve(rhs);
    }else{
      Eigen::ConjugateGradient< Eigen::SparseMatrix<float> > solver;
      solver.setMaxIterations(50);
      solver.compute(lhs);
      filter_solution = solver.solve(rhs);
    }

    return filter_solution;
}

cv::Mat SRDCF::compute_response(const std::vector<cv::Mat>& filter, const std::vector<cv::Mat>& sample) {
    cv::Mat response;

    cv::Mat resp_dft = channelMultiply(filter, sample, 0, false);
    cv::dft(resp_dft, response, cv::DFT_INVERSE | cv::DFT_SCALE);

    return response;
}
}