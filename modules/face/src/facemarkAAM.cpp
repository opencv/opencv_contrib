#include "opencv2/face.hpp"
#include "opencv2/imgcodecs.hpp"
#include "precomp.hpp"

namespace cv
{
    //namespace face {

    /*
    * Parameters
    */
    FacemarkAAM::Params::Params(){
        detect_thresh = 0.5;
        sigma=0.2;
    }

    void FacemarkAAM::Params::read( const cv::FileNode& fn ){
        *this = FacemarkAAM::Params();

        if (!fn["detect_thresh"].empty())
            fn["detect_thresh"] >> detect_thresh;

        if (!fn["sigma"].empty())
            fn["sigma"] >> sigma;

    }

    void FacemarkAAM::Params::write( cv::FileStorage& fs ) const{
        fs << "detect_thresh" << detect_thresh;
        fs << "sigma" << sigma;
    }

    class FacemarkAAMImpl : public FacemarkAAM {
    public:
        FacemarkAAMImpl( const FacemarkAAM::Params &parameters = FacemarkAAM::Params() );
        void read( const FileNode& /*fn*/ );
        void write( FileStorage& /*fs*/ ) const;

        void saveModel(FileStorage& fs);
        void loadModel(FileStorage& fs);

    protected:

        bool detectImpl( InputArray image, std::vector<Point2f> & landmarks );
        void trainingImpl(String imageList, String groundTruth, const FacemarkAAM::Params &parameters);
        void trainingImpl(String imageList, String groundTruth);

        Mat procrustes(std::vector<Point2f> , std::vector<Point2f> , Mat & , Scalar & , float & );
        void calcMeanShape(std::vector<std::vector<Point2f> > ,std::vector<Point2f> & );
        void procrustesAnalysis(std::vector<std::vector<Point2f> > , std::vector<std::vector<Point2f> > & , std::vector<Point2f> & );

        inline Mat linearize(std::vector<Point2f> );
        Mat getProjection(const Mat , int );
        void calcSimilarityEig(std::vector<Point2f> ,Mat , Mat & , Mat & );
        Mat orthonormal(Mat );
        void delaunay(std::vector<Point2f> , std::vector<Vec3i> & );
        Mat createMask(std::vector<Point2f> , Rect );
        Mat createTextureBase(std::vector<Point2f> , std::vector<Vec3i> , Rect , std::vector<std::vector<Point> > & );
        Mat warpImage(Mat , std::vector<Point2f> , std::vector<Point2f> , std::vector<Vec3i> , Rect , std::vector<std::vector<Point> > );
        Mat getFeature(const Mat , Mat , Mat ); // TODO: remove this function, use directly
        void createMaskMapping(const Mat , const Mat ,Mat & , Mat & , Mat & , Mat & , Mat & , Mat & );

        FacemarkAAM::Params params;
        FacemarkAAM::Model AAM;

    private:
        int test;
    };

    /*
    * Constructor
    */
    Ptr<FacemarkAAM> FacemarkAAM::create(const FacemarkAAM::Params &parameters){
        return Ptr<FacemarkAAMImpl>(new FacemarkAAMImpl(parameters));
    }

    Ptr<FacemarkAAM> FacemarkAAM::create(){
        return Ptr<FacemarkAAMImpl>(new FacemarkAAMImpl());
    }

    FacemarkAAMImpl::FacemarkAAMImpl( const FacemarkAAM::Params &parameters ) :
        params( parameters )
    {
        isSetDetector =false;
        test = 11;
    }

    void FacemarkAAMImpl::read( const cv::FileNode& fn ){
        params.read( fn );
    }

    void FacemarkAAMImpl::write( cv::FileStorage& fs ) const {
        params.write( fs );
    }


    // void FacemarkAAM::training(String imageList, String groundTruth, const FacemarkAAM::Params &parameters){
    //     trainingImpl(imageList, groundTruth, parameters);
    // }

    // void FacemarkAAMImpl::trainingImpl(String imageList, String groundTruth, const FacemarkAAM::Params &parameters){
    //     params = parameters;
    //     trainingImpl(imageList, groundTruth);
    // }

    void FacemarkAAMImpl::trainingImpl(String imageList, String groundTruth){

        std::vector<String> images;
        std::vector<std::vector<Point2f> > facePoints, normalized;
        Mat erode_kernel = getStructuringElement(MORPH_RECT, Size(3,3), Point(1,1));
        Mat image;

        /* initialize the values TODO: set them based on the params*/
        AAM.scales.push_back(1);
        AAM.scales.push_back(2);
        AAM.textures.resize(AAM.scales.size());

        /*-------------- A. Load the training data---------*/
        if(groundTruth==""){
            loadTrainingData(imageList, images, facePoints);
        }else{
            loadTrainingData(imageList, groundTruth, images, facePoints);
        }
        procrustesAnalysis(facePoints, normalized,AAM.s0);

        /*-------------- B. Create the shape model---------*/
        Mat s0_lin = linearize(AAM.s0).t() ;
        // linearize all shapes  data, all x and then all y for each shape
        Mat M;
        for(unsigned i=0;i<normalized.size();i++){
            M.push_back(linearize(normalized[i]).t()-s0_lin);
        }

        /* get PCA Projection vectors */
        Mat S = getProjection(M.t(),136);
        /* Create similarity eig*/
        Mat shape_S,shape_Q;
        calcSimilarityEig(AAM.s0,S,AAM.Q,AAM.S);

        /* ----------C. Create the coordinate frame ------------*/
        delaunay(AAM.s0,AAM.triangles);

        for(size_t scale=0; scale<AAM.scales.size();scale++){
            AAM.textures[scale].max_m = 145;
            printf("Training for scale %i ...\n", AAM.scales[scale]);
            Mat s0_scaled_m = Mat(AAM.s0)/AAM.scales[scale]; // scale the shape
            std::vector<Point2f> s0_scaled = s0_scaled_m.reshape(2); //convert to points

            /*get the min and max of x and y coordinate*/
            double min_x, max_x, min_y, max_y;
            s0_scaled_m = s0_scaled_m.reshape(1);
            Mat s0_scaled_x = s0_scaled_m.col(0);
            Mat s0_scaled_y = s0_scaled_m.col(1);
            minMaxIdx(s0_scaled_x, &min_x, &max_x);
            minMaxIdx(s0_scaled_y, &min_y, &max_y);

            std::vector<Point2f> base_shape = Mat(Mat(s0_scaled)-Scalar(min_x-2.0,min_y-2.0)).reshape(2);
            AAM.textures[scale].base_shape = base_shape;
            AAM.textures[scale].resolution = Rect(0,0,(int)ceil(max_x-min_x+3),(int)ceil(max_y-min_y+3));

            Mat base_texture = createTextureBase(base_shape, AAM.triangles, AAM.textures[scale].resolution, AAM.textures[scale].textureIdx);

            Mat mask = base_texture>0;
            Mat mask2;
            erode(mask, mask, erode_kernel);
            erode(mask, mask2, erode_kernel);

            Mat warped;
            createMaskMapping(mask,mask2,AAM.textures[scale].featMapx, AAM.textures[scale].featMapy,AAM.textures[scale].featMapx2, AAM.textures[scale].featMapy2,AAM.textures[scale].rec_y, AAM.textures[scale].map_erod2);
            Mat feat = getFeature(warped, AAM.textures[scale].featMapx, AAM.textures[scale].featMapy);

            AAM.textures[scale].mask2 = mask2.clone();
            /* ------------ Part D. Get textures -------------*/
            Mat texture_feats;
            for(size_t i=0; i<images.size();i++){
                image = imread(images[i]);
                warped = warpImage(image,base_shape, facePoints[i], AAM.triangles, AAM.textures[scale].resolution,AAM.textures[scale].textureIdx);
                feat = getFeature(warped, AAM.textures[scale].featMapx, AAM.textures[scale].featMapy);
                texture_feats.push_back(feat.t());
            }
            Mat textures= texture_feats.t();

            /* -------------- E. Create the texture model -----------------*/
            Mat T;
            textures.convertTo(T,CV_32F);
            reduce(T,AAM.textures[scale].A0,1, CV_REDUCE_AVG);

            Mat A0_mtx = repeat(AAM.textures[scale].A0,1,textures.cols);
            Mat textures_normalized = T - A0_mtx;

            AAM.textures[scale].A = getProjection(textures_normalized,550);

            remap(AAM.textures[scale].A0,AAM.textures[scale].AA0,AAM.textures[scale].featMapx2, AAM.textures[scale].featMapy2,INTER_NEAREST);

            Mat U_data;
            for(int i =0;i<AAM.textures[scale].A.cols;i++){
                Mat c = AAM.textures[scale].A.col(i);
                Mat ud;
                remap(c,ud,AAM.textures[scale].featMapx2, AAM.textures[scale].featMapy2,INTER_NEAREST);
                U_data.push_back(ud.t());
            }
            Mat U = U_data.t();
            AAM.textures[scale].AA = orthonormal(U);
        } // scale
        printf("training is finished\n");
    }

    bool FacemarkAAMImpl::detectImpl( InputArray image, std::vector<Point2f>& landmarks ){
        if (landmarks.size()>0)
            landmarks.clear();

        /*dummy function, will be updated soon*/
        landmarks.push_back(Point2f((float)2.0,(float)3.3));
        landmarks.push_back(Point2f((float)1.5,(float)2.2));
        Mat img = image.getMat();
        printf("detect::rows->%i landmarks-> %i\n",(int)img.rows,(int)landmarks.size());
        return true;
    }

    void FacemarkAAMImpl::saveModel(FileStorage& fs){
        fs << "AAM_tri" << AAM.triangles;
        fs << "scales" << AAM.scales;
        fs << "s0" << AAM.s0;
        fs << "S" << AAM.S;
        fs << "Q" << AAM.Q;

        char x[256];
        for(int i=0;i< (int)AAM.scales.size();i++){
            sprintf(x,"scale%i_max_m",i);
            fs << x << AAM.textures[i].max_m;

            sprintf(x,"scale%i_resolution",i);
            fs << x << AAM.textures[i].resolution;

            sprintf(x,"scale%i_textureIdx",i);
            fs << x << AAM.textures[i].textureIdx;

            sprintf(x,"scale%i_base_shape",i);
            fs << x << AAM.textures[i].base_shape;

            sprintf(x,"scale%i_featMapx",i);
            fs << x << AAM.textures[i].featMapx;

            sprintf(x,"scale%i_featMapy",i);
            fs << x << AAM.textures[i].featMapy;

            sprintf(x,"scale%i_featMapx2",i);
            fs << x << AAM.textures[i].featMapx2;

            sprintf(x,"scale%i_featMapy2",i);
            fs << x << AAM.textures[i].featMapy2;

            sprintf(x,"scale%i_A",i);
            fs << x << AAM.textures[i].A;

            sprintf(x,"scale%i_A0",i);
            fs << x << AAM.textures[i].A0;

            sprintf(x,"scale%i_AA",i);
            fs << x << AAM.textures[i].AA;

            sprintf(x,"scale%i_AA0",i);
            fs << x << AAM.textures[i].AA0;

            sprintf(x,"scale%i_mask2",i);
            fs << x << AAM.textures[i].mask2;

            sprintf(x,"scale%i_rec_y",i);
            fs << x << AAM.textures[i].rec_y;

            sprintf(x,"scale%i_map_erod2",i);
            fs << x << AAM.textures[i].map_erod2;

        }
        fs.release();
        printf("The model is successfully saved! \n");
    }

    void FacemarkAAMImpl::loadModel(FileStorage& fs){
        char x[256];
        fs["AAM_tri"] >> AAM.triangles;
        fs["scales"] >> AAM.scales;
        fs["s0"] >> AAM.s0;
        fs["S"] >> AAM.S;
        fs["Q"] >> AAM.Q;


        AAM.textures.resize(AAM.scales.size());
        for(int i=0;i< (int)AAM.scales.size();i++){
            sprintf(x,"scale%i_max_m",i);
            fs[x] >> AAM.textures[i].max_m;

            sprintf(x,"scale%i_resolution",i);
            fs[x] >> AAM.textures[i].resolution;

            sprintf(x,"scale%i_textureIdx",i);
            fs[x] >> AAM.textures[i].textureIdx;

            sprintf(x,"scale%i_base_shape",i);
            fs[x] >> AAM.textures[i].base_shape;

            sprintf(x,"scale%i_featMapx",i);
            fs[x] >> AAM.textures[i].featMapx;

            sprintf(x,"scale%i_featMapy",i);
            fs[x] >> AAM.textures[i].featMapy;

            sprintf(x,"scale%i_featMapx2",i);
            fs[x] >> AAM.textures[i].featMapx2;

            sprintf(x,"scale%i_featMapy2",i);
            fs[x] >> AAM.textures[i].featMapy2;

            sprintf(x,"scale%i_A",i);
            fs[x] >> AAM.textures[i].A;

            sprintf(x,"scale%i_A0",i);
            fs[x] >> AAM.textures[i].A0;

            sprintf(x,"scale%i_AA",i);
            fs[x] >> AAM.textures[i].AA;

            sprintf(x,"scale%i_AA0",i);
            fs[x] >> AAM.textures[i].AA0;

            sprintf(x,"scale%i_mask2",i);
            fs[x] >> AAM.textures[i].mask2;

            sprintf(x,"scale%i_rec_y",i);
            fs[x] >> AAM.textures[i].rec_y;

            sprintf(x,"scale%i_map_erod2",i);
            fs[x] >> AAM.textures[i].map_erod2;
        }

        fs.release();

        printf("the model has been loaded\n");
    }

    Mat FacemarkAAMImpl::procrustes(std::vector<Point2f> P, std::vector<Point2f> Q, Mat & rot, Scalar & trans, float & scale){

        // calculate average
        Scalar mx = mean(P);
        Scalar my = mean(Q);

        // zero centered data
        Mat X0 = Mat(P) - mx;
        Mat Y0 = Mat(Q) - my;

        // calculate magnitude
        Mat Xs, Ys;
        multiply(X0,X0,Xs);
        multiply(Y0,Y0,Ys);

        // cout<<Xs<<endl;

        // calculate the sum
        Mat sumXs, sumYs;
        reduce(Xs,sumXs, 0, CV_REDUCE_SUM);
        reduce(Ys,sumYs, 0, CV_REDUCE_SUM);

        //calculate the normrnd
        double normX = sqrt(sumXs.at<float>(0)+sumXs.at<float>(1));
        double normY = sqrt(sumYs.at<float>(0)+sumYs.at<float>(1));

        //normalization
        X0 = X0/normX;
        Y0 = Y0/normY;

        //reshape, convert to 2D Matrix
        Mat Xn=X0.reshape(1);
        Mat Yn=Y0.reshape(1);

        //calculate the covariance matrix
        Mat M = Xn.t()*Yn;

        // decompose
        Mat U,S,Vt;
        SVD::compute(M, S, U, Vt);

        // extract the transformations
        scale = (S.at<float>(0)+S.at<float>(1))*(float)normX/(float)normY;
        rot = Vt.t()*U.t();

        Mat muX(mx),mX; muX.pop_back();muX.pop_back();
        Mat muY(my),mY; muY.pop_back();muY.pop_back();
        muX.convertTo(mX,CV_32FC1);
        muY.convertTo(mY,CV_32FC1);

        Mat t = mX.t()-scale*mY.t()*rot;
        trans[0] = t.at<float>(0);
        trans[1] = t.at<float>(1);

        // calculate the recovered form
        Mat Qmat = Mat(Q).reshape(1);

        return Mat(scale*Qmat*rot+trans).clone();
    }

    void FacemarkAAMImpl::procrustesAnalysis(std::vector<std::vector<Point2f> > shapes, std::vector<std::vector<Point2f> > & normalized, std::vector<Point2f> & new_mean){

        std::vector<Scalar> mean_every_shape;
        mean_every_shape.resize(shapes.size());

        Point2f temp;

        // calculate the mean of every shape
        for(size_t i=0; i< shapes.size();i++){
            mean_every_shape[i] = mean(shapes[i]);
        }

        //normalize every shapes
        Mat tShape;
        normalized.clear();
        for(size_t i=0; i< shapes.size();i++){
            normalized.push_back((Mat)(Mat(shapes[i]) - mean_every_shape[i]));
        }

        // calculate the mean shape
        std::vector<Point2f> mean_shape;
        calcMeanShape(normalized, mean_shape);

        // update the mean shape and normalized shapes iteratively
        int maxIter = 100;
        Mat R;
        Scalar t;
        float s;
        Mat aligned;
        for(int i=0;i<maxIter;i++){
            // align
            for(unsigned k=0;k< normalized.size();k++){
                aligned=procrustes(mean_shape, normalized[k], R, t, s);
                aligned.reshape(2).copyTo(normalized[k]);
            }

            //calc new mean
            calcMeanShape(normalized, new_mean);
            // align the new mean
            aligned=procrustes(mean_shape, new_mean, R, t, s);
            // update
            aligned.reshape(2).copyTo(mean_shape);
        }
    }

    void FacemarkAAMImpl::calcMeanShape(std::vector<std::vector<Point2f> > shapes,std::vector<Point2f> & mean){
        mean.resize(shapes[0].size());
        Point2f tmp;
        for(unsigned i=0;i<shapes[0].size();i++){
            tmp.x=0;
            tmp.y=0;
            for(unsigned k=0;k< shapes.size();k++){
                tmp.x+= shapes[k][i].x;
                tmp.y+= shapes[k][i].y;
            }
            tmp.x/=shapes.size();
            tmp.y/=shapes.size();
            mean[i] = tmp;
        }
    }

    Mat FacemarkAAMImpl::getProjection(Mat M, int n){
        Mat U,S,Vt,S1,proj;
        int k;
        if(M.rows < M.cols){
            SVD::compute(M*M.t(), S, U, Vt);

            // find the minimum between number of non-zero eigval,
            // compressed dim, row, and column
            threshold(S,S1,0.00001,1,THRESH_BINARY);
            k= countNonZero(S1);
            if(k>n)k=n;
            if(k>M.rows)k=M.rows;
            if(k>M.cols)k=M.cols;

            // cut the column of eigen vector
            U.colRange(0,k).copyTo(proj);
        }else{
            SVD::compute(M.t()*M, S, U, Vt);

            threshold(S,S1,0.00001,1,THRESH_BINARY);
            k= countNonZero(S1);
            if(k>n)k=n;
            if(k>M.rows)k=M.rows;
            if(k>M.cols)k=M.cols;

            // cut the eigen values to k-amount
            Mat D = Mat::zeros(k,k,CV_32FC1);
            Mat diag = D.diag();
            Mat s; pow(S,-0.5,s);
            s(Range(0,k), Range::all()).copyTo(diag);

            // cut the eigen vector to k-column,
            proj = M*U.colRange(0,k)*D;

        }
        return  proj.clone();
    }

    Mat FacemarkAAMImpl::orthonormal(Mat Mo){
        Mat M;
        Mo.convertTo(M,CV_32FC1);

        // TODO: float precission is only 1e-7, but MATLAB version use thresh=2.2204e-16
        float thresh = (float)2.2204e-6;

        Mat O = Mat::zeros(M.rows, M.cols, CV_32FC1);

        int k = 0; //storing index

        Mat w,nv;
        float n;
        for(int i=0;i<M.cols;i++){
            Mat v = M.col(i); // processed column to orthogonalize

            // subtract projection over previous vectors
            for(int j=0;j<k;j++){
                Mat o=O.col(j);
                w = v-o*(o.t()*v);
                w.copyTo(v);
            }

            // only keep non zero vector
            n = (float)norm(v);
            if(n>thresh){
                Mat ok=O.col(k);
                // nv=v/n;
                normalize(v,nv);
                nv.copyTo(ok);
                k+=1;
            }

        }

        return O.colRange(0,k).clone();
    }

    void FacemarkAAMImpl::calcSimilarityEig(std::vector<Point2f> s0,Mat S, Mat & Q_orth, Mat & S_orth){
        // cout<<"similarity eig"<<endl;
        int npts = (int)s0.size();

        Mat Q = Mat::zeros(2*npts,4,CV_32FC1);
        Mat c0 = Q.col(0);
        Mat c1 = Q.col(1);
        Mat c2 = Q.col(2);
        Mat c3 = Q.col(3);

        /*c0 = s0(:)*/
        Mat w = linearize(s0);
        // w.convertTo(w, CV_64FC1);
        w.copyTo(c0);

        /*c1 = [-s0(npts:2*npts); s0(0:npts-1)]*/
        Mat s0_mat = Mat(s0).reshape(1);
        // s0_mat.convertTo(s0_mat, CV_64FC1);
        Mat swapper = Mat::zeros(2,npts,CV_32FC1);
        Mat s00 = s0_mat.col(0);
        Mat s01 = s0_mat.col(1);
        Mat sw0 = swapper.row(0);
        Mat sw1 = swapper.row(1);
        Mat(s00.t()).copyTo(sw1);
        s01 = -s01;
        Mat(s01.t()).copyTo(sw0);

        Mat(swapper.reshape(1,2*npts)).copyTo(c1);

        /*c2 - [ones(npts); zeros(npts)]*/
        Mat ones = Mat::ones(1,npts,CV_32FC1);
        Mat c2_mat = Mat::zeros(2,npts,CV_32FC1);
        Mat c20 = c2_mat.row(0);
        ones.copyTo(c20);
        Mat(c2_mat.reshape(1,2*npts)).copyTo(c2);

        /*c3 - [zeros(npts); ones(npts)]*/
        Mat c3_mat = Mat::zeros(2,npts,CV_32FC1);
        Mat c31 = c3_mat.row(1);
        ones.copyTo(c31);
        Mat(c3_mat.reshape(1,2*npts)).copyTo(c3);
        // cout<<Q<<endl;

        Mat Qo = orthonormal(Q);
        // cout<<Qo<<endl;
        // matInfo(Qo);

        Mat all = Qo.t();
        all.push_back(S.t());

        Mat allOrth = orthonormal(all.t());
        // cout<<allOrth.colRange(0,8)<<endl;
        Q_orth =  allOrth.colRange(0,4).clone();
        S_orth =  allOrth.colRange(4,allOrth.cols).clone();

    }

    inline Mat FacemarkAAMImpl::linearize(std::vector<Point2f> s){ // all x values and then all y values
        return Mat(Mat(s).reshape(1).t()).reshape(1,2*(int)s.size());
    }

    void FacemarkAAMImpl::delaunay(std::vector<Point2f> s, std::vector<Vec3i> & triangles){

        triangles.clear();

        std::vector<int> idx;
        std::vector<Vec6f> tp;

        double min_x, max_x, min_y, max_y;
        Mat S = Mat(s).reshape(1);
        Mat s_x = S.col(0);
        Mat s_y = S.col(1);
        minMaxIdx(s_x, &min_x, &max_x);
        minMaxIdx(s_y, &min_y, &max_y);

        // TODO: set the rectangle as configurable parameter
        Subdiv2D subdiv(Rect(-500,-500,1000,1000));
        subdiv.insert(s);

        int a,b;
        subdiv.locate(s.back(),a,b);
        idx.resize(b+1);

        Point2f p;
        for(unsigned i=0;i<s.size();i++){
            subdiv.locate(s[i],a,b);
            idx[b] = i;
        }

        int v1,v2,v3;
        subdiv.getTriangleList(tp);

        for(unsigned i=0;i<tp.size();i++){
            Vec6f t = tp[i];

            //accept only vertex point
            if(t[0]>=min_x && t[0]<=max_x && t[1]>=min_y && t[1]<=max_y
                && t[2]>=min_x && t[2]<=max_x && t[3]>=min_y && t[3]<=max_y
                && t[4]>=min_x && t[4]<=max_x && t[5]>=min_y && t[5]<=max_y
            ){
                subdiv.locate(Point2f(t[0],t[1]),a,v1);
                subdiv.locate(Point2f(t[2],t[3]),a,v2);
                subdiv.locate(Point2f(t[4],t[5]),a,v3);
                triangles.push_back(Vec3i(idx[v1],idx[v2],idx[v3]));
            } //if
        } // for
    }

    Mat FacemarkAAMImpl::createMask(std::vector<Point2f> base_shape,  Rect res){
        Mat mask = Mat::zeros(res.height, res.width, CV_8U);
        std::vector<Point> hull;
        std::vector<Point> shape;
        Mat(base_shape).convertTo(shape, CV_32S);
        convexHull(shape,hull);
        fillConvexPoly(mask, &hull[0], (int)hull.size(), 255, 8 ,0);
        return mask.clone();
    }

    Mat FacemarkAAMImpl::createTextureBase(std::vector<Point2f> shape, std::vector<Vec3i> triangles, Rect res, std::vector<std::vector<Point> > & textureIdx){
        // max supported amount of triangles only 255
        Mat mask = Mat::zeros(res.height, res.width, CV_8U);

        std::vector<Point2f> p(3);
        textureIdx.clear();
        for(size_t i=0;i<triangles.size();i++){
            p[0] = shape[triangles[i][0]];
            p[1] = shape[triangles[i][1]];
            p[2] = shape[triangles[i][2]];


            std::vector<Point> polygon;
            approxPolyDP(p,polygon, 1.0, true);
            fillConvexPoly(mask, &polygon[0], (int)polygon.size(), (double)i+1,8,0 );

            std::vector<Point> list;
            for(int y=0;y<res.height;y++){
                for(int x=0;x<res.width;x++){
                    if(mask.at<uchar>(y,x)==(uchar)(i+1)){
                        list.push_back(Point(x,y));
                    }
                }
            }
            textureIdx.push_back(list);

        }

        return mask.clone();
    }

    Mat FacemarkAAMImpl::warpImage(Mat img, std::vector<Point2f> target_shape, std::vector<Point2f> curr_shape, std::vector<Vec3i> triangles, Rect res, std::vector<std::vector<Point> > textureIdx){
        // TODO: this part can be optimized, collect tranformation pair form all triangles first, then do one time remapping
        Mat warped = Mat::zeros(res.height, res.width, CV_8U);
        Mat warped2 = Mat::zeros(res.height, res.width, CV_8U);
        Mat image,part, warped_part;

        if(img.channels()>1){
            cvtColor(img,image,CV_BGR2GRAY);
        }else{
            image = img;
        }

        Mat A,R,t;
        std::vector<Point2f> target(3),source(3);
        std::vector<Point> polygon;
        for(size_t i=0;i<triangles.size();i++){
            target[0] = target_shape[triangles[i][0]];
            target[1] = target_shape[triangles[i][1]];
            target[2] = target_shape[triangles[i][2]];

            source[0] = curr_shape[triangles[i][0]];
            source[1] = curr_shape[triangles[i][1]];
            source[2] = curr_shape[triangles[i][2]];

            A = getAffineTransform(target,source);

            approxPolyDP(target,polygon, 1.0, true);

            Mat mask = Mat::zeros(warped.size(), CV_8U);
            fillConvexPoly(mask, &polygon[0], (int)polygon.size(), 255,8,0 );

            R=A.colRange(0,2);
            t=A.colRange(2,3);

            Mat pts = Mat(textureIdx[i]).reshape(1).t();

            Mat pts_f;
            pts.convertTo(pts_f,CV_64FC1);
            pts_f.push_back(Mat::ones(1,(int)textureIdx[i].size(),CV_64FC1));

            Mat trans = (A*pts_f).t();
            Mat map;
            trans.convertTo(map, CV_32F);

            std::vector<Point2f> dest = trans.reshape(2);
            Mat map_x=-1*Mat::ones(res.height, res.width,  CV_32F);
            Mat map_y=-1*Mat::ones(res.height, res.width,  CV_32F);

            for(size_t k =0;k<textureIdx[i].size();k++){
                map_y.at<float>(textureIdx[i][k].y,textureIdx[i][k].x) = dest[k].y;
                map_x.at<float>(textureIdx[i][k].y,textureIdx[i][k].x) = dest[k].x;
            }

            remap(image, warped, map_x, map_y, INTER_NEAREST );
            warped.copyTo(warped2, mask);
        }

        return warped2.clone();
    }

    Mat FacemarkAAMImpl::getFeature(const Mat img, Mat map_x, Mat map_y){
        Mat feat;
        remap(img, feat, map_x, map_y, INTER_NEAREST );
        return feat.clone();
    }

    void FacemarkAAMImpl::createMaskMapping(const Mat mask, const Mat mask2, Mat & map_x, Mat & map_y,  Mat & map_x2, Mat & map_y2, Mat & rec_y, Mat & map_erod2){
        std::vector<float> x,x2;
        std::vector<float> y,y2;
        std::vector<float> y_direct;
        int cnt = 0, idx=0;

        rec_y = -1*Mat::ones(mask.rows, mask.cols,  CV_32F);

        for(int i=0;i<mask.rows;i++){
            for(int j=0;j<mask.cols;j++){
                if(mask.at<uchar>(i,j)>0){
                    x.push_back((float)j);
                    y.push_back((float)i);
                    if(mask2.at<uchar>(i,j)>0){
                        x2.push_back(0.0);
                        y2.push_back((float)cnt);

                        y_direct.push_back((float)idx);
                    }

                    rec_y.at<float>(i,j) = (float)cnt;

                    cnt +=1;
                }
                idx+=1;
            } // j
        } // i

        map_x = Mat(x).clone();
        map_y = Mat(y).clone();
        map_x2 = Mat(x2).clone();
        map_y2 = Mat(y2).clone();
        map_erod2 = Mat(y_direct).clone();
    }

//  } /* namespace face */
} /* namespace cv */
