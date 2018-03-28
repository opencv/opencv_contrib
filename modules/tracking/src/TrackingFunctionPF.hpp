#include <cmath>
#include <opencv2/imgproc/imgproc_c.h>
#define CLIP(x,a,b) MIN(MAX((x),(a)),(b))
#define HIST_SIZE 50

namespace cv{

    class TrackingFunctionPF : public PFSolver::Function{
        public:
            TrackingFunctionPF(const Mat& chosenRect);
            void update(const Mat& image);
            int getDims() const CV_OVERRIDE { return 4; }
            double calc(const double* x) const CV_OVERRIDE;
            void correctParams(double* pt)const CV_OVERRIDE;
        private:
            Mat _image;
            static inline Rect rectFromRow(const double* row);
            const int _nh,_ns,_nv;
            class TrackingHistogram{
            public:
                TrackingHistogram(const Mat& img,int nh,int ns,int nv);
                double dist(const TrackingHistogram& hist)const;
            private:
                Mat_<double> HShist, Vhist;
            };
            TrackingHistogram _origHist;
			const TrackingFunctionPF & operator = (const TrackingFunctionPF &);
    };

    TrackingFunctionPF::TrackingHistogram::TrackingHistogram(const Mat& img,int nh,int ns,int nv){

        Mat hsv;
        img.convertTo(hsv,CV_32F,1.0/255.0);
        cvtColor(hsv,hsv,CV_BGR2HSV);

        HShist=Mat_<double>(nh,ns,0.0);
        Vhist=Mat_<double>(1,nv,0.0);

        for(int i=0;i<img.rows;i++){
            for(int j=0;j<img.cols;j++){
                const Vec3f& pt=hsv.at<Vec3f>(i,j);

                if(pt.val[1]>0.1 && pt.val[2]>0.2){
                    HShist(MIN(nh-1,(int)(nh*pt.val[0]/360.0)),MIN(ns-1,(int)(ns*pt.val[1])))++;
                }else{
                    Vhist(0,MIN(nv-1,(int)(nv*pt.val[2])))++;
                }
            }}

        double total=*(sum(HShist)+sum(Vhist)).val;
        HShist/=total;
        Vhist/=total;
    }
    double TrackingFunctionPF::TrackingHistogram::dist(const TrackingHistogram& hist)const{
        double res=1.0;

        for(int i=0;i<HShist.rows;i++){
            for(int j=0;j<HShist.cols;j++){
                res-=sqrt(HShist(i,j)*hist.HShist(i,j));
            }}
        for(int j=0;j<Vhist.cols;j++){
            res-=sqrt(Vhist(0,j)*hist.Vhist(0,j));
        }

        return sqrt(res);
    }
    double TrackingFunctionPF::calc(const double* x) const{
        Rect rect=rectFromRow(x);
        if(rect.area()==0){
            return 2.0;
        }
        return _origHist.dist(TrackingHistogram(_image(rect),_nh,_ns,_nv));
    }
    TrackingFunctionPF::TrackingFunctionPF(const Mat& chosenRect):_nh(HIST_SIZE),_ns(HIST_SIZE),_nv(HIST_SIZE),_origHist(chosenRect,_nh,_ns,_nv){
    }
    void TrackingFunctionPF::update(const Mat& image){
        _image=image;

        TrackingHistogram hist(image,_nh,_ns,_nv);
    }
    void TrackingFunctionPF::correctParams(double* pt)const{
        pt[0]=CLIP(pt[0],0.0,_image.cols+0.9);
        pt[1]=CLIP(pt[1],0.0,_image.rows+0.9);
        pt[2]=CLIP(pt[2],0.0,_image.cols+0.9);
        pt[3]=CLIP(pt[3],0.0,_image.rows+0.9);
        if(pt[0]>pt[2]){
            double tmp=pt[0];
            pt[0]=pt[2];
            pt[2]=tmp;
        }
        if(pt[1]>pt[3]){
            double tmp=pt[1];
            pt[1]=pt[3];
            pt[3]=tmp;
        }
    }
    Rect TrackingFunctionPF::rectFromRow(const double* row){
        return Rect(Point_<int>((int)row[0],(int)row[1]),Point_<int>((int)row[2],(int)row[3]));
    }
}
