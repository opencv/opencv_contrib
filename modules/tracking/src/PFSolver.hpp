#include "opencv2/core.hpp"
#include "opencv2/core/core_c.h"
#include <algorithm>
#include <typeinfo>
#include <cmath>
#define WEIGHTED

namespace cv{

    //!particle filtering class
    class PFSolver : public MinProblemSolver{
    public:
        class Function : public MinProblemSolver::Function
        {
        public:
            //!if parameters have no sense due to some reason (e.g. lie outside of function domain), this function "corrects" them,
            //!that is brings to the function domain
            virtual void correctParams(double* /*optParams*/)const{}
            //!is used when there is a dependence on the number of iterations done in calc(), note that levels are counted starting from 1
            virtual void setLevel(int /*level*/, int /*levelsNum*/){}
        };
        PFSolver();
        void getOptParam(OutputArray params)const;
        int iteration();
        double minimize(InputOutputArray x) CV_OVERRIDE;

        void setParticlesNum(int num);
        int getParticlesNum();
        void setAlpha(double AlphaM);
        double getAlpha();
        void getParamsSTD(OutputArray std)const;
        void setParamsSTD(InputArray std);

        Ptr<MinProblemSolver::Function> getFunction() const CV_OVERRIDE;
        void setFunction(const Ptr<MinProblemSolver::Function>& f) CV_OVERRIDE;
        TermCriteria getTermCriteria() const CV_OVERRIDE;
        void setTermCriteria(const TermCriteria& termcrit) CV_OVERRIDE;
    private:
        Mat_<double> _std,_particles,_logweight;
        Ptr<MinProblemSolver::Function> _Function;
        PFSolver::Function* _real_function;
        TermCriteria _termcrit;
        int _maxItNum,_iter,_particlesNum;
        double _alpha;
        inline void normalize(Mat_<double>& row);
        RNG rng;
    };

    CV_EXPORTS_W Ptr<PFSolver> createPFSolver(const Ptr<MinProblemSolver::Function>& f=Ptr<MinProblemSolver::Function>(),InputArray std=Mat(),
            TermCriteria termcrit=TermCriteria(TermCriteria::MAX_ITER,5,0.0),int particlesNum=100,double alpha=0.6);

    PFSolver::PFSolver(){
        _Function=Ptr<MinProblemSolver::Function>();
        _real_function=NULL;
        _std=Mat_<double>();
        rng=RNG(getTickCount());
    }
    void PFSolver::getOptParam(OutputArray params)const{
        params.create(1,_std.rows,CV_64FC1);
        Mat mat(1,_std.rows,CV_64FC1);
#ifdef WEIGHTED
        mat.setTo(0.0);
        for(int i=0;i<_particles.rows;i++){
            mat+=_particles.row(i)/exp(-_logweight(0,i));
        }
        _real_function->correctParams((double*)mat.data);
        mat.copyTo(params);
#else
        params.create(1,_std.rows,CV_64FC1);
        Mat optimus=_particles.row(std::max_element(_logweight.begin(),_logweight.end())-_logweight.begin());
        _real_function->correctParams(optimus.data);
        optimus.copyTo(params);
#endif
    }
    int PFSolver::iteration(){
        if(_iter>=_maxItNum){
            return _maxItNum+1;
        }

        _real_function->setLevel(_iter+1,_maxItNum);

        //perturb
        for(int j=0;j<_particles.cols;j++){
            double sigma=_std(0,j);
            for(int i=0;i<_particles.rows;i++){
                    _particles(i,j)+=rng.gaussian(sigma);
            }
        }

        //measure
        for(int i=0;i<_particles.rows;i++){
            _real_function->correctParams((double*)_particles.row(i).data);
            _logweight(0,i)=-(_real_function->calc((double*)_particles.row(i).data));
        }
        //normalize
        normalize(_logweight);
        //replicate
        Mat_<double> new_particles(_particlesNum,_std.cols);
        int num_particles=0;
        for(int i=0;i<_particles.rows;i++){
            int num_replicons=cvFloor(new_particles.rows/exp(-_logweight(0,i)));
            for(int j=0;j<num_replicons;j++,num_particles++){
                _particles.row(i).copyTo(new_particles.row(num_particles));
            }
        }
        //Mat_<double> maxrow=_particles.row(std::max_element(_logweight.begin(),_logweight.end())-_logweight.begin());
        double max_element;
        minMaxLoc(_logweight, 0, &max_element);
        Mat_<double> maxrow=_particles.row((int)max_element);
        for(;num_particles<new_particles.rows;num_particles++){
                maxrow.copyTo(new_particles.row(num_particles));
        }

        if(_particles.rows!=new_particles.rows){
            _particles=new_particles;
        }else{
            new_particles.copyTo(_particles);
        }
        _std=_std*_alpha;
        _iter++;
        return _iter;
    }
    double PFSolver::minimize(InputOutputArray x){
        CV_Assert(_Function.empty()==false);
        CV_Assert(_std.rows==1 && _std.cols>0);
        Mat mat_x=x.getMat();
        CV_Assert(mat_x.type()==CV_64FC1 && MIN(mat_x.rows,mat_x.cols)==1 && MAX(mat_x.rows,mat_x.cols)==_std.cols);

        _iter=0;
        _particles=Mat_<double>(_particlesNum,_std.cols);
        if(mat_x.rows>1){
            mat_x=mat_x.t();
        }
        for(int i=0;i<_particles.rows;i++){
            mat_x.copyTo(_particles.row(i));
        }

        _logweight.create(1,_particles.rows);
        _logweight.setTo(-log((double)_particles.rows));
        return 0.0;
    }

    void PFSolver::setParticlesNum(int num){
        CV_Assert(num>0);
        _particlesNum=num;
    }
    int PFSolver::getParticlesNum(){
        return _particlesNum;
    }
    void PFSolver::setAlpha(double AlphaM){
        CV_Assert(0<AlphaM && AlphaM<=1);
        _alpha=AlphaM;
    }
    double PFSolver::getAlpha(){
        return _alpha;
    }
    Ptr<MinProblemSolver::Function> PFSolver::getFunction() const{
        return _Function;
    }
    void PFSolver::setFunction(const Ptr<MinProblemSolver::Function>& f){
        CV_Assert(f.empty()==false);

        Ptr<MinProblemSolver::Function> non_const_f(f);
        MinProblemSolver::Function* f_ptr=static_cast<MinProblemSolver::Function*>(non_const_f);

        PFSolver::Function *pff=dynamic_cast<PFSolver::Function*>(f_ptr);
        CV_Assert(pff!=NULL);
        _Function=f;
        _real_function=pff;
    }
    TermCriteria PFSolver::getTermCriteria() const{
        return TermCriteria(TermCriteria::MAX_ITER,_maxItNum,0.0);
    }
    void PFSolver::setTermCriteria(const TermCriteria& termcrit){
        CV_Assert(termcrit.type==TermCriteria::MAX_ITER && termcrit.maxCount>0);
        _maxItNum=termcrit.maxCount;
    }
    void PFSolver::getParamsSTD(OutputArray std)const{
        std.create(1,_std.cols,CV_64FC1);
        _std.copyTo(std);
    }
    void PFSolver::setParamsSTD(InputArray std){
        Mat m=std.getMat();
        CV_Assert(MIN(m.cols,m.rows)==1 && m.type()==CV_64FC1);
        int ndim=MAX(m.cols,m.rows);
        if(ndim!=_std.cols){
            _std=Mat_<double>(1,ndim);
        }
        if(m.rows==1){
            m.copyTo(_std);
        }else{
            Mat std_t=Mat_<double>(ndim,1,(double*)_std.data);
            m.copyTo(std_t);
        }
    }

    Ptr<PFSolver> createPFSolver(const Ptr<MinProblemSolver::Function>& f,InputArray std,TermCriteria termcrit,int particlesNum,double alpha){
            Ptr<PFSolver> ptr(new PFSolver());

            if(f.empty()==false){
                ptr->setFunction(f);
            }
            Mat mystd=std.getMat();
            if(mystd.cols!=0 || mystd.rows!=0){
                ptr->setParamsSTD(std);
            }
            ptr->setTermCriteria(termcrit);
            ptr->setParticlesNum(particlesNum);
            ptr->setAlpha(alpha);
            return ptr;
    }
    void PFSolver::normalize(Mat_<double>& row){
        double logsum=0.0;
        //double max=*(std::max_element(row.begin(),row.end()));
        double max;
        minMaxLoc(row, 0, &max);
        row-=max;
        for(int i=0;i<row.cols;i++){
            logsum+=exp(row(0,i));
        }
        logsum=log(logsum);
        row-=logsum;
    }
}
