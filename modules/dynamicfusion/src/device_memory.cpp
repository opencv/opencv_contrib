#include <opencv2/dynamicfusion/cuda/device_memory.hpp>
#include <opencv2/dynamicfusion/cuda/safe_call.hpp>
#include <cassert>
#include <iostream>
#include <cstdlib>
using namespace cv;
void kfusion::cuda::error(const char *error_string, const char *file, const int line, const char *func)
{
    std::cout << "KinFu2 error: " << error_string << "\t" << file << ":" << line << std::endl;
    exit(0);
}

//////////////////////////    XADD    ///////////////////////////////

#ifdef __GNUC__
    
    #if __GNUC__*10 + __GNUC_MINOR__ >= 42

        #if !defined WIN32 && (defined __i486__ || defined __i586__ || defined __i686__ || defined __MMX__ || defined __SSE__  || defined __ppc__)
            #define CV_XADD __sync_fetch_and_add
        #else
            #include <ext/atomicity.h>
            #define CV_XADD __gnu_cxx::__exchange_and_add
        #endif
    #else
        #include <bits/atomicity.h>
        #if __GNUC__*10 + __GNUC_MINOR__ >= 34
            #define CV_XADD __gnu_cxx::__exchange_and_add
        #else
            #define CV_XADD __exchange_and_add
        #endif
  #endif
    
#elif defined WIN32 || defined _WIN32
    #include <intrin.h>
    #define CV_XADD(addr,delta) _InterlockedExchangeAdd((long volatile*)(addr), (delta))
#else

    template<typename _Tp> static inline _Tp CV_XADD(_Tp* addr, _Tp delta)
    { int tmp = *addr; *addr += delta; return tmp; }
    
#endif

////////////////////////    DeviceArray    /////////////////////////////
    
kfusion::cuda::DeviceMemory::DeviceMemory() : data_(0), sizeBytes_(0), refcount_(0) {}
kfusion::cuda::DeviceMemory::DeviceMemory(void *ptr_arg, size_t sizeBytes_arg) : data_(ptr_arg), sizeBytes_(sizeBytes_arg), refcount_(0){}
kfusion::cuda::DeviceMemory::DeviceMemory(size_t sizeBtes_arg)  : data_(0), sizeBytes_(0), refcount_(0) { create(sizeBtes_arg); }
kfusion::cuda::DeviceMemory::~DeviceMemory() { release(); }

kfusion::cuda::DeviceMemory::DeviceMemory(const DeviceMemory& other_arg)
    : data_(other_arg.data_), sizeBytes_(other_arg.sizeBytes_), refcount_(other_arg.refcount_)
{
    if( refcount_ )
        CV_XADD(refcount_, 1);
}

kfusion::cuda::DeviceMemory& kfusion::cuda::DeviceMemory::operator = (const kfusion::cuda::DeviceMemory& other_arg)
{
    if( this != &other_arg )
    {
        if( other_arg.refcount_ )
            CV_XADD(other_arg.refcount_, 1);
        release();
        
        data_      = other_arg.data_;
        sizeBytes_ = other_arg.sizeBytes_;                
        refcount_  = other_arg.refcount_;
    }
    return *this;
}

void kfusion::cuda::DeviceMemory::create(size_t sizeBytes_arg)
{
    if (sizeBytes_arg == sizeBytes_)
        return;
            
    if( sizeBytes_arg > 0)
    {        
        if( data_ )
            release();

        sizeBytes_ = sizeBytes_arg;
                        
        cudaSafeCall( cudaMalloc(&data_, sizeBytes_) );
        
        //refcount_ = (int*)cv::fastMalloc(sizeof(*refcount_));
        refcount_ = new int;
        *refcount_ = 1;
    }
}

void kfusion::cuda::DeviceMemory::copyTo(DeviceMemory& other) const
{
    if (empty())
        other.release();
    else
    {    
        other.create(sizeBytes_);    
        cudaSafeCall( cudaMemcpy(other.data_, data_, sizeBytes_, cudaMemcpyDeviceToDevice) );
    }
}

void kfusion::cuda::DeviceMemory::release()
{
    if( refcount_ && CV_XADD(refcount_, -1) == 1 )
    {
        //cv::fastFree(refcount);
        delete refcount_;
        cudaSafeCall( cudaFree(data_) );
    }
    data_ = 0;
    sizeBytes_ = 0;
    refcount_ = 0;
}

void kfusion::cuda::DeviceMemory::upload(const void *host_ptr_arg, size_t sizeBytes_arg)
{
    create(sizeBytes_arg);
    cudaSafeCall( cudaMemcpy(data_, host_ptr_arg, sizeBytes_, cudaMemcpyHostToDevice) );
}

void kfusion::cuda::DeviceMemory::download(void *host_ptr_arg) const
{    
    cudaSafeCall( cudaMemcpy(host_ptr_arg, data_, sizeBytes_, cudaMemcpyDeviceToHost) );
}          

void kfusion::cuda::DeviceMemory::swap(DeviceMemory& other_arg)
{
    std::swap(data_, other_arg.data_);
    std::swap(sizeBytes_, other_arg.sizeBytes_);
    std::swap(refcount_, other_arg.refcount_);
}

bool kfusion::cuda::DeviceMemory::empty() const { return !data_; }
size_t kfusion::cuda::DeviceMemory::sizeBytes() const { return sizeBytes_; }


////////////////////////    DeviceArray2D    /////////////////////////////

kfusion::cuda::DeviceMemory2D::DeviceMemory2D() : data_(0), step_(0), colsBytes_(0), rows_(0), refcount_(0) {}

kfusion::cuda::DeviceMemory2D::DeviceMemory2D(int rows_arg, int colsBytes_arg)
    : data_(0), step_(0), colsBytes_(0), rows_(0), refcount_(0)
{ 
    create(rows_arg, colsBytes_arg); 
}

kfusion::cuda::DeviceMemory2D::DeviceMemory2D(int rows_arg, int colsBytes_arg, void *data_arg, size_t step_arg)
    :  data_(data_arg), step_(step_arg), colsBytes_(colsBytes_arg), rows_(rows_arg), refcount_(0) {}

kfusion::cuda::DeviceMemory2D::~DeviceMemory2D() { release(); }


kfusion::cuda::DeviceMemory2D::DeviceMemory2D(const DeviceMemory2D& other_arg) :
    data_(other_arg.data_), step_(other_arg.step_), colsBytes_(other_arg.colsBytes_), rows_(other_arg.rows_), refcount_(other_arg.refcount_)
{
    if( refcount_ )
        CV_XADD(refcount_, 1);
}

kfusion::cuda::DeviceMemory2D& kfusion::cuda::DeviceMemory2D::operator = (const kfusion::cuda::DeviceMemory2D& other_arg)
{
    if( this != &other_arg )
    {
        if( other_arg.refcount_ )
            CV_XADD(other_arg.refcount_, 1);
        release();
        
        colsBytes_ = other_arg.colsBytes_;
        rows_ = other_arg.rows_;
        data_ = other_arg.data_;
        step_ = other_arg.step_;
                
        refcount_ = other_arg.refcount_;
    }
    return *this;
}

void kfusion::cuda::DeviceMemory2D::create(int rows_arg, int colsBytes_arg)
{
    if (colsBytes_ == colsBytes_arg && rows_ == rows_arg)
        return;
            
    if( rows_arg > 0 && colsBytes_arg > 0)
    {        
        if( data_ )
            release();
              
        colsBytes_ = colsBytes_arg;
        rows_ = rows_arg;
                        
        cudaSafeCall( cudaMallocPitch( (void**)&data_, &step_, colsBytes_, rows_) );        

        //refcount = (int*)cv::fastMalloc(sizeof(*refcount));
        refcount_ = new int;
        *refcount_ = 1;
    }
}

void kfusion::cuda::DeviceMemory2D::release()
{
    if( refcount_ && CV_XADD(refcount_, -1) == 1 )
    {
        //cv::fastFree(refcount);
        delete refcount_;
        cudaSafeCall( cudaFree(data_) );
    }

    colsBytes_ = 0;
    rows_ = 0;    
    data_ = 0;    
    step_ = 0;
    refcount_ = 0;
}

void kfusion::cuda::DeviceMemory2D::copyTo(DeviceMemory2D& other) const
{
    if (empty())
        other.release();
    else
    {
        other.create(rows_, colsBytes_);    
        cudaSafeCall( cudaMemcpy2D(other.data_, other.step_, data_, step_, colsBytes_, rows_, cudaMemcpyDeviceToDevice) );
    }
}

void kfusion::cuda::DeviceMemory2D::upload(const void *host_ptr_arg, size_t host_step_arg, int rows_arg, int colsBytes_arg)
{
    create(rows_arg, colsBytes_arg);
    cudaSafeCall( cudaMemcpy2D(data_, step_, host_ptr_arg, host_step_arg, colsBytes_, rows_, cudaMemcpyHostToDevice) );        
}

void kfusion::cuda::DeviceMemory2D::download(void *host_ptr_arg, size_t host_step_arg) const
{    
    cudaSafeCall( cudaMemcpy2D(host_ptr_arg, host_step_arg, data_, step_, colsBytes_, rows_, cudaMemcpyDeviceToHost) );
}      

void kfusion::cuda::DeviceMemory2D::swap(DeviceMemory2D& other_arg)
{    
    std::swap(data_, other_arg.data_);
    std::swap(step_, other_arg.step_);

    std::swap(colsBytes_, other_arg.colsBytes_);
    std::swap(rows_, other_arg.rows_);
    std::swap(refcount_, other_arg.refcount_);                 
}

bool kfusion::cuda::DeviceMemory2D::empty() const { return !data_; }
int kfusion::cuda::DeviceMemory2D::colsBytes() const { return colsBytes_; }
int kfusion::cuda::DeviceMemory2D::rows() const { return rows_; }
size_t kfusion::cuda::DeviceMemory2D::step() const { return step_; }
