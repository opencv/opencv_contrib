#include <kfusion/kinfu.hpp>
#include "safe_call.hpp"

#include <cuda.h>
#include <cstdio>
#include <iostream>

int kf::cuda::getCudaEnabledDeviceCount()
{
    int count;
    cudaError_t error = cudaGetDeviceCount( &count );

    if (error == cudaErrorInsufficientDriver)
        return -1;

    if (error == cudaErrorNoDevice)
        return 0;

    cudaSafeCall(error);
    return count;  
}

void kf::cuda::setDevice(int device)
{
    cudaSafeCall( cudaSetDevice( device ) );
}

std::string kf::cuda::getDeviceName(int device)
{
    cudaDeviceProp prop;
    cudaSafeCall( cudaGetDeviceProperties(&prop, device) );

    return prop.name;
}

bool kf::cuda::checkIfPreFermiGPU(int device)
{
  if (device < 0)
    cudaSafeCall( cudaGetDevice(&device) );

  cudaDeviceProp prop;
  cudaSafeCall( cudaGetDeviceProperties(&prop, device) );
  return prop.major < 2; // CC == 1.x
}

namespace 
{
    template <class T> inline void getCudaAttribute(T *attribute, CUdevice_attribute device_attribute, int device)
    {
        *attribute = T();
        CUresult error = cuDeviceGetAttribute( attribute, device_attribute, device );
        if( CUDA_SUCCESS == error ) 
            return;        

        printf("Driver API error = %04d\n", error);
        kfusion::cuda::error("driver API error", __FILE__, __LINE__);
    }

    inline int convertSMVer2Cores(int major, int minor)
    {
        // Defines for GPU Architecture types (using the SM version to determine the # of cores per SM
        typedef struct {
            int SM; // 0xMm (hexidecimal notation), M = SM Major version, and m = SM minor version
            int Cores;
        } SMtoCores;

        SMtoCores gpuArchCoresPerSM[] =  { { 0x10,  8 }, { 0x11,  8 }, { 0x12,  8 }, { 0x13,  8 }, { 0x20, 32 }, { 0x21, 48 }, {0x30, 192}, {0x35, 192}, {0x50, 128}, {0x52, 128}, {0x61, 128}, { -1, -1 }  };

        int index = 0;
        while (gpuArchCoresPerSM[index].SM != -1) 
        {
            if (gpuArchCoresPerSM[index].SM == ((major << 4) + minor) ) 
                return gpuArchCoresPerSM[index].Cores;
            index++;
        }
        printf("\nCan't determine number of cores. Unknown SM version %d.%d!\n", major, minor);
        return 0;
    }
}

void kf::cuda::printCudaDeviceInfo(int device)
{
    int count = getCudaEnabledDeviceCount();
    bool valid = (device >= 0) && (device < count);

    int beg = valid ? device   : 0;
    int end = valid ? device+1 : count;

    printf("*** CUDA Device Query (Runtime API) version (CUDART static linking) *** \n\n");
    printf("Device count: %d\n", count);

    int driverVersion = 0, runtimeVersion = 0;
    cudaSafeCall( cudaDriverGetVersion(&driverVersion) );
    cudaSafeCall( cudaRuntimeGetVersion(&runtimeVersion) );

    const char *computeMode[] = {
        "Default (multiple host threads can use ::cudaSetDevice() with device simultaneously)",
        "Exclusive (only one host thread in one process is able to use ::cudaSetDevice() with this device)",
        "Prohibited (no host thread can use ::cudaSetDevice() with this device)",
        "Exclusive Process (many threads in one process is able to use ::cudaSetDevice() with this device)",
        "Unknown",
        NULL
    };

    for(int dev = beg; dev < end; ++dev)
    {                
        cudaDeviceProp prop;
        cudaSafeCall( cudaGetDeviceProperties(&prop, dev) );

        int sm_cores = convertSMVer2Cores(prop.major, prop.minor);

        printf("\nDevice %d: \"%s\"\n", dev, prop.name);        
        printf("  CUDA Driver Version / Runtime Version          %d.%d / %d.%d\n", driverVersion/1000, driverVersion%100, runtimeVersion/1000, runtimeVersion%100);
        printf("  CUDA Capability Major/Minor version number:    %d.%d\n", prop.major, prop.minor);        
        printf("  Total amount of global memory:                 %.0f MBytes (%llu bytes)\n", (float)prop.totalGlobalMem/1048576.0f, (unsigned long long) prop.totalGlobalMem);            
        printf("  (%2d) Multiprocessors x (%2d) CUDA Cores/MP:     %d CUDA Cores\n", prop.multiProcessorCount, sm_cores, sm_cores * prop.multiProcessorCount);
        printf("  GPU Clock Speed:                               %.2f GHz\n", prop.clockRate * 1e-6f);

#if (CUDART_VERSION >= 4000)
        // This is not available in the CUDA Runtime API, so we make the necessary calls the driver API to support this for output
        int memoryClock, memBusWidth, L2CacheSize;
        getCudaAttribute<int>( &memoryClock, CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE, dev );        
        getCudaAttribute<int>( &memBusWidth, CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH, dev );                
        getCudaAttribute<int>( &L2CacheSize, CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE, dev );

        printf("  Memory Clock rate:                             %.2f Mhz\n", memoryClock * 1e-3f);
        printf("  Memory Bus Width:                              %d-bit\n", memBusWidth);
        if (L2CacheSize)
            printf("  L2 Cache Size:                                 %d bytes\n", L2CacheSize);
        
        printf("  Max Texture Dimension Size (x,y,z)             1D=(%d), 2D=(%d,%d), 3D=(%d,%d,%d)\n",
            prop.maxTexture1D, prop.maxTexture2D[0], prop.maxTexture2D[1],
            prop.maxTexture3D[0], prop.maxTexture3D[1], prop.maxTexture3D[2]);
        printf("  Max Layered Texture Size (dim) x layers        1D=(%d) x %d, 2D=(%d,%d) x %d\n",
            prop.maxTexture1DLayered[0], prop.maxTexture1DLayered[1],
            prop.maxTexture2DLayered[0], prop.maxTexture2DLayered[1], prop.maxTexture2DLayered[2]);
#endif
        printf("  Total amount of constant memory:               %u bytes\n", (int)prop.totalConstMem);
        printf("  Total amount of shared memory per block:       %u bytes\n", (int)prop.sharedMemPerBlock);
        printf("  Total number of registers available per block: %d\n", prop.regsPerBlock);
        printf("  Warp size:                                     %d\n", prop.warpSize);
        printf("  Maximum number of threads per block:           %d\n", prop.maxThreadsPerBlock);
        printf("  Maximum sizes of each dimension of a block:    %d x %d x %d\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
        printf("  Maximum sizes of each dimension of a grid:     %d x %d x %d\n", prop.maxGridSize[0], prop.maxGridSize[1],  prop.maxGridSize[2]);
        printf("  Maximum memory pitch:                          %u bytes\n", (int)prop.memPitch);
        printf("  Texture alignment:                             %u bytes\n", (int)prop.textureAlignment);

#if CUDART_VERSION >= 4000
        printf("  Concurrent copy and execution:                 %s with %d copy engine(s)\n", (prop.deviceOverlap ? "Yes" : "No"), prop.asyncEngineCount);
#else
        printf("  Concurrent copy and execution:                 %s\n", prop.deviceOverlap ? "Yes" : "No");
#endif
        printf("  Run time limit on kernels:                     %s\n", prop.kernelExecTimeoutEnabled ? "Yes" : "No");
        printf("  Integrated GPU sharing Host Memory:            %s\n", prop.integrated ? "Yes" : "No");
        printf("  Support host page-locked memory mapping:       %s\n", prop.canMapHostMemory ? "Yes" : "No");

        printf("  Concurrent kernel execution:                   %s\n", prop.concurrentKernels ? "Yes" : "No");
        printf("  Alignment requirement for Surfaces:            %s\n", prop.surfaceAlignment ? "Yes" : "No");
        printf("  Device has ECC support enabled:                %s\n", prop.ECCEnabled ? "Yes" : "No");
        printf("  Device is using TCC driver mode:               %s\n", prop.tccDriver ? "Yes" : "No");
#if CUDART_VERSION >= 4000
        printf("  Device supports Unified Addressing (UVA):      %s\n", prop.unifiedAddressing ? "Yes" : "No");
        printf("  Device PCI Bus ID / PCI location ID:           %d / %d\n", prop.pciBusID, prop.pciDeviceID );
#endif
        printf("  Compute Mode:\n");
        printf("      %s \n", computeMode[prop.computeMode]);
    }
    
    printf("\n");    
    printf("deviceQuery, CUDA Driver = CUDART");
    printf(", CUDA Driver Version  = %d.%d", driverVersion / 1000, driverVersion % 100);
    printf(", CUDA Runtime Version = %d.%d", runtimeVersion/1000, runtimeVersion%100);
    printf(", NumDevs = %d\n\n", count);                
    fflush(stdout);
}

void kf::cuda::printShortCudaDeviceInfo(int device)
{
    int count = getCudaEnabledDeviceCount();
    bool valid = (device >= 0) && (device < count);

    int beg = valid ? device   : 0;
    int end = valid ? device+1 : count;

    int driverVersion = 0, runtimeVersion = 0;
    cudaSafeCall( cudaDriverGetVersion(&driverVersion) );
    cudaSafeCall( cudaRuntimeGetVersion(&runtimeVersion) );

    for(int dev = beg; dev < end; ++dev)
    {                
        cudaDeviceProp prop;
        cudaSafeCall( cudaGetDeviceProperties(&prop, dev) );

        const char *arch_str = prop.major < 2 ? " (pre-Fermi)" : "";
        printf("Device %d:  \"%s\"  %.0fMb", dev, prop.name, (float)prop.totalGlobalMem/1048576.0f);                
        printf(", sm_%d%d%s, %d cores", prop.major, prop.minor, arch_str, convertSMVer2Cores(prop.major, prop.minor) * prop.multiProcessorCount);                
        printf(", Driver/Runtime ver.%d.%d/%d.%d\n", driverVersion/1000, driverVersion%100, runtimeVersion/1000, runtimeVersion%100);
    }
    fflush(stdout);
}

kf::SampledScopeTime::SampledScopeTime(double& time_ms) : time_ms_(time_ms)
{
    start = (double)cv::getTickCount();
}
kf::SampledScopeTime::~SampledScopeTime()
{
    static int i_ = 0;
    time_ms_ += getTime ();
    if (i_ % EACH == 0 && i_)
    {
        std::cout << "Average frame time = " << time_ms_ / EACH << "ms ( " << 1000.f * EACH / time_ms_ << "fps )" << std::endl;
        time_ms_ = 0.0;
    }
    ++i_;
}

double kf::SampledScopeTime::getTime()
{
    return ((double)cv::getTickCount() - start)*1000.0/cv::getTickFrequency();
}

kf::ScopeTime::ScopeTime(const char *name_) : name(name_)
{
    start = (double)cv::getTickCount();
}
kf::ScopeTime::~ScopeTime()
{
    double time_ms =  ((double)cv::getTickCount() - start)*1000.0/cv::getTickFrequency();
    std::cout << "Time(" << name << ") = " << time_ms << "ms" << std::endl;
}
