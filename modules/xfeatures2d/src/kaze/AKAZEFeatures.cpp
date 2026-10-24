/**
 * @file AKAZEFeatures.cpp
 * @brief Main class for detecting and describing binary features in an
 * accelerated nonlinear scale space
 * @date Sep 15, 2013
 * @author Pablo F. Alcantarilla, Jesus Nuevo
 */

#include "../precomp.hpp"
#include "AKAZEFeatures.h"
#include "fed.h"
#include "nldiffusion_functions.h"
#include "utils.h"
#include "opencl_kernels_xfeatures2d.hpp"
#include "opencv2/core/hal/intrin.hpp"
#include "opencv2/core/utils/tls.hpp"

#include <iostream>

// Namespaces
namespace cv
{
namespace xfeatures2d
{
using namespace std;

/**
 * @brief Runs body(row_begin, row_end) over horizontal stripes of an image in parallel
 * @note Filters applied to a row range of a Mat read the real neighbouring rows
 * (no BORDER_ISOLATED), so per-stripe filtering gives exactly the same result as
 * filtering the whole image at once.
 */
template <typename Body>
static inline void parallel_for_rows(int rows, const Body& body, int min_stripe = 32)
{
  const int nstripes = std::max(1, std::min(rows / min_stripe, getNumThreads() * 4));
  parallel_for_(Range(0, rows), [&](const Range& r) { body(r.start, r.end); }, nstripes);
}

/**
 * @brief Returns a CV_32F header of size sz on the memory of buf, growing buf if needed
 * @note Reusing big temporaries matters: freshly allocated images are mmap'ed and
 * page-faulted again on every allocation.
 */
static inline Mat scratchMat(Mat& buf, Size sz)
{
  const int n = sz.area();
  if ((int)buf.total() < n)
    buf.create(1, n, CV_32F);
  return Mat(sz, CV_32F, buf.ptr<float>());
}

/// Per-thread temporaries for the stripe-parallel stages
struct ScratchBuffers { Mat buf[4]; };

/* ************************************************************************* */
/**
 * @brief AKAZEFeatures constructor with input options
 * @param options AKAZEFeatures configuration options
 * @note This constructor allocates memory for the nonlinear scale space
 */
AKAZEFeatures::AKAZEFeatures(const AKAZEOptions& options) : options_(options) {

  ncycles_ = 0;
  reordering_ = true;

  if (options_.descriptor_size > 0 && options_.descriptor >= AKAZE::DESCRIPTOR_MLDB_UPRIGHT) {
    generateDescriptorSubsample(descriptorSamples_, descriptorBits_, options_.descriptor_size,
                                options_.descriptor_pattern_size, options_.descriptor_channels);
  }

  Allocate_Memory_Evolution();
}

/* ************************************************************************* */
/**
 * @brief This method allocates the memory for the nonlinear diffusion evolution
 */
void AKAZEFeatures::Allocate_Memory_Evolution(void) {
  CV_INSTRUMENT_REGION();

  float rfactor = 0.0f;
  int level_height = 0, level_width = 0;

  // maximum size of the area for the descriptor computation
  float smax = 0.0;
  if (options_.descriptor == AKAZE::DESCRIPTOR_MLDB_UPRIGHT || options_.descriptor == AKAZE::DESCRIPTOR_MLDB) {
    smax = 10.0f*sqrtf(2.0f);
  }
  else if (options_.descriptor == AKAZE::DESCRIPTOR_KAZE_UPRIGHT || options_.descriptor == AKAZE::DESCRIPTOR_KAZE) {
    smax = 12.0f*sqrtf(2.0f);
  }

  // Allocate the dimension of the matrices for the evolution
  for (int i = 0, power = 1; i <= options_.omax - 1; i++, power *= 2) {
    rfactor = 1.0f / power;
    level_height = (int)(options_.img_height*rfactor);
    level_width = (int)(options_.img_width*rfactor);

    // Smallest possible octave and allow one scale if the image is small
    if ((level_width < 80 || level_height < 40) && i != 0) {
      options_.omax = i;
      break;
    }

    for (int j = 0; j < options_.nsublevels; j++) {
      MEvolution step;
      step.size = Size(level_width, level_height);
      step.esigma = options_.soffset*pow(2.f, (float)(j) / (float)(options_.nsublevels) + i);
      step.sigma_size = cvRound(step.esigma * options_.derivative_factor / power);  // In fact sigma_size only depends on j
      step.etime = 0.5f * (step.esigma * step.esigma);
      step.octave = i;
      step.sublevel = j;
      step.octave_ratio = (float)power;
      step.border = cvRound(smax * step.sigma_size) + 1;

      evolution_.push_back(step);
    }
  }

  // Allocate memory for the number of cycles and time steps
  for (size_t i = 1; i < evolution_.size(); i++) {
    int naux = 0;
    vector<float> tau;
    float ttime = 0.0f;
    ttime = evolution_[i].etime - evolution_[i - 1].etime;
    naux = fed_tau_by_process_time(ttime, 1, 0.25f, reordering_, tau);
    nsteps_.push_back(naux);
    tsteps_.push_back(tau);
    ncycles_++;
  }
}

/* ************************************************************************* */
/**
 * @brief Computes kernel size for Gaussian smoothing if the image
 * @param sigma Kernel standard deviation
 * @returns kernel size
 */
static inline int getGaussianKernelSize(float sigma) {
  // Compute an appropriate kernel size according to the specified sigma
  int ksize = (int)cvCeil(2.0f*(1.0f + (sigma - 0.8f) / (0.3f)));
  ksize |= 1; // kernel should be odd
  return ksize;
}

/* ************************************************************************* */
/**
* @brief This function computes a scalar non-linear diffusion step
* @param Lt Base image in the evolution
* @param Lf Conductivity image
* @param Lstep Output image that gives the difference between the current
* Ld and the next Ld being evolved
* @param row_begin row where to start
* @param row_end last row to fill exclusive. the range is [row_begin, row_end).
* @note Forward Euler Scheme 3x3 stencil
* The function c is a scalar value that depends on the gradient norm
* dL_by_ds = d(c dL_by_dx)_by_dx + d(c dL_by_dy)_by_dy
*/
static inline void
nld_step_scalar_one_lane(const Mat& Lt, const Mat& Lf, Mat& Lstep, float step_size, int row_begin, int row_end)
{
  CV_INSTRUMENT_REGION();
  /* The labeling scheme for this five star stencil:
   [    a    ]
   [ -1 c +1 ]
   [    b    ]
   */

  Lstep.create(Lt.size(), Lt.type());
  const int cols = Lt.cols - 2;
  int row = row_begin;

  const float *lt_a, *lt_c, *lt_b;
  const float *lf_a, *lf_c, *lf_b;
  float *dst;
  float step_r = 0.f;

  // Process the top row
  if (row == 0) {
    lt_c = Lt.ptr<float>(0) + 1;  /* Skip the left-most column by +1 */
    lf_c = Lf.ptr<float>(0) + 1;
    lt_b = Lt.ptr<float>(1) + 1;
    lf_b = Lf.ptr<float>(1) + 1;

    // fill the corner to prevent uninitialized values
    dst = Lstep.ptr<float>(0);
    dst[0] = 0.0f;
    ++dst;

    for (int j = 0; j < cols; j++) {
      step_r = (lf_c[j] + lf_c[j + 1])*(lt_c[j + 1] - lt_c[j]) +
               (lf_c[j] + lf_c[j - 1])*(lt_c[j - 1] - lt_c[j]) +
               (lf_c[j] + lf_b[j    ])*(lt_b[j    ] - lt_c[j]);
      dst[j] = step_r * step_size;
    }

    // fill the corner to prevent uninitialized values
    dst[cols] = 0.0f;
    ++row;
  }

  // Process the middle rows
  int middle_end = std::min(Lt.rows - 1, row_end);
  for (; row < middle_end; ++row)
  {
    lt_a = Lt.ptr<float>(row - 1);
    lf_a = Lf.ptr<float>(row - 1);
    lt_c = Lt.ptr<float>(row    );
    lf_c = Lf.ptr<float>(row    );
    lt_b = Lt.ptr<float>(row + 1);
    lf_b = Lf.ptr<float>(row + 1);
    dst = Lstep.ptr<float>(row);

    // The left-most column
    step_r = (lf_c[0] + lf_c[1])*(lt_c[1] - lt_c[0]) +
             (lf_c[0] + lf_b[0])*(lt_b[0] - lt_c[0]) +
             (lf_c[0] + lf_a[0])*(lt_a[0] - lt_c[0]);
    dst[0] = step_r * step_size;

    lt_a++; lt_c++; lt_b++;
    lf_a++; lf_c++; lf_b++;
    dst++;

    // The middle columns
    for (int j = 0; j < cols; j++)
    {
      step_r = (lf_c[j] + lf_c[j + 1])*(lt_c[j + 1] - lt_c[j]) +
               (lf_c[j] + lf_c[j - 1])*(lt_c[j - 1] - lt_c[j]) +
               (lf_c[j] + lf_b[j    ])*(lt_b[j    ] - lt_c[j]) +
               (lf_c[j] + lf_a[j    ])*(lt_a[j    ] - lt_c[j]);
      dst[j] = step_r * step_size;
    }

    // The right-most column
    step_r = (lf_c[cols] + lf_c[cols - 1])*(lt_c[cols - 1] - lt_c[cols]) +
             (lf_c[cols] + lf_b[cols    ])*(lt_b[cols    ] - lt_c[cols]) +
             (lf_c[cols] + lf_a[cols    ])*(lt_a[cols    ] - lt_c[cols]);
    dst[cols] = step_r * step_size;
  }

  // Process the bottom row (row == Lt.rows - 1)
  if (row_end == Lt.rows) {
    lt_a = Lt.ptr<float>(row - 1) + 1;  /* Skip the left-most column by +1 */
    lf_a = Lf.ptr<float>(row - 1) + 1;
    lt_c = Lt.ptr<float>(row    ) + 1;
    lf_c = Lf.ptr<float>(row    ) + 1;

    // fill the corner to prevent uninitialized values
    dst = Lstep.ptr<float>(row);
    dst[0] = 0.0f;
    ++dst;

    for (int j = 0; j < cols; j++) {
      step_r = (lf_c[j] + lf_c[j + 1])*(lt_c[j + 1] - lt_c[j]) +
               (lf_c[j] + lf_c[j - 1])*(lt_c[j - 1] - lt_c[j]) +
               (lf_c[j] + lf_a[j    ])*(lt_a[j    ] - lt_c[j]);
      dst[j] = step_r * step_size;
    }

    // fill the corner to prevent uninitialized values
    dst[cols] = 0.0f;
  }
}

class NonLinearScalarDiffusionStep : public ParallelLoopBody
{
public:
  NonLinearScalarDiffusionStep(const Mat& Lt, const Mat& Lf, Mat& Lstep, float step_size)
    : Lt_(&Lt), Lf_(&Lf), Lstep_(&Lstep), step_size_(step_size)
  {}

  void operator()(const Range& range) const CV_OVERRIDE
  {
    nld_step_scalar_one_lane(*Lt_, *Lf_, *Lstep_, step_size_, range.start, range.end);
  }

private:
  const Mat* Lt_;
  const Mat* Lf_;
  Mat* Lstep_;
  float step_size_;
};

#ifdef HAVE_OPENCL
static inline bool
ocl_non_linear_diffusion_step(InputArray Lt_, InputArray Lf_, OutputArray Lstep_, float step_size)
{
  if(!Lt_.isContinuous())
    return false;

  UMat Lt = Lt_.getUMat();
  UMat Lf = Lf_.getUMat();
  UMat Lstep = Lstep_.getUMat();

  size_t globalSize[] = {(size_t)Lt.cols, (size_t)Lt.rows};

  ocl::Kernel ker("AKAZE_nld_step_scalar", ocl::xfeatures2d::akaze_oclsrc);
  if( ker.empty() )
    return false;

  return ker.args(
    ocl::KernelArg::ReadOnly(Lt),
    ocl::KernelArg::PtrReadOnly(Lf),
    ocl::KernelArg::PtrWriteOnly(Lstep),
    step_size).run(2, globalSize, 0, true);
}
#endif // HAVE_OPENCL

static inline void
non_linear_diffusion_step(InputArray Lt_, InputArray Lf_, OutputArray Lstep_, float step_size)
{
  CV_INSTRUMENT_REGION();

  Lstep_.create(Lt_.size(), Lt_.type());

  CV_OCL_RUN(Lt_.isUMat() && Lf_.isUMat() && Lstep_.isUMat(),
    ocl_non_linear_diffusion_step(Lt_, Lf_, Lstep_, step_size));

  Mat Lt = Lt_.getMat();
  Mat Lf = Lf_.getMat();
  Mat Lstep = Lstep_.getMat();
  parallel_for_(Range(0, Lt.rows), NonLinearScalarDiffusionStep(Lt, Lf, Lstep, step_size));
}

/**
 * @brief Fused FED step for one row: dst = Lt + step_size * Lstep
 * @details Same arithmetic as nld_step_scalar_one_lane() followed by add(Lt, Lstep),
 * but in a single pass and without the intermediate Lstep image.
 * lt_a/lf_a (row above) is NULL for the top row and lt_b/lf_b (row below) is NULL
 * for the bottom row. dst must not alias the input rows.
 */
static inline void
nld_step_fused_row(const float* lt_a, const float* lt_c, const float* lt_b,
                   const float* lf_a, const float* lf_c, const float* lf_b,
                   float* dst, int width, float step_size)
{
  const int cols = width - 2;
  float step_r = 0.f;

  if (!lt_a || !lt_b) {
    // top or bottom row: the corners are not updated
    const float *lt_n = lt_a ? lt_a : lt_b, *lf_n = lf_a ? lf_a : lf_b;
    dst[0] = lt_c[0] + 0.0f;
    for (int j = 1; j <= cols; j++) {
      step_r = (lf_c[j] + lf_c[j + 1])*(lt_c[j + 1] - lt_c[j]) +
               (lf_c[j] + lf_c[j - 1])*(lt_c[j - 1] - lt_c[j]) +
               (lf_c[j] + lf_n[j    ])*(lt_n[j    ] - lt_c[j]);
      dst[j] = lt_c[j] + step_r * step_size;
    }
    dst[cols + 1] = lt_c[cols + 1] + 0.0f;
    return;
  }

  // The left-most column
  step_r = (lf_c[0] + lf_c[1])*(lt_c[1] - lt_c[0]) +
           (lf_c[0] + lf_b[0])*(lt_b[0] - lt_c[0]) +
           (lf_c[0] + lf_a[0])*(lt_a[0] - lt_c[0]);
  dst[0] = lt_c[0] + step_r * step_size;

  // The middle columns
  int j = 1;
#if (CV_SIMD || CV_SIMD_SCALABLE)
  const int vlanes = VTraits<v_float32>::vlanes();
  const v_float32 vstep = vx_setall_f32(step_size);
  for (; j <= cols + 1 - vlanes; j += vlanes)
  {
    v_float32 c_t = vx_load(lt_c + j), c_f = vx_load(lf_c + j);
    v_float32 r = v_mul(v_add(c_f, vx_load(lf_c + j + 1)), v_sub(vx_load(lt_c + j + 1), c_t));
    r = v_add(r, v_mul(v_add(c_f, vx_load(lf_c + j - 1)), v_sub(vx_load(lt_c + j - 1), c_t)));
    r = v_add(r, v_mul(v_add(c_f, vx_load(lf_b + j)), v_sub(vx_load(lt_b + j), c_t)));
    r = v_add(r, v_mul(v_add(c_f, vx_load(lf_a + j)), v_sub(vx_load(lt_a + j), c_t)));
    v_store(dst + j, v_add(c_t, v_mul(r, vstep)));
  }
#endif
  for (; j <= cols; j++)
  {
    step_r = (lf_c[j] + lf_c[j + 1])*(lt_c[j + 1] - lt_c[j]) +
             (lf_c[j] + lf_c[j - 1])*(lt_c[j - 1] - lt_c[j]) +
             (lf_c[j] + lf_b[j    ])*(lt_b[j    ] - lt_c[j]) +
             (lf_c[j] + lf_a[j    ])*(lt_a[j    ] - lt_c[j]);
    dst[j] = lt_c[j] + step_r * step_size;
  }

  // The right-most column
  const int r = cols + 1;
  step_r = (lf_c[r] + lf_c[r - 1])*(lt_c[r - 1] - lt_c[r]) +
           (lf_c[r] + lf_b[r    ])*(lt_b[r    ] - lt_c[r]) +
           (lf_c[r] + lf_a[r    ])*(lt_a[r    ] - lt_c[r]);
  dst[r] = lt_c[r] + step_r * step_size;
}

/**
 * @brief Performs all the FED steps of one evolution level: Lout = FED(Lstart)
 * @details Big images are processed in row stripes; each stripe runs all the steps
 * while it is in cache (temporal blocking), recomputing a halo of one row per
 * remaining step. Small images (already in cache, but with many steps) are processed
 * step by step, each step in parallel. Every pixel is computed exactly as in the
 * original step-by-step scheme.
 */
static void
fed_cycle(const Mat& Lstart, const Mat& Lflow, const std::vector<float>& tsteps, Mat& Lout,
          TLSData<ScratchBuffers>& scratch, Mat* tmp_buf)
{
  CV_INSTRUMENT_REGION();

  const int rows = Lstart.rows, cols = Lstart.cols;
  const int nsteps = (int)tsteps.size();
  CV_Assert(nsteps > 0 && rows >= 2 && cols >= 2);
  Lout.create(Lstart.size(), CV_32F);

  // Stripe height: the halo costs nsteps*(nsteps-1) extra rows per stripe, keep it below ~50%
  const int min_rows = std::max(32, 2 * nsteps * (nsteps - 1));
  const int nstripes = std::max(1, std::min(rows / min_rows, std::max(getNumThreads() * 4, rows / 64)));
  const size_t level_bytes = (size_t)rows * cols * sizeof(float);
  if (level_bytes < ((size_t)1 << 21) || (nstripes < getNumThreads() && nstripes < 8)) {
    Mat src = Lstart;
    for (int s = 0; s < nsteps; s++) {
      const float step_size = tsteps[s] * 0.5f;
      Mat dst = s == nsteps - 1 ? Lout : scratchMat(tmp_buf[s & 1], Lstart.size());
      parallel_for_rows(rows, [&](int y0, int y1) {
        for (int y = y0; y < y1; y++)
          nld_step_fused_row(y > 0 ? src.ptr<float>(y - 1) : NULL, src.ptr<float>(y),
                             y < rows - 1 ? src.ptr<float>(y + 1) : NULL,
                             y > 0 ? Lflow.ptr<float>(y - 1) : NULL, Lflow.ptr<float>(y),
                             y < rows - 1 ? Lflow.ptr<float>(y + 1) : NULL,
                             dst.ptr<float>(y), cols, step_size);
      }, 8);
      src = dst;
    }
    return;
  }

  parallel_for_(Range(0, nstripes), [&](const Range& range) {
    ScratchBuffers *sb = scratch.get();
    for (int t = range.start; t < range.end; t++) {
      const int r0 = t * rows / nstripes, r1 = (t + 1) * rows / nstripes;
      // rows [b0, b1) of the intermediate results are kept in the stripe buffers
      const int b0 = std::max(0, r0 - nsteps), b1 = std::min(rows, r1 + nsteps);
      Mat buf[2] = { scratchMat(sb->buf[0], Size(cols, b1 - b0)), scratchMat(sb->buf[1], Size(cols, b1 - b0)) };

      for (int s = 0; s < nsteps; s++) {
        const float step_size = tsteps[s] * 0.5f;
        const Mat &src = s == 0 ? Lstart : buf[(s - 1) & 1];
        const int src_ofs = s == 0 ? 0 : b0;
        Mat &dst = s == nsteps - 1 ? Lout : buf[s & 1];
        const int dst_ofs = s == nsteps - 1 ? 0 : b0;
        // rows still needed by the remaining steps
        const int lo = std::max(0, r0 - (nsteps - 1 - s)), hi = std::min(rows, r1 + (nsteps - 1 - s));
        for (int y = lo; y < hi; y++) {
          nld_step_fused_row(
            y > 0 ? src.ptr<float>(y - 1 - src_ofs) : NULL,
            src.ptr<float>(y - src_ofs),
            y < rows - 1 ? src.ptr<float>(y + 1 - src_ofs) : NULL,
            y > 0 ? Lflow.ptr<float>(y - 1) : NULL,
            Lflow.ptr<float>(y),
            y < rows - 1 ? Lflow.ptr<float>(y + 1) : NULL,
            dst.ptr<float>(y - dst_ofs), cols, step_size);
        }
      }
    }
  }, nstripes);
}

/**
 * @brief This function computes a good empirical value for the k contrast factor
 * given two gradient images, the percentile (0-1), the temporal storage to hold
 * gradient norms and the histogram bins
 * @param Lx Horizontal gradient of the input image
 * @param Ly Vertical gradient of the input image
 * @param nbins Number of histogram bins
 * @return k contrast factor
 */
static inline float
compute_kcontrast(InputArray Lx_, InputArray Ly_, float perc, int nbins)
{
  CV_INSTRUMENT_REGION();

  CV_Assert(nbins > 2);
  CV_Assert(!Lx_.empty());

  Mat Lx = Lx_.getMat();
  Mat Ly = Ly_.getMat();

  // temporary square roots of dot product
  Mat modgs (Lx.rows - 2, Lx.cols - 2, CV_32F);
  const int total = modgs.cols * modgs.rows;
  float *modg = modgs.ptr<float>();
  float hmax = 0.0f;

  for (int i = 1; i < Lx.rows - 1; i++) {
    const float *lx = Lx.ptr<float>(i) + 1;
    const float *ly = Ly.ptr<float>(i) + 1;
    const int cols = Lx.cols - 2;

    for (int j = 0; j < cols; j++) {
      float dist = sqrtf(lx[j] * lx[j] + ly[j] * ly[j]);
      *modg++ = dist;
      hmax = std::max(hmax, dist);
    }
  }
  modg = modgs.ptr<float>();

  if (hmax == 0.0f)
    return 0.03f;  // e.g. a blank image

  // Compute the bin numbers: the value range [0, hmax] -> [0, nbins-1]
  modgs *= (nbins - 1) / hmax;

  // Count up histogram
  std::vector<int> hist(nbins, 0);
  for (int i = 0; i < total; i++)
    hist[(int)modg[i]]++;

  // Now find the perc of the histogram percentile
  const int nthreshold = (int)((total - hist[0]) * perc);  // Exclude hist[0] as background
  int nelements = 0;
  for (int k = 1; k < nbins; k++) {
    if (nelements >= nthreshold)
        return (float)hmax * k / nbins;

    nelements += hist[k];
  }

  return 0.03f;
}

#ifdef HAVE_OPENCL
static inline bool
ocl_pm_g2(InputArray Lx_, InputArray Ly_, OutputArray Lflow_, float kcontrast)
{
  UMat Lx = Lx_.getUMat();
  UMat Ly = Ly_.getUMat();
  UMat Lflow = Lflow_.getUMat();

  int total = Lx.rows * Lx.cols;
  size_t globalSize[] = {(size_t)total};

  ocl::Kernel ker("AKAZE_pm_g2", ocl::xfeatures2d::akaze_oclsrc);
  if( ker.empty() )
    return false;

  return ker.args(
    ocl::KernelArg::PtrReadOnly(Lx),
    ocl::KernelArg::PtrReadOnly(Ly),
    ocl::KernelArg::PtrWriteOnly(Lflow),
    kcontrast, total).run(1, globalSize, 0, true);
}
#endif // HAVE_OPENCL

static inline void
compute_diffusivity(InputArray Lx, InputArray Ly, OutputArray Lflow, float kcontrast, KAZE::DiffusivityType diffusivity)
{
  CV_INSTRUMENT_REGION();

  Lflow.create(Lx.size(), Lx.type());

  switch (diffusivity) {
    case KAZE::DIFF_PM_G1:
      pm_g1(Lx, Ly, Lflow, kcontrast);
    break;
    case KAZE::DIFF_PM_G2:
      CV_OCL_RUN(Lx.isUMat() && Ly.isUMat() && Lflow.isUMat(), ocl_pm_g2(Lx, Ly, Lflow, kcontrast));
      pm_g2(Lx, Ly, Lflow, kcontrast);
    break;
    case KAZE::DIFF_WEICKERT:
      weickert_diffusivity(Lx, Ly, Lflow, kcontrast);
    break;
    case KAZE::DIFF_CHARBONNIER:
      charbonnier_diffusivity(Lx, Ly, Lflow, kcontrast);
    break;
    default:
      CV_Error_(Error::StsError, ("Diffusivity is not supported: %d", static_cast<int>(diffusivity)));
    break;
  }
}

/**
 * @brief Converts input image to grayscale float image
 *
 * @param image any image
 * @param dst grayscale float image
 */
static inline void prepareInputImage(InputArray image, OutputArray dst)
{
  Mat img = image.getMat();
  if (img.channels() > 1)
    cvtColor(image, img, COLOR_BGR2GRAY);

  if ( img.depth() == CV_32F )
    dst.assign(img);
  else if ( img.depth() == CV_8U )
    img.convertTo(dst, CV_32F, 1.0 / 255.0, 0);
  else if ( img.depth() == CV_16U )
    img.convertTo(dst, CV_32F, 1.0 / 65535.0, 0);
}

/**
 * @brief This method creates the nonlinear scale space for a given image
 * @param image Input image for which the nonlinear scale space needs to be created
 */
template<typename MatType>
static inline void
create_nonlinear_scale_space(InputArray image, const AKAZEOptions &options,
  const std::vector<std::vector<float > > &tsteps_evolution, std::vector<Evolution<MatType> > &evolution)
{
  CV_INSTRUMENT_REGION();
  CV_Assert(evolution.size() > 0);

  // convert input to grayscale float image if needed
  MatType img;
  prepareInputImage(image, img);

  // create first level of the evolution
  int ksize = getGaussianKernelSize(options.soffset);
  GaussianBlur(img, evolution[0].Lsmooth, Size(ksize, ksize), options.soffset, options.soffset, BORDER_REPLICATE);
  evolution[0].Lsmooth.copyTo(evolution[0].Lt);

  if (evolution.size() == 1) {
    // we don't need to compute kcontrast factor
    Compute_Determinant_Hessian_Response(evolution);
    return;
  }

  // derivatives, flow and diffusion step
  MatType Lx, Ly, Lsmooth, Lflow, Lstep;

  // compute derivatives for computing k contrast
  GaussianBlur(img, Lsmooth, Size(5, 5), 1.0f, 1.0f, BORDER_REPLICATE);
  Scharr(Lsmooth, Lx, CV_32F, 1, 0, 1, 0, BORDER_DEFAULT);
  Scharr(Lsmooth, Ly, CV_32F, 0, 1, 1, 0, BORDER_DEFAULT);
  Lsmooth.release();
  // compute the kcontrast factor
  float kcontrast = compute_kcontrast(Lx, Ly, options.kcontrast_percentile, options.kcontrast_nbins);

  // Now generate the rest of evolution levels
  for (size_t i = 1; i < evolution.size(); i++) {
    Evolution<MatType> &e = evolution[i];

    if (e.octave > evolution[i - 1].octave) {
      // new octave will be half the size
      resize(evolution[i - 1].Lt, e.Lt, e.size, 0, 0, INTER_AREA);
      kcontrast *= 0.75f;
    }
    else {
      evolution[i - 1].Lt.copyTo(e.Lt);
    }

    GaussianBlur(e.Lt, e.Lsmooth, Size(5, 5), 1.0f, 1.0f, BORDER_REPLICATE);

    // Compute the Gaussian derivatives Lx and Ly
    Scharr(e.Lsmooth, Lx, CV_32F, 1, 0, 1.0, 0, BORDER_DEFAULT);
    Scharr(e.Lsmooth, Ly, CV_32F, 0, 1, 1.0, 0, BORDER_DEFAULT);

    // Compute the conductivity equation
    compute_diffusivity(Lx, Ly, Lflow, kcontrast, options.diffusivity);

    // Perform Fast Explicit Diffusion on Lt
    const std::vector<float> &tsteps = tsteps_evolution[i - 1];
    for (size_t j = 0; j < tsteps.size(); j++) {
      const float step_size = tsteps[j] * 0.5f;
      non_linear_diffusion_step(e.Lt, Lflow, Lstep, step_size);
      add(e.Lt, Lstep, e.Lt);
    }
  }

  Compute_Determinant_Hessian_Response(evolution);

  return;
}

/**
 * @brief Converts between UMatPyramid and Pyramid and vice versa
 * @details Matrices in evolution levels will be copied
 *
 * @param src source pyramid
 * @param dst destination pyramid
 */
template<typename MatTypeSrc, typename MatTypeDst>
static inline void
convertScalePyramid(const std::vector<Evolution<MatTypeSrc> >& src, std::vector<Evolution<MatTypeDst> > &dst)
{
  dst.resize(src.size());
  for (size_t i = 0; i < src.size(); ++i) {
    dst[i] = Evolution<MatTypeDst>(src[i]);
  }
}

/**
 * @brief This method creates the nonlinear scale space for a given image
 * @param image Input image for which the nonlinear scale space needs to be created
 */
static void
create_nonlinear_scale_space_cpu(InputArray image, const AKAZEOptions &options,
  const std::vector<std::vector<float > > &tsteps_evolution, Pyramid &evolution);

void AKAZEFeatures::Create_Nonlinear_Scale_Space(InputArray image)
{
  if (ocl::isOpenCLActivated() && image.isUMat()) {
    // will run OCL version of scale space pyramid
    UMatPyramid uPyr;
    // init UMat pyramid with sizes
    convertScalePyramid(evolution_, uPyr);
    create_nonlinear_scale_space(image, options_, tsteps_, uPyr);
    // download pyramid from GPU
    convertScalePyramid(uPyr, evolution_);
  } else {
    // CPU version
    create_nonlinear_scale_space_cpu(image, options_, tsteps_, evolution_);
  }
}

/* ************************************************************************* */

#ifdef HAVE_OPENCL
static inline bool
ocl_compute_determinant(InputArray Lxx_, InputArray Lxy_, InputArray Lyy_,
  OutputArray Ldet_, float sigma)
{
  UMat Lxx = Lxx_.getUMat();
  UMat Lxy = Lxy_.getUMat();
  UMat Lyy = Lyy_.getUMat();
  UMat Ldet = Ldet_.getUMat();

  const int total = Lxx.rows * Lxx.cols;
  size_t globalSize[] = {(size_t)total};

  ocl::Kernel ker("AKAZE_compute_determinant", ocl::xfeatures2d::akaze_oclsrc);
  if( ker.empty() )
    return false;

  return ker.args(
    ocl::KernelArg::PtrReadOnly(Lxx),
    ocl::KernelArg::PtrReadOnly(Lxy),
    ocl::KernelArg::PtrReadOnly(Lyy),
    ocl::KernelArg::PtrWriteOnly(Ldet),
    sigma, total).run(1, globalSize, 0, true);
}
#endif // HAVE_OPENCL

/**
 * @brief Compute determinant from hessians
 * @details Compute Ldet by (Lxx.mul(Lyy) - Lxy.mul(Lxy)) * sigma
 *
 * @param Lxx spatial derivates
 * @param Lxy spatial derivates
 * @param Lyy spatial derivates
 * @param Ldet output determinant
 * @param sigma determinant will be scaled by this sigma
 */
static inline void compute_determinant(InputArray Lxx_, InputArray Lxy_, InputArray Lyy_,
  OutputArray Ldet_, float sigma)
{
  CV_INSTRUMENT_REGION();

  Ldet_.create(Lxx_.size(), Lxx_.type());

  CV_OCL_RUN(Lxx_.isUMat() && Ldet_.isUMat(), ocl_compute_determinant(Lxx_, Lxy_, Lyy_, Ldet_, sigma));

  // output determinant
  Mat Lxx = Lxx_.getMat(), Lxy = Lxy_.getMat(), Lyy = Lyy_.getMat(), Ldet = Ldet_.getMat();
  float *lxx = Lxx.ptr<float>();
  float *lxy = Lxy.ptr<float>();
  float *lyy = Lyy.ptr<float>();
  float *ldet = Ldet.ptr<float>();
  const int total = Lxx.cols * Lxx.rows;
  for (int j = 0; j < total; j++) {
    ldet[j] = (lxx[j] * lyy[j] - lxy[j] * lxy[j]) * sigma;
  }

}

template <typename MatType>
class DeterminantHessianResponse : public ParallelLoopBody
{
public:
    explicit DeterminantHessianResponse(std::vector<Evolution<MatType> >& ev)
    : evolution_(&ev)
  {
  }

  void operator()(const Range& range) const CV_OVERRIDE
  {
    MatType Lxx, Lxy, Lyy;

    for (int i = range.start; i < range.end; i++)
    {
      Evolution<MatType> &e = (*evolution_)[i];

      // we cannot use cv:Scharr here, because we need to handle also
      // kernel sizes other than 3, by default we are using 9x9, 5x5 and 7x7

      // compute kernels
      Mat DxKx, DxKy, DyKx, DyKy;
      compute_derivative_kernels(DxKx, DxKy, 1, 0, e.sigma_size);
      compute_derivative_kernels(DyKx, DyKy, 0, 1, e.sigma_size);

      // compute the multiscale derivatives
      sepFilter2D(e.Lsmooth, e.Lx, CV_32F, DxKx, DxKy);
      sepFilter2D(e.Lx, Lxx, CV_32F, DxKx, DxKy);
      sepFilter2D(e.Lx, Lxy, CV_32F, DyKx, DyKy);
      sepFilter2D(e.Lsmooth, e.Ly, CV_32F, DyKx, DyKy);
      sepFilter2D(e.Ly, Lyy, CV_32F, DyKx, DyKy);

      // free Lsmooth to same some space in the pyramid, it is not needed anymore
      e.Lsmooth.release();

      // compute determinant scaled by sigma
      float sigma_size_quat = (float)(e.sigma_size * e.sigma_size * e.sigma_size * e.sigma_size);
      compute_determinant(Lxx, Lxy, Lyy, e.Ldet, sigma_size_quat);
    }
  }

private:
  std::vector<Evolution<MatType> >*  evolution_;
};


/**
 * @brief This method computes the feature detector response for the nonlinear scale space
 * @details OCL version
 * @note We use the Hessian determinant as the feature detector response
 */
static inline void
Compute_Determinant_Hessian_Response(UMatPyramid &evolution) {
  CV_INSTRUMENT_REGION();

  DeterminantHessianResponse<UMat> body (evolution);
  body(Range(0, (int)evolution.size()));
}

/**
 * @brief One row of the multiscale derivative filters, exploiting their sparsity
 * @details For scale > 1 the kernels of compute_derivative_kernels() have only three
 * non-zero taps, at -scale, 0 and +scale. With deriv_x the result is
 * sepFilter2D(src, DxKx, DxKy) (difference along x, smoothing along y), otherwise
 * sepFilter2D(src, DyKx, DyKy). Border: BORDER_REFLECT_101, as sepFilter2D.
 * @note Not bit-exact with sepFilter2D: the sums are done in a different order.
 * @param buf temporary of at least src.cols + 2*scale floats
 */
static void
sparse_derivative_row(const Mat& src, int y, bool deriv_x, int scale, float* buf, float* dst)
{
  const float w = 10.0f / 3.0f;
  const float a = 1.0f / (2.0f * scale * (w + 2.0f)), b = w * a;
  const int rows = src.rows, cols = src.cols;
  const float *up = src.ptr<float>(borderInterpolate(y - scale, rows, BORDER_REFLECT_101));
  const float *c = src.ptr<float>(y);
  const float *dn = src.ptr<float>(borderInterpolate(y + scale, rows, BORDER_REFLECT_101));
  float *v = buf + scale;

  // vertical pass
  if (deriv_x)
    for (int x = 0; x < cols; x++)
      v[x] = a * (up[x] + dn[x]) + b * c[x];
  else
    for (int x = 0; x < cols; x++)
      v[x] = dn[x] - up[x];

  // horizontal border
  for (int k = 1; k <= scale; k++) {
    v[-k] = v[borderInterpolate(-k, cols, BORDER_REFLECT_101)];
    v[cols - 1 + k] = v[borderInterpolate(cols - 1 + k, cols, BORDER_REFLECT_101)];
  }

  // horizontal pass
  if (deriv_x)
    for (int x = 0; x < cols; x++)
      dst[x] = v[x + scale] - v[x - scale];
  else
    for (int x = 0; x < cols; x++)
      dst[x] = a * (v[x - scale] + v[x + scale]) + b * v[x];
}

/**
 * @brief Computes the multiscale first derivatives Lx, Ly of one level from its smoothed image
 * @details Same filters as DeterminantHessianResponse<Mat>, in parallel over row stripes.
 */
static void
hessian_first_derivatives(MEvolution &e, const Mat &Lsmooth)
{
  Mat DxKx, DxKy, DyKx, DyKy;
  compute_derivative_kernels(DxKx, DxKy, 1, 0, e.sigma_size);
  compute_derivative_kernels(DyKx, DyKy, 0, 1, e.sigma_size);
  e.Lx.create(Lsmooth.size(), CV_32F);
  e.Ly.create(Lsmooth.size(), CV_32F);
  if (e.sigma_size > 1) {
    const int scale = e.sigma_size;
    parallel_for_rows(Lsmooth.rows, [&](int r0, int r1) {
      std::vector<float> buf(Lsmooth.cols + 2 * scale);
      for (int y = r0; y < r1; y++) {
        sparse_derivative_row(Lsmooth, y, true, scale, buf.data(), e.Lx.ptr<float>(y));
        sparse_derivative_row(Lsmooth, y, false, scale, buf.data(), e.Ly.ptr<float>(y));
      }
    });
    return;
  }
  parallel_for_rows(Lsmooth.rows, [&](int r0, int r1) {
    const Mat src = Lsmooth.rowRange(r0, r1);
    Mat Lx = e.Lx.rowRange(r0, r1);
    Mat Ly = e.Ly.rowRange(r0, r1);
    sepFilter2D(src, Lx, CV_32F, DxKx, DxKy);
    sepFilter2D(src, Ly, CV_32F, DyKx, DyKy);
  });
}

/**
 * @brief Computes the detector response Ldet of all levels from their first derivatives
 * @note We use the Hessian determinant as the feature detector response
 * @details Same filters as DeterminantHessianResponse<Mat>, but the work is split into
 * row stripes of all levels, so the large levels of the first octave do not serialize
 * the computation. The second derivatives are stripe-local temporaries.
 */
static void
hessian_determinants(Pyramid &evolution)
{
  struct Stripe { int level, row_begin, row_end; };
  const int nlevels = (int)evolution.size();
  std::vector<Mat> DxKx(nlevels), DxKy(nlevels), DyKx(nlevels), DyKy(nlevels);
  std::vector<Stripe> stripes;
  const int stripe_rows = 64;
  for (int i = 0; i < nlevels; i++) {
    MEvolution &e = evolution[i];
    CV_Assert(!e.Lx.empty() && !e.Ly.empty());
    compute_derivative_kernels(DxKx[i], DxKy[i], 1, 0, e.sigma_size);
    compute_derivative_kernels(DyKx[i], DyKy[i], 0, 1, e.sigma_size);
    e.Ldet.create(e.Lx.size(), CV_32F);
    const int rows = e.Lx.rows;
    const int n = std::max(1, rows / stripe_rows);
    for (int k = 0; k < n; k++)
      stripes.push_back({i, k * rows / n, (k + 1) * rows / n});
  }
  // biggest stripes first for better load balance
  std::stable_sort(stripes.begin(), stripes.end(), [&](const Stripe& a, const Stripe& b) {
    return (size_t)(a.row_end - a.row_begin) * evolution[a.level].Lx.cols >
           (size_t)(b.row_end - b.row_begin) * evolution[b.level].Lx.cols;
  });

  TLSData<ScratchBuffers> scratch;
  parallel_for_(Range(0, (int)stripes.size()), [&](const Range& range) {
    ScratchBuffers *sb = scratch.get();
    for (int s = range.start; s < range.end; s++) {
      const Stripe &st = stripes[s];
      MEvolution &e = evolution[st.level];
      const Size sz(e.Lx.cols, st.row_end - st.row_begin);
      Mat Lxx = scratchMat(sb->buf[0], sz), Lxy = scratchMat(sb->buf[1], sz), Lyy = scratchMat(sb->buf[2], sz);
      if (e.sigma_size > 1) {
        const int scale = e.sigma_size;
        float *buf = scratchMat(sb->buf[3], Size(e.Lx.cols + 2 * scale, 1)).ptr<float>();
        for (int y = st.row_begin; y < st.row_end; y++) {
          sparse_derivative_row(e.Lx, y, true, scale, buf, Lxx.ptr<float>(y - st.row_begin));
          sparse_derivative_row(e.Lx, y, false, scale, buf, Lxy.ptr<float>(y - st.row_begin));
          sparse_derivative_row(e.Ly, y, false, scale, buf, Lyy.ptr<float>(y - st.row_begin));
        }
      }
      else {
        const Mat Lx = e.Lx.rowRange(st.row_begin, st.row_end);
        const Mat Ly = e.Ly.rowRange(st.row_begin, st.row_end);
        sepFilter2D(Lx, Lxx, CV_32F, DxKx[st.level], DxKy[st.level]);
        sepFilter2D(Lx, Lxy, CV_32F, DyKx[st.level], DyKy[st.level]);
        sepFilter2D(Ly, Lyy, CV_32F, DyKx[st.level], DyKy[st.level]);
      }

      // determinant scaled by sigma, as compute_determinant()
      const float sigma = (float)(e.sigma_size * e.sigma_size * e.sigma_size * e.sigma_size);
      const int cols = Lxx.cols;
      for (int y = 0; y < Lxx.rows; y++) {
        const float *lxx = Lxx.ptr<float>(y), *lxy = Lxy.ptr<float>(y), *lyy = Lyy.ptr<float>(y);
        float *ldet = e.Ldet.ptr<float>(st.row_begin + y);
        int x = 0;
#if (CV_SIMD || CV_SIMD_SCALABLE)
        const int vlanes = VTraits<v_float32>::vlanes();
        const v_float32 vsigma = vx_setall_f32(sigma);
        for (; x <= cols - vlanes; x += vlanes) {
          v_float32 vxy = vx_load(lxy + x);
          v_store(ldet + x, v_mul(v_sub(v_mul(vx_load(lxx + x), vx_load(lyy + x)), v_mul(vxy, vxy)), vsigma));
        }
#endif
        for (; x < cols; x++)
          ldet[x] = (lxx[x] * lyy[x] - lxy[x] * lxy[x]) * sigma;
      }
    }
  }, (double)stripes.size());
}

/**
 * @brief Parallel version of compute_kcontrast() working directly on the input image
 * @details Smooths the image, computes Scharr gradients and their histogram per
 * row stripe; gives the same result as the sequential version.
 */
static float
compute_kcontrast_parallel(const Mat& img, float perc, int nbins, Mat& buf0, Mat& buf1)
{
  CV_INSTRUMENT_REGION();
  CV_Assert(nbins > 2);
  CV_Assert(!img.empty());

  Mat Lsmooth = scratchMat(buf0, img.size());
  parallel_for_rows(img.rows, [&](int r0, int r1) {
    Mat dst = Lsmooth.rowRange(r0, r1);
    GaussianBlur(img.rowRange(r0, r1), dst, Size(5, 5), 1.0f, 1.0f, BORDER_REPLICATE);
  });

  // square roots of dot product of the inner pixels
  if (img.rows <= 2 || img.cols <= 2)
    return 0.03f;
  Mat modgs = scratchMat(buf1, Size(img.cols - 2, img.rows - 2));
  Mutex mtx;
  float hmax = 0.0f;
  TLSData<ScratchBuffers> scratch;
  parallel_for_rows(modgs.rows, [&](int r0, int r1) {
    ScratchBuffers *sb = scratch.get();
    Mat Lx = scratchMat(sb->buf[0], Size(img.cols, r1 - r0));
    Mat Ly = scratchMat(sb->buf[1], Size(img.cols, r1 - r0));
    const Mat src = Lsmooth.rowRange(r0 + 1, r1 + 1);
    Scharr(src, Lx, CV_32F, 1, 0, 1, 0, BORDER_DEFAULT);
    Scharr(src, Ly, CV_32F, 0, 1, 1, 0, BORDER_DEFAULT);
    float lmax = 0.0f;
    const int cols = modgs.cols;
    for (int i = 0; i < r1 - r0; i++) {
      const float *lx = Lx.ptr<float>(i) + 1;
      const float *ly = Ly.ptr<float>(i) + 1;
      float *modg = modgs.ptr<float>(r0 + i);
      int j = 0;
#if (CV_SIMD || CV_SIMD_SCALABLE)
      const int vlanes = VTraits<v_float32>::vlanes();
      v_float32 vmax = vx_setzero_f32();
      for (; j <= cols - vlanes; j += vlanes) {
        v_float32 x = vx_load(lx + j), y = vx_load(ly + j);
        v_float32 d = v_sqrt(v_add(v_mul(x, x), v_mul(y, y)));
        v_store(modg + j, d);
        vmax = v_max(vmax, d);
      }
      lmax = std::max(lmax, v_reduce_max(vmax));
#endif
      for (; j < cols; j++) {
        float dist = sqrtf(lx[j] * lx[j] + ly[j] * ly[j]);
        modg[j] = dist;
        lmax = std::max(lmax, dist);
      }
    }
    AutoLock lock(mtx);
    hmax = std::max(hmax, lmax);
  });

  if (hmax == 0.0f)
    return 0.03f;  // e.g. a blank image

  // Compute the bin numbers: the value range [0, hmax] -> [0, nbins-1] and count up histogram
  std::vector<int> hist(nbins, 0);
  parallel_for_rows(modgs.rows, [&](int r0, int r1) {
    Mat m = modgs.rowRange(r0, r1);
    m *= (nbins - 1) / hmax;
    std::vector<int> lhist(nbins, 0);
    for (int i = 0; i < m.rows; i++) {
      const float *modg = m.ptr<float>(i);
      for (int j = 0; j < m.cols; j++)
        lhist[(int)modg[j]]++;
    }
    AutoLock lock(mtx);
    for (int k = 0; k < nbins; k++)
      hist[k] += lhist[k];
  });

  // Now find the perc of the histogram percentile
  const int total = modgs.cols * modgs.rows;
  const int nthreshold = (int)((total - hist[0]) * perc);  // Exclude hist[0] as background
  int nelements = 0;
  for (int k = 1; k < nbins; k++) {
    if (nelements >= nthreshold)
        return (float)hmax * k / nbins;

    nelements += hist[k];
  }

  return 0.03f;
}

/**
 * @brief CPU version of create_nonlinear_scale_space()
 * @details Produces the same pyramid, but every filtering stage runs in parallel
 * over row stripes, Scharr derivatives are fused with the diffusivity, and the
 * FED steps are fused with the update (no Lstep image, no copies of Lt).
 */
static void
create_nonlinear_scale_space_cpu(InputArray image, const AKAZEOptions &options,
  const std::vector<std::vector<float > > &tsteps_evolution, Pyramid &evolution)
{
  CV_INSTRUMENT_REGION();
  CV_Assert(evolution.size() > 0);

  // convert input to grayscale float image if needed
  Mat img;
  {
    Mat gray = image.getMat();
    if (gray.channels() > 1)
      cvtColor(gray, gray, COLOR_BGR2GRAY);
    if (gray.depth() == CV_8U || gray.depth() == CV_16U) {
      const double scale = gray.depth() == CV_8U ? 1.0 / 255.0 : 1.0 / 65535.0;
      img.create(gray.size(), CV_32F);
      parallel_for_rows(gray.rows, [&](int r0, int r1) {
        Mat dst = img.rowRange(r0, r1);
        gray.rowRange(r0, r1).convertTo(dst, CV_32F, scale, 0);
      });
    }
    else {
      prepareInputImage(gray, img);
    }
  }

  // create first level of the evolution
  MEvolution &e0 = evolution[0];
  const int ksize = getGaussianKernelSize(options.soffset);
  // Lt of the first level is its smoothed image (it is never modified)
  e0.Lt.create(img.size(), CV_32F);
  parallel_for_rows(img.rows, [&](int r0, int r1) {
    Mat dst = e0.Lt.rowRange(r0, r1);
    GaussianBlur(img.rowRange(r0, r1), dst, Size(ksize, ksize), options.soffset, options.soffset, BORDER_REPLICATE);
  });
  // The first derivatives of each level are computed as soon as its smoothed image
  // is ready, so the smoothed images need not be kept in the pyramid
  hessian_first_derivatives(e0, e0.Lt);

  if (evolution.size() == 1) {
    // we don't need to compute kcontrast factor
    hessian_determinants(evolution);
    return;
  }

  // compute the kcontrast factor
  // buffers reused by all levels (the first level is the biggest one)
  Mat Lflow_buf, Ltmp_buf[2], Lsmooth_buf;
  float kcontrast = compute_kcontrast_parallel(img, options.kcontrast_percentile, options.kcontrast_nbins,
                                               Ltmp_buf[0], Ltmp_buf[1]);
  img.release();
  TLSData<ScratchBuffers> scratch;

  // Now generate the rest of evolution levels
  for (size_t i = 1; i < evolution.size(); i++) {
    MEvolution &e = evolution[i];
    const int rows = e.size.height;

    // Lt before diffusion: the previous level, or half of it on a new octave
    Mat Lstart;
    if (e.octave > evolution[i - 1].octave) {
      resize(evolution[i - 1].Lt, Lstart, e.size, 0, 0, INTER_AREA);
      kcontrast *= 0.75f;
    }
    else {
      Lstart = evolution[i - 1].Lt;
    }

    Mat Lsmooth = scratchMat(Lsmooth_buf, e.size);
    parallel_for_rows(rows, [&](int r0, int r1) {
      Mat dst = Lsmooth.rowRange(r0, r1);
      GaussianBlur(Lstart.rowRange(r0, r1), dst, Size(5, 5), 1.0f, 1.0f, BORDER_REPLICATE);
    });

    // Compute the Gaussian derivatives Lx and Ly and the conductivity equation
    Mat Lflow = scratchMat(Lflow_buf, e.size);
    parallel_for_rows(rows, [&](int r0, int r1) {
      ScratchBuffers *sb = scratch.get();
      Mat Lx = scratchMat(sb->buf[0], Size(e.size.width, r1 - r0));
      Mat Ly = scratchMat(sb->buf[1], Size(e.size.width, r1 - r0));
      const Mat src = Lsmooth.rowRange(r0, r1);
      Scharr(src, Lx, CV_32F, 1, 0, 1.0, 0, BORDER_DEFAULT);
      Scharr(src, Ly, CV_32F, 0, 1, 1.0, 0, BORDER_DEFAULT);
      Mat dst = Lflow.rowRange(r0, r1);
      compute_diffusivity(Lx, Ly, dst, kcontrast, options.diffusivity);
    });
    hessian_first_derivatives(e, Lsmooth);

    // Perform Fast Explicit Diffusion on Lt
    const std::vector<float> &tsteps = tsteps_evolution[i - 1];
    if (tsteps.empty())
      e.Lt = Lstart.clone();
    else
      fed_cycle(Lstart, Lflow, tsteps, e.Lt, scratch, Ltmp_buf);
  }

  hessian_determinants(evolution);
}

/* ************************************************************************* */

/**
 * @brief This method selects interesting keypoints through the nonlinear scale space
 * @param kpts Vector of detected keypoints
 */
void AKAZEFeatures::Feature_Detection(std::vector<KeyPoint>& kpts)
{
  CV_INSTRUMENT_REGION();

  kpts.clear();
  std::vector<Mat> keypoints_by_layers;
  std::vector<std::vector<int> > candidates_by_layers;
  Find_Scale_Space_Extrema(keypoints_by_layers, candidates_by_layers);
  Do_Subpixel_Refinement(keypoints_by_layers, candidates_by_layers, kpts);
  Compute_Keypoints_Orientation(kpts);
}

/**
 * @brief This method searches v for a neighbor point of the point candidate p
 * @param x Coordinates of the keypoint candidate to search a neighbor
 * @param y Coordinates of the keypoint candidate to search a neighbor
 * @param mask Matrix holding keypoints positions
 * @param search_radius neighbour radius for searching keypoints
 * @param idx The index to mask, pointing to keypoint found.
 * @return true if a neighbor point is found; false otherwise
 */
static inline bool
find_neighbor_point(const int x, const int y, const Mat &mask, const int search_radius, int &idx)
{
  // search neighborhood for keypoints
  for (int i = y - search_radius; i < y + search_radius; ++i) {
    const uchar *curr = mask.ptr<uchar>(i);
    for (int j = x - search_radius; j < x + search_radius; ++j) {
      if (curr[j] == 0) {
        continue; // skip non-keypoint
      }
      // fine-compare with L2 metric (L2 is smaller than our search window)
      int dx = j - x;
      int dy = i - y;
      if (dx * dx + dy * dy <= search_radius * search_radius) {
        idx = i * mask.cols + j;
        return true;
      }
    }
  }

  return false;
}

/**
 * @brief Find keypoints in parallel for each pyramid layer
 */
class FindKeypointsSameScale : public ParallelLoopBody
{
public:
    explicit FindKeypointsSameScale(const Pyramid& ev,
      std::vector<Mat>& kpts, std::vector<std::vector<int> >& candidates, float dthreshold)
    : evolution_(&ev), keypoints_by_layers_(&kpts), candidates_by_layers_(&candidates), dthreshold_(dthreshold)
  {}

  void operator()(const Range& range) const CV_OVERRIDE
  {
    for (int i = range.start; i < range.end; i++)
    {
      const MEvolution &e = (*evolution_)[i];
      Mat &kpts = (*keypoints_by_layers_)[i];
      std::vector<int> &candidates = (*candidates_by_layers_)[i];
      candidates.clear();
      // this mask will hold positions of keypoints in this level
      kpts = Mat::zeros(e.Ldet.size(), CV_8UC1);

      // if border is too big we shouldn't search any keypoints
      if (e.border + 1 >= e.Ldet.rows)
        continue;

      const float * prev = e.Ldet.ptr<float>(e.border - 1);
      const float * curr = e.Ldet.ptr<float>(e.border    );
      const float * next = e.Ldet.ptr<float>(e.border + 1);
      const float * ldet = e.Ldet.ptr<float>();
      uchar *mask = kpts.ptr<uchar>();
      const int search_radius = e.sigma_size; // size of keypoint in this level

      for (int y = e.border; y < e.Ldet.rows - e.border; y++) {
        for (int x = e.border; x < e.Ldet.cols - e.border; x++) {
          const float value = curr[x];

          // Filter the points with the detector threshold
          if (value <= dthreshold_)
            continue;
          if (value <= curr[x-1] || value <= curr[x+1])
            continue;
          if (value <= prev[x-1] || value <= prev[x  ] || value <= prev[x+1])
            continue;
          if (value <= next[x-1] || value <= next[x  ] || value <= next[x+1])
            continue;

          int idx = 0;
          // Compare response with the same scale
          if (find_neighbor_point(x, y, kpts, search_radius, idx)) {
            if (value > ldet[idx]) {
              mask[idx] = 0;  // clear old point - we have better candidate now
            } else {
              continue; // there already is a better keypoint
            }
          }

          kpts.at<uchar>(y, x) = 1; // we have a new keypoint
          candidates.push_back(y * e.Ldet.cols + x);
        }

        prev = curr;
        curr = next;
        next += e.Ldet.cols;
      }

      // keep the positions (in raster order) of the keypoints that survived
      candidates.erase(std::remove_if(candidates.begin(), candidates.end(),
                                      [mask](int idx) { return mask[idx] == 0; }),
                       candidates.end());
    }
  }

private:
  const Pyramid*  evolution_;
  std::vector<Mat>* keypoints_by_layers_;
  std::vector<std::vector<int> >* candidates_by_layers_;
  float dthreshold_; ///< Detector response threshold to accept point
};

/**
 * @brief Keeps the candidates whose position is still set in the mask
 */
static inline void
filter_candidates(const std::vector<int>& candidates, const Mat& mask, std::vector<int>& dst)
{
  const uchar *m = mask.ptr<uchar>();
  dst.clear();
  for (size_t k = 0; k < candidates.size(); k++)
    if (m[candidates[k]])
      dst.push_back(candidates[k]);
}

/**
 * @brief This method finds extrema in the nonlinear scale space
 * @param keypoints_by_layers Output masks of detected keypoints; one mask for each evolution level
 * @param candidates_by_layers Output positions of the keypoints in raster order; one vector for each evolution level
 * @details The comparisons between adjacent levels are done in parallel. This is
 * equivalent to the sequential sweeps: in the upward sweep the step for level i only
 * reads/writes the mask of level i-1 and iterates the keypoints of level i, which
 * are not modified before this step (only step i+1 modifies them). The same holds
 * for the downward sweep, iterating over the keypoints that survived the upward sweep.
 */
void AKAZEFeatures::Find_Scale_Space_Extrema(std::vector<Mat>& keypoints_by_layers,
                                             std::vector<std::vector<int> >& candidates_by_layers)
{
  CV_INSTRUMENT_REGION();

  const int nlevels = (int)evolution_.size();
  keypoints_by_layers.resize(nlevels);
  candidates_by_layers.resize(nlevels);

  // find points in the same level
  parallel_for_(Range(0, nlevels),
    FindKeypointsSameScale(evolution_, keypoints_by_layers, candidates_by_layers, options_.dthreshold));

  // Filter points with the lower scale level
  parallel_for_(Range(1, nlevels), [&](const Range& range) {
    for (int i = range.start; i < range.end; i++) {
      // constants for this level
      const std::vector<int> &candidates = candidates_by_layers[i];
      const int cols = keypoints_by_layers[i].cols;
      uchar *const kpts_prev = keypoints_by_layers[i-1].ptr<uchar>();
      const float *const ldet = evolution_[i].Ldet.ptr<float>();
      const float *const ldet_prev = evolution_[i-1].Ldet.ptr<float>();
      // ratios are just powers of 2
      const int diff_ratio = (int)evolution_[i].octave_ratio / (int)evolution_[i-1].octave_ratio;
      const int search_radius = evolution_[i].sigma_size * diff_ratio; // size of keypoint in this level

      for (size_t k = 0; k < candidates.size(); k++) {
        const int j = candidates[k];
        int idx = 0;
        // project point to lower scale layer
        const int p_x = (j % cols) * diff_ratio;
        const int p_y = (j / cols) * diff_ratio;
        if (find_neighbor_point(p_x, p_y, keypoints_by_layers[i-1], search_radius, idx)) {
          if (ldet[j] > ldet_prev[idx]) {
            kpts_prev[idx] = 0;  // clear keypoint in lower layer
          }
          // else this pt may be pruned by the upper scale
        }
      }
    }
  });

  // Keypoints that survived the upward sweep
  std::vector<std::vector<int> > survivors(nlevels);
  parallel_for_(Range(0, nlevels), [&](const Range& range) {
    for (int i = range.start; i < range.end; i++)
      filter_candidates(candidates_by_layers[i], keypoints_by_layers[i], survivors[i]);
  });

  // Now filter points with the upper scale level (the other direction)
  parallel_for_(Range(0, nlevels - 1), [&](const Range& range) {
    for (int i = range.start; i < range.end; i++) {
      // constants for this level
      const std::vector<int> &candidates = survivors[i];
      const int cols = keypoints_by_layers[i].cols;
      uchar *const kpts_next = keypoints_by_layers[i+1].ptr<uchar>();
      const float *const ldet = evolution_[i].Ldet.ptr<float>();
      const float *const ldet_next = evolution_[i+1].Ldet.ptr<float>();
      // ratios are just powers of 2, i+1 ratio is always greater or equal to i
      const int diff_ratio = (int)evolution_[i+1].octave_ratio / (int)evolution_[i].octave_ratio;
      const int search_radius = evolution_[i+1].sigma_size; // size of keypoints in upper level

      for (size_t k = 0; k < candidates.size(); k++) {
        const int j = candidates[k];
        int idx = 0;
        // project point to upper scale layer
        const int p_x = (j % cols) / diff_ratio;
        const int p_y = (j / cols) / diff_ratio;
        if (find_neighbor_point(p_x, p_y, keypoints_by_layers[i+1], search_radius, idx)) {
          if (ldet[j] > ldet_next[idx]) {
            kpts_next[idx] = 0;  // clear keypoint in upper layer
          }
        }
      }
    }
  });

  // Final keypoints of each level
  parallel_for_(Range(0, nlevels), [&](const Range& range) {
    for (int i = range.start; i < range.end; i++)
      filter_candidates(survivors[i], keypoints_by_layers[i], candidates_by_layers[i]);
  });
}

/* ************************************************************************* */
/**
 * @brief This method performs subpixel refinement of the detected keypoints
 * @param keypoints_by_layers Input masks of detected keypoints, sorted by evolution levels
 * @param candidates_by_layers Input positions of the detected keypoints, sorted by evolution levels
 * @param kpts Output vector of the final refined keypoints
 */
void AKAZEFeatures::Do_Subpixel_Refinement(
  std::vector<Mat>& keypoints_by_layers, std::vector<std::vector<int> >& candidates_by_layers,
  std::vector<KeyPoint>& output_keypoints)
{
  CV_INSTRUMENT_REGION();

  const int nlevels = (int)keypoints_by_layers.size();
  std::vector<std::vector<KeyPoint> > refined(nlevels);

  parallel_for_(Range(0, nlevels), [&](const Range& range) {
   for (int i = range.start; i < range.end; i++) {
    const MEvolution &e = evolution_[i];
    const float * const ldet = e.Ldet.ptr<float>();
    const float ratio = e.octave_ratio;
    const int cols = e.Ldet.cols;
    const std::vector<int> &candidates = candidates_by_layers[i];
    std::vector<KeyPoint> &level_keypoints = refined[i];
    level_keypoints.reserve(candidates.size());

    for (size_t k = 0; k < candidates.size(); k++) {
        const int j = candidates[k];
        const int x = j % cols;
        const int y = j / cols;

        // create a new keypoint
        KeyPoint kp;
        kp.pt.x = x * e.octave_ratio;
        kp.pt.y = y * e.octave_ratio;
        kp.size = e.esigma * options_.derivative_factor;
        kp.angle = -1;
        kp.response = ldet[j];
        kp.octave = e.octave;
        kp.class_id = static_cast<int>(i);

        // Compute the gradient
        float Dx = 0.5f * (ldet[ y     *cols + x + 1] - ldet[ y     *cols + x - 1]);
        float Dy = 0.5f * (ldet[(y + 1)*cols + x    ] - ldet[(y - 1)*cols + x    ]);

        // Compute the Hessian
        float Dxx = ldet[ y     *cols + x + 1] + ldet[ y     *cols + x - 1] - 2.0f * ldet[y*cols + x];
        float Dyy = ldet[(y + 1)*cols + x    ] + ldet[(y - 1)*cols + x    ] - 2.0f * ldet[y*cols + x];
        float Dxy = 0.25f * (ldet[(y + 1)*cols + x + 1] + ldet[(y - 1)*cols + x - 1] -
                            ldet[(y - 1)*cols + x + 1] - ldet[(y + 1)*cols + x - 1]);

        // Solve the linear system
        Matx22f A( Dxx, Dxy,
                   Dxy, Dyy );
        Vec2f   b( -Dx, -Dy );
        Vec2f   dst( 0.0f, 0.0f );
        solve(A, b, dst, DECOMP_LU);

        float dx = dst(0);
        float dy = dst(1);

        if (fabs(dx) > 1.0f || fabs(dy) > 1.0f)
          continue; // Ignore the point that is not stable

        // Refine the coordinates
        kp.pt.x += dx * ratio + .5f*(ratio-1.f);
        kp.pt.y += dy * ratio + .5f*(ratio-1.f);

        kp.angle = 0.0;
        kp.size *= 2.0f; // In OpenCV the size of a keypoint is the diameter

        // Push the refined keypoint to the level storage
        level_keypoints.push_back(kp);
    }
   }
  });

  // Concatenate the levels in order
  size_t total = output_keypoints.size();
  for (int i = 0; i < nlevels; i++)
    total += refined[i].size();
  output_keypoints.reserve(total);
  for (int i = 0; i < nlevels; i++)
    output_keypoints.insert(output_keypoints.end(), refined[i].begin(), refined[i].end());
}

/* ************************************************************************* */

class SURF_Descriptor_Upright_64_Invoker : public ParallelLoopBody
{
public:
  SURF_Descriptor_Upright_64_Invoker(std::vector<KeyPoint>& kpts, Mat& desc, const Pyramid& evolution)
    : keypoints_(&kpts)
    , descriptors_(&desc)
    , evolution_(&evolution)
  {
  }

  void operator() (const Range& range) const CV_OVERRIDE
  {
    for (int i = range.start; i < range.end; i++)
    {
      Get_SURF_Descriptor_Upright_64((*keypoints_)[i], descriptors_->ptr<float>(i), descriptors_->cols);
    }
  }

  void Get_SURF_Descriptor_Upright_64(const KeyPoint& kpt, float* desc, int desc_size) const;

private:
  std::vector<KeyPoint>* keypoints_;
  Mat*                   descriptors_;
  const Pyramid*   evolution_;
};

class SURF_Descriptor_64_Invoker : public ParallelLoopBody
{
public:
  SURF_Descriptor_64_Invoker(std::vector<KeyPoint>& kpts, Mat& desc, Pyramid& evolution)
    : keypoints_(&kpts)
    , descriptors_(&desc)
    , evolution_(&evolution)
  {
  }

  void operator()(const Range& range) const CV_OVERRIDE
  {
    for (int i = range.start; i < range.end; i++)
    {
      Get_SURF_Descriptor_64((*keypoints_)[i], descriptors_->ptr<float>(i), descriptors_->cols);
    }
  }

  void Get_SURF_Descriptor_64(const KeyPoint& kpt, float* desc, int desc_size) const;

private:
  std::vector<KeyPoint>* keypoints_;
  Mat*                   descriptors_;
  Pyramid*   evolution_;
};

class MSURF_Upright_Descriptor_64_Invoker : public ParallelLoopBody
{
public:
  MSURF_Upright_Descriptor_64_Invoker(std::vector<KeyPoint>& kpts, Mat& desc, Pyramid& evolution)
    : keypoints_(&kpts)
    , descriptors_(&desc)
    , evolution_(&evolution)
  {
  }

  void operator()(const Range& range) const CV_OVERRIDE
  {
    for (int i = range.start; i < range.end; i++)
    {
      Get_MSURF_Upright_Descriptor_64((*keypoints_)[i], descriptors_->ptr<float>(i), descriptors_->cols);
    }
  }

  void Get_MSURF_Upright_Descriptor_64(const KeyPoint& kpt, float* desc, int desc_size) const;

private:
  std::vector<KeyPoint>* keypoints_;
  Mat*                   descriptors_;
  Pyramid*   evolution_;
};

class MSURF_Descriptor_64_Invoker : public ParallelLoopBody
{
public:
  MSURF_Descriptor_64_Invoker(std::vector<KeyPoint>& kpts, Mat& desc, Pyramid& evolution)
    : keypoints_(&kpts)
    , descriptors_(&desc)
    , evolution_(&evolution)
  {
  }

  void operator() (const Range& range) const CV_OVERRIDE
  {
    for (int i = range.start; i < range.end; i++)
    {
      Get_MSURF_Descriptor_64((*keypoints_)[i], descriptors_->ptr<float>(i), descriptors_->cols);
    }
  }

  void Get_MSURF_Descriptor_64(const KeyPoint& kpt, float* desc, int desc_size) const;

private:
  std::vector<KeyPoint>* keypoints_;
  Mat*                   descriptors_;
  Pyramid*   evolution_;
};

class Upright_MLDB_Full_Descriptor_Invoker : public ParallelLoopBody
{
public:
  Upright_MLDB_Full_Descriptor_Invoker(std::vector<KeyPoint>& kpts, Mat& desc, Pyramid& evolution, AKAZEOptions& options)
    : keypoints_(&kpts)
    , descriptors_(&desc)
    , evolution_(&evolution)
    , options_(&options)
  {
  }

  void operator() (const Range& range) const CV_OVERRIDE
  {
    for (int i = range.start; i < range.end; i++)
    {
      Get_Upright_MLDB_Full_Descriptor((*keypoints_)[i], descriptors_->ptr<unsigned char>(i), descriptors_->cols);
    }
  }

  void Get_Upright_MLDB_Full_Descriptor(const KeyPoint& kpt, unsigned char* desc, int desc_size) const;

private:
  std::vector<KeyPoint>* keypoints_;
  Mat*                   descriptors_;
  Pyramid*   evolution_;
  AKAZEOptions*              options_;
};

class Upright_MLDB_Descriptor_Subset_Invoker : public ParallelLoopBody
{
public:
  Upright_MLDB_Descriptor_Subset_Invoker(std::vector<KeyPoint>& kpts,
                                         Mat& desc,
                                         Pyramid& evolution,
                                         AKAZEOptions& options,
                                         Mat descriptorSamples,
                                         Mat descriptorBits)
    : keypoints_(&kpts)
    , descriptors_(&desc)
    , evolution_(&evolution)
    , options_(&options)
    , descriptorSamples_(descriptorSamples)
    , descriptorBits_(descriptorBits)
  {
  }

  void operator() (const Range& range) const CV_OVERRIDE
  {
    for (int i = range.start; i < range.end; i++)
    {
      Get_Upright_MLDB_Descriptor_Subset((*keypoints_)[i], descriptors_->ptr<unsigned char>(i), descriptors_->cols);
    }
  }

  void Get_Upright_MLDB_Descriptor_Subset(const KeyPoint& kpt, unsigned char* desc, int desc_size) const;

private:
  std::vector<KeyPoint>* keypoints_;
  Mat*                   descriptors_;
  Pyramid*   evolution_;
  AKAZEOptions*              options_;

  Mat descriptorSamples_;  // List of positions in the grids to sample LDB bits from.
  Mat descriptorBits_;
};

class MLDB_Full_Descriptor_Invoker : public ParallelLoopBody
{
public:
  MLDB_Full_Descriptor_Invoker(std::vector<KeyPoint>& kpts, Mat& desc, Pyramid& evolution, AKAZEOptions& options)
    : keypoints_(&kpts)
    , descriptors_(&desc)
    , evolution_(&evolution)
    , options_(&options)
  {
  }

  void operator() (const Range& range) const CV_OVERRIDE
  {
    for (int i = range.start; i < range.end; i++)
    {
      Get_MLDB_Full_Descriptor((*keypoints_)[i], descriptors_->ptr<unsigned char>(i), descriptors_->cols);
    }
  }

  void Get_MLDB_Full_Descriptor(const KeyPoint& kpt, unsigned char* desc, int desc_size) const;
  void MLDB_Sample_Grid(float* grid, uchar* valid, int grid_min, int grid_size, int level,
                        float xf, float yf, float co, float si, float scale) const;
  void MLDB_Fill_Values(float* values, int sample_step, const float* grid, const uchar* valid,
                        int grid_min, int grid_size) const;
  void MLDB_Binary_Comparisons(float* values, unsigned char* desc,
                               int count, int& dpos) const;

private:
  std::vector<KeyPoint>* keypoints_;
  Mat*                   descriptors_;
  Pyramid*   evolution_;
  AKAZEOptions*              options_;
};

class MLDB_Descriptor_Subset_Invoker : public ParallelLoopBody
{
public:
  MLDB_Descriptor_Subset_Invoker(std::vector<KeyPoint>& kpts,
                                 Mat& desc,
                                 Pyramid& evolution,
                                 AKAZEOptions& options,
                                 Mat descriptorSamples,
                                 Mat descriptorBits)
    : keypoints_(&kpts)
    , descriptors_(&desc)
    , evolution_(&evolution)
    , options_(&options)
    , descriptorSamples_(descriptorSamples)
    , descriptorBits_(descriptorBits)
  {
  }

  void operator() (const Range& range) const CV_OVERRIDE
  {
    for (int i = range.start; i < range.end; i++)
    {
      Get_MLDB_Descriptor_Subset((*keypoints_)[i], descriptors_->ptr<unsigned char>(i), descriptors_->cols);
    }
  }

  void Get_MLDB_Descriptor_Subset(const KeyPoint& kpt, unsigned char* desc, int desc_size) const;

private:
  std::vector<KeyPoint>* keypoints_;
  Mat*                   descriptors_;
  Pyramid*   evolution_;
  AKAZEOptions*              options_;

  Mat descriptorSamples_;  // List of positions in the grids to sample LDB bits from.
  Mat descriptorBits_;
};

/**
 * @brief This method  computes the set of descriptors through the nonlinear scale space
 * @param kpts Vector of detected keypoints
 * @param desc Matrix to store the descriptors
 */
void AKAZEFeatures::Compute_Descriptors(std::vector<KeyPoint>& kpts, OutputArray descriptors)
{
  CV_INSTRUMENT_REGION();

  for(size_t i = 0; i < kpts.size(); i++)
  {
      CV_Assert(0 <= kpts[i].class_id && kpts[i].class_id < static_cast<int>(evolution_.size()));
  }

  // Allocate memory for the matrix with the descriptors
  int descriptor_size = 64;
  int descriptor_type = CV_32FC1;
  if (options_.descriptor >= AKAZE::DESCRIPTOR_MLDB_UPRIGHT)
  {
    int descriptor_bits = (options_.descriptor_size == 0)
          ? (6 + 36 + 120)*options_.descriptor_channels // the full length binary descriptor -> 486 bits
          : options_.descriptor_size; // the random bit selection length binary descriptor
    descriptor_size = divUp(descriptor_bits, 8);
    descriptor_type = CV_8UC1;
  }
  descriptors.create((int)kpts.size(), descriptor_size, descriptor_type);

  Mat desc = descriptors.getMat();

  switch (options_.descriptor)
  {
    case AKAZE::DESCRIPTOR_KAZE_UPRIGHT: // Upright descriptors, not invariant to rotation
    {
      parallel_for_(Range(0, (int)kpts.size()), MSURF_Upright_Descriptor_64_Invoker(kpts, desc, evolution_));
    }
    break;
    case AKAZE::DESCRIPTOR_KAZE:
    {
      parallel_for_(Range(0, (int)kpts.size()), MSURF_Descriptor_64_Invoker(kpts, desc, evolution_));
    }
    break;
    case AKAZE::DESCRIPTOR_MLDB_UPRIGHT: // Upright descriptors, not invariant to rotation
    {
      if (options_.descriptor_size == 0)
        parallel_for_(Range(0, (int)kpts.size()), Upright_MLDB_Full_Descriptor_Invoker(kpts, desc, evolution_, options_));
      else
        parallel_for_(Range(0, (int)kpts.size()), Upright_MLDB_Descriptor_Subset_Invoker(kpts, desc, evolution_, options_, descriptorSamples_, descriptorBits_));
    }
    break;
    case AKAZE::DESCRIPTOR_MLDB:
    {
      if (options_.descriptor_size == 0)
        parallel_for_(Range(0, (int)kpts.size()), MLDB_Full_Descriptor_Invoker(kpts, desc, evolution_, options_));
      else
        parallel_for_(Range(0, (int)kpts.size()), MLDB_Descriptor_Subset_Invoker(kpts, desc, evolution_, options_, descriptorSamples_, descriptorBits_));
    }
    break;
  }
}

/* ************************************************************************* */
/**
 * @brief This function samples the derivative responses Lx and Ly for the points
 * within the radius of 6*scale from (x0, y0), then multiply 2D Gaussian weight
 * @param Lx Horizontal derivative
 * @param Ly Vertical derivative
 * @param x0 X-coordinate of the center point
 * @param y0 Y-coordinate of the center point
 * @param scale The sampling step
 * @param resX Output array of the weighted horizontal derivative responses
 * @param resY Output array of the weighted vertical derivative responses
 */
static inline
void Sample_Derivative_Response_Radius6(const Mat &Lx, const Mat &Ly,
                                  const int x0, const int y0, const int scale,
                                  float *resX, float *resY)
{
    /* ************************************************************************* */
    /// Lookup table for 2d gaussian (sigma = 2.5) where (0,0) is top left and (6,6) is bottom right
    static const float gauss25[7][7] =
    {
        { 0.02546481f, 0.02350698f, 0.01849125f, 0.01239505f, 0.00708017f, 0.00344629f, 0.00142946f },
        { 0.02350698f, 0.02169968f, 0.01706957f, 0.01144208f, 0.00653582f, 0.00318132f, 0.00131956f },
        { 0.01849125f, 0.01706957f, 0.01342740f, 0.00900066f, 0.00514126f, 0.00250252f, 0.00103800f },
        { 0.01239505f, 0.01144208f, 0.00900066f, 0.00603332f, 0.00344629f, 0.00167749f, 0.00069579f },
        { 0.00708017f, 0.00653582f, 0.00514126f, 0.00344629f, 0.00196855f, 0.00095820f, 0.00039744f },
        { 0.00344629f, 0.00318132f, 0.00250252f, 0.00167749f, 0.00095820f, 0.00046640f, 0.00019346f },
        { 0.00142946f, 0.00131956f, 0.00103800f, 0.00069579f, 0.00039744f, 0.00019346f, 0.00008024f }
    };
    static const struct gtable
    {
      float weight[109];
      int xidx[109];
      int yidx[109];

      explicit gtable(void)
      {
        // Generate the weight and indices by one-time initialization
        int k = 0;
        for (int i = -6; i <= 6; ++i) {
          for (int j = -6; j <= 6; ++j) {
            if (i*i + j*j < 36) {
              CV_Assert(k < 109);
              weight[k] = gauss25[abs(i)][abs(j)];
              yidx[k] = i;
              xidx[k] = j;
              ++k;
            }
          }
        }
      }
    } g;

  CV_Assert(x0 - 6 * scale >= 0 && x0 + 6 * scale < Lx.cols);
  CV_Assert(y0 - 6 * scale >= 0 && y0 + 6 * scale < Lx.rows);

  for (int i = 0; i < 109; i++)
  {
    int y = y0 + g.yidx[i] * scale;
    int x = x0 + g.xidx[i] * scale;

    float w = g.weight[i];
    resX[i] = w * Lx.at<float>(y, x);
    resY[i] = w * Ly.at<float>(y, x);
  }
}

/**
 * @brief This function sorts a[] by quantized float values
 * @param a[] Input floating point array to sort
 * @param n The length of a[]
 * @param quantum The interval to convert a[i]'s float values to integers
 * @param nkeys a[i] < nkeys * quantum
 * @param idx[] Output array of the indices: a[idx[i]] forms a sorted array
 * @param cum[] Output array of the starting indices of quantized floats
 * @note The values of a[] in [k*quantum, (k + 1)*quantum) is labeled by
 * the integer k, which is calculated by floor(a[i]/quantum).  After sorting,
 * the values from a[idx[cum[k]]] to a[idx[cum[k+1]-1]] are all labeled by k.
 * This sorting is unstable to reduce the memory access.
 */
static inline
void quantized_counting_sort(const float a[], const int n,
                             const float quantum, const int nkeys,
                             int idx[/*n*/], int cum[/*nkeys + 1*/])
{
  CV_Assert(nkeys > 0);
  memset(cum, 0, sizeof(cum[0]) * (nkeys + 1));

  // Count up the quantized values
  for (int i = 0; i < n; i++)
  {
    int b = (int)(a[i] / quantum);
    if (b < 0 || b >= nkeys)
      b = 0;
    cum[b]++;
  }

  // Compute the inclusive prefix sum i.e. the end indices; cum[nkeys] is the total
  for (int i = 1; i <= nkeys; i++)
  {
    cum[i] += cum[i - 1];
  }
  CV_Assert(cum[nkeys] == n);

  // Generate the sorted indices; cum[] becomes the exclusive prefix sum i.e. the start indices of keys
  for (int i = 0; i < n; i++)
  {
    int b = (int)(a[i] / quantum);
    if (b < 0 || b >= nkeys)
      b = 0;
    idx[--cum[b]] = i;
  }
}

/**
 * @brief This function computes the main orientation for a given keypoint
 * @param kpt Input keypoint
 * @note The orientation is computed using a similar approach as described in the
 * original SURF method. See Bay et al., Speeded Up Robust Features, ECCV 2006
 */
static inline
void Compute_Main_Orientation(KeyPoint& kpt, const Pyramid& evolution)
{
  // get the right evolution level for this keypoint
  const MEvolution& e = evolution[kpt.class_id];
  // Get the information from the keypoint
  int scale = cvRound(0.5f * kpt.size / e.octave_ratio);
  int x0 = cvRound(kpt.pt.x / e.octave_ratio);
  int y0 = cvRound(kpt.pt.y / e.octave_ratio);

  // Sample derivatives responses for the points within radius of 6*scale
  const int ang_size = 109;
  float resX[ang_size], resY[ang_size];
  Sample_Derivative_Response_Radius6(e.Lx, e.Ly, x0, y0, scale, resX, resY);

  // Compute the angle of each gradient vector
  float Ang[ang_size];
  hal::fastAtan2(resY, resX, Ang, ang_size, false);

  // Sort by the angles; angles are labeled by slices of 0.15 radian
  const int slices = 42;
  const float ang_step = (float)(2.0 * CV_PI / slices);
  int slice[slices + 1];
  int sorted_idx[ang_size];
  quantized_counting_sort(Ang, ang_size, ang_step, slices, sorted_idx, slice);

  // Find the main angle by sliding a window of 7-slice size(=PI/3) around the keypoint
  const int win = 7;

  float maxX = 0.0f, maxY = 0.0f;
  for (int i = slice[0]; i < slice[win]; i++) {
    const int idx = sorted_idx[i];
    maxX += resX[idx];
    maxY += resY[idx];
  }
  float maxNorm = maxX * maxX + maxY * maxY;

  for (int sn = 1; sn <= slices - win; sn++) {

    if (slice[sn] == slice[sn - 1] && slice[sn + win] == slice[sn + win - 1])
      continue;  // The contents of the window didn't change; don't repeat the computation

    float sumX = 0.0f, sumY = 0.0f;
    for (int i = slice[sn]; i < slice[sn + win]; i++) {
      const int idx = sorted_idx[i];
      sumX += resX[idx];
      sumY += resY[idx];
    }

    float norm = sumX * sumX + sumY * sumY;
    if (norm > maxNorm)
        maxNorm = norm, maxX = sumX, maxY = sumY;  // Found bigger one; update
  }

  for (int sn = slices - win + 1; sn < slices; sn++) {
    int remain = sn + win - slices;

    if (slice[sn] == slice[sn - 1] && slice[remain] == slice[remain - 1])
      continue;

    float sumX = 0.0f, sumY = 0.0f;
    for (int i = slice[sn]; i < slice[slices]; i++) {
      const int idx = sorted_idx[i];
      sumX += resX[idx];
      sumY += resY[idx];
    }
    for (int i = slice[0]; i < slice[remain]; i++) {
      const int idx = sorted_idx[i];
      sumX += resX[idx];
      sumY += resY[idx];
    }

    float norm = sumX * sumX + sumY * sumY;
    if (norm > maxNorm)
        maxNorm = norm, maxX = sumX, maxY = sumY;
  }

  // Store the final result
  kpt.angle = fastAtan2(maxY, maxX);
}

class ComputeKeypointOrientation : public ParallelLoopBody
{
public:
  ComputeKeypointOrientation(std::vector<KeyPoint>& kpts,
                             const Pyramid& evolution)
    : keypoints_(&kpts)
    , evolution_(&evolution)
  {
  }

  void operator() (const Range& range) const CV_OVERRIDE
  {
    for (int i = range.start; i < range.end; i++)
    {
      Compute_Main_Orientation((*keypoints_)[i], *evolution_);
    }
  }
private:
  std::vector<KeyPoint>* keypoints_;
  const Pyramid* evolution_;
};

/**
 * @brief This method computes the main orientation for a given keypoints
 * @param kpts Input keypoints
 */
void AKAZEFeatures::Compute_Keypoints_Orientation(std::vector<KeyPoint>& kpts) const
{
  CV_INSTRUMENT_REGION();

  parallel_for_(Range(0, (int)kpts.size()), ComputeKeypointOrientation(kpts, evolution_));
}

/* ************************************************************************* */
/**
 * @brief This method computes the upright descriptor (not rotation invariant) of
 * the provided keypoint
 * @param kpt Input keypoint
 * @param desc Descriptor vector
 * @note Rectangular grid of 24 s x 24 s. Descriptor Length 64. The descriptor is inspired
 * from Agrawal et al., CenSurE: Center Surround Extremas for Realtime Feature Detection and Matching,
 * ECCV 2008
 */
void MSURF_Upright_Descriptor_64_Invoker::Get_MSURF_Upright_Descriptor_64(const KeyPoint& kpt, float *desc, int desc_size) const {

  const int dsize = 64;
  CV_Assert(desc_size == dsize);

  float dx = 0.0, dy = 0.0, mdx = 0.0, mdy = 0.0, gauss_s1 = 0.0, gauss_s2 = 0.0;
  float rx = 0.0, ry = 0.0, len = 0.0, xf = 0.0, yf = 0.0, ys = 0.0, xs = 0.0;
  float sample_x = 0.0, sample_y = 0.0;
  int x1 = 0, y1 = 0, sample_step = 0, pattern_size = 0;
  int x2 = 0, y2 = 0, kx = 0, ky = 0, i = 0, j = 0, dcount = 0;
  float fx = 0.0, fy = 0.0, ratio = 0.0, res1 = 0.0, res2 = 0.0, res3 = 0.0, res4 = 0.0;
  int scale = 0;

  // Subregion centers for the 4x4 gaussian weighting
  float cx = -0.5f, cy = 0.5f;

  const Pyramid& evolution = *evolution_;

  // Set the descriptor size and the sample and pattern sizes
  sample_step = 5;
  pattern_size = 12;

  // Get the information from the keypoint
  ratio = (float)(1 << kpt.octave);
  scale = cvRound(0.5f*kpt.size / ratio);
  const int level = kpt.class_id;
  const Mat Lx = evolution[level].Lx;
  const Mat Ly = evolution[level].Ly;
  yf = kpt.pt.y / ratio;
  xf = kpt.pt.x / ratio;

  i = -8;

  // Calculate descriptor for this interest point
  // Area of size 24 s x 24 s
  while (i < pattern_size) {
    j = -8;
    i = i - 4;

    cx += 1.0f;
    cy = -0.5f;

    while (j < pattern_size) {
      dx = dy = mdx = mdy = 0.0;
      cy += 1.0f;
      j = j - 4;

      ky = i + sample_step;
      kx = j + sample_step;

      ys = yf + (ky*scale);
      xs = xf + (kx*scale);

      for (int k = i; k < i + 9; k++) {
        for (int l = j; l < j + 9; l++) {
          sample_y = k*scale + yf;
          sample_x = l*scale + xf;

          //Get the gaussian weighted x and y responses
          gauss_s1 = gaussian(xs - sample_x, ys - sample_y, 2.50f*scale);

          y1 = cvFloor(sample_y);
          x1 = cvFloor(sample_x);

          y2 = y1 + 1;
          x2 = x1 + 1;

          if (x1 < 0 || y1 < 0 || x2 >= Lx.cols || y2 >= Lx.rows)
              continue; // FIXIT Boundaries

          fx = sample_x - x1;
          fy = sample_y - y1;

          res1 = Lx.at<float>(y1, x1);
          res2 = Lx.at<float>(y1, x2);
          res3 = Lx.at<float>(y2, x1);
          res4 = Lx.at<float>(y2, x2);
          rx = (1.0f - fx)*(1.0f - fy)*res1 + fx*(1.0f - fy)*res2 + (1.0f - fx)*fy*res3 + fx*fy*res4;

          res1 = Ly.at<float>(y1, x1);
          res2 = Ly.at<float>(y1, x2);
          res3 = Ly.at<float>(y2, x1);
          res4 = Ly.at<float>(y2, x2);
          ry = (1.0f - fx)*(1.0f - fy)*res1 + fx*(1.0f - fy)*res2 + (1.0f - fx)*fy*res3 + fx*fy*res4;

          rx = gauss_s1*rx;
          ry = gauss_s1*ry;

          // Sum the derivatives to the cumulative descriptor
          dx += rx;
          dy += ry;
          mdx += fabs(rx);
          mdy += fabs(ry);
        }
      }

      // Add the values to the descriptor vector
      gauss_s2 = gaussian(cx - 2.0f, cy - 2.0f, 1.5f);

      desc[dcount++] = dx*gauss_s2;
      desc[dcount++] = dy*gauss_s2;
      desc[dcount++] = mdx*gauss_s2;
      desc[dcount++] = mdy*gauss_s2;

      len += (dx*dx + dy*dy + mdx*mdx + mdy*mdy)*gauss_s2*gauss_s2;

      j += 9;
    }

    i += 9;
  }

  CV_Assert(dcount == desc_size);

  // convert to unit vector
  len = sqrt(len);

  const float len_inv = 1.0f / len;
  for (i = 0; i < dsize; i++) {
    desc[i] *= len_inv;
  }
}

/* ************************************************************************* */
/**
 * @brief This method computes the descriptor of the provided keypoint given the
 * main orientation of the keypoint
 * @param kpt Input keypoint
 * @param desc Descriptor vector
 * @note Rectangular grid of 24 s x 24 s. Descriptor Length 64. The descriptor is inspired
 * from Agrawal et al., CenSurE: Center Surround Extremas for Realtime Feature Detection and Matching,
 * ECCV 2008
 */
void MSURF_Descriptor_64_Invoker::Get_MSURF_Descriptor_64(const KeyPoint& kpt, float *desc, int desc_size) const {

  const int dsize = 64;
  CV_Assert(desc_size == dsize);

  float dx = 0.0, dy = 0.0, mdx = 0.0, mdy = 0.0, gauss_s1 = 0.0, gauss_s2 = 0.0;
  float rx = 0.0, ry = 0.0, rrx = 0.0, rry = 0.0, len = 0.0, xf = 0.0, yf = 0.0, ys = 0.0, xs = 0.0;
  float sample_x = 0.0, sample_y = 0.0, co = 0.0, si = 0.0, angle = 0.0;
  float fx = 0.0, fy = 0.0, ratio = 0.0, res1 = 0.0, res2 = 0.0, res3 = 0.0, res4 = 0.0;
  int x1 = 0, y1 = 0, x2 = 0, y2 = 0, sample_step = 0, pattern_size = 0;
  int kx = 0, ky = 0, i = 0, j = 0, dcount = 0;
  int scale = 0;

  // Subregion centers for the 4x4 gaussian weighting
  float cx = -0.5f, cy = 0.5f;

  const Pyramid& evolution = *evolution_;

  // Set the descriptor size and the sample and pattern sizes
  sample_step = 5;
  pattern_size = 12;

  // Get the information from the keypoint
  ratio = (float)(1 << kpt.octave);
  scale = cvRound(0.5f*kpt.size / ratio);
  angle = kpt.angle * static_cast<float>(CV_PI / 180.f);
  const int level = kpt.class_id;
  const Mat Lx = evolution[level].Lx;
  const Mat Ly = evolution[level].Ly;
  yf = kpt.pt.y / ratio;
  xf = kpt.pt.x / ratio;
  co = cos(angle);
  si = sin(angle);

  i = -8;

  // Calculate descriptor for this interest point
  // Area of size 24 s x 24 s
  while (i < pattern_size) {
    j = -8;
    i = i - 4;

    cx += 1.0f;
    cy = -0.5f;

    while (j < pattern_size) {
      dx = dy = mdx = mdy = 0.0;
      cy += 1.0f;
      j = j - 4;

      ky = i + sample_step;
      kx = j + sample_step;

      xs = xf + (-kx*scale*si + ky*scale*co);
      ys = yf + (kx*scale*co + ky*scale*si);

      for (int k = i; k < i + 9; ++k) {
        for (int l = j; l < j + 9; ++l) {
          // Get coords of sample point on the rotated axis
          sample_y = yf + (l*scale*co + k*scale*si);
          sample_x = xf + (-l*scale*si + k*scale*co);

          // Get the gaussian weighted x and y responses
          gauss_s1 = gaussian(xs - sample_x, ys - sample_y, 2.5f*scale);

          y1 = cvFloor(sample_y);
          x1 = cvFloor(sample_x);

          y2 = y1 + 1;
          x2 = x1 + 1;

          if (x1 < 0 || y1 < 0 || x2 >= Lx.cols || y2 >= Lx.rows)
              continue; // FIXIT Boundaries

          fx = sample_x - x1;
          fy = sample_y - y1;

          res1 = Lx.at<float>(y1, x1);
          res2 = Lx.at<float>(y1, x2);
          res3 = Lx.at<float>(y2, x1);
          res4 = Lx.at<float>(y2, x2);
          rx = (1.0f - fx)*(1.0f - fy)*res1 + fx*(1.0f - fy)*res2 + (1.0f - fx)*fy*res3 + fx*fy*res4;

          res1 = Ly.at<float>(y1, x1);
          res2 = Ly.at<float>(y1, x2);
          res3 = Ly.at<float>(y2, x1);
          res4 = Ly.at<float>(y2, x2);
          ry = (1.0f - fx)*(1.0f - fy)*res1 + fx*(1.0f - fy)*res2 + (1.0f - fx)*fy*res3 + fx*fy*res4;

          // Get the x and y derivatives on the rotated axis
          rry = gauss_s1*(rx*co + ry*si);
          rrx = gauss_s1*(-rx*si + ry*co);

          // Sum the derivatives to the cumulative descriptor
          dx += rrx;
          dy += rry;
          mdx += fabs(rrx);
          mdy += fabs(rry);
        }
      }

      // Add the values to the descriptor vector
      gauss_s2 = gaussian(cx - 2.0f, cy - 2.0f, 1.5f);
      desc[dcount++] = dx*gauss_s2;
      desc[dcount++] = dy*gauss_s2;
      desc[dcount++] = mdx*gauss_s2;
      desc[dcount++] = mdy*gauss_s2;

      len += (dx*dx + dy*dy + mdx*mdx + mdy*mdy)*gauss_s2*gauss_s2;

      j += 9;
    }

    i += 9;
  }

  CV_Assert(dcount == desc_size);

  // convert to unit vector
  len = sqrt(len);

  const float len_inv = 1.0f / len;
  for (i = 0; i < dsize; i++) {
    desc[i] *= len_inv;
  }
}

/* ************************************************************************* */
/**
 * @brief This method computes the rupright descriptor (not rotation invariant) of
 * the provided keypoint
 * @param kpt Input keypoint
 * @param desc Descriptor vector
 */
void Upright_MLDB_Full_Descriptor_Invoker::Get_Upright_MLDB_Full_Descriptor(const KeyPoint& kpt, unsigned char *desc, int desc_size) const {

  const AKAZEOptions & options = *options_;
  const Pyramid& evolution = *evolution_;

  // Buffer for the M-LDB descriptor
  const int max_channels = 3;
  CV_Assert(options.descriptor_channels <= max_channels);
  float values[16*max_channels];

  // Get the information from the keypoint
  const float ratio = (float)(1 << kpt.octave);
  const int scale = cvRound(0.5f*kpt.size / ratio);
  const int level = kpt.class_id;
  const Mat Lx = evolution[level].Lx;
  const Mat Ly = evolution[level].Ly;
  const Mat Lt = evolution[level].Lt;
  const float yf = kpt.pt.y / ratio;
  const float xf = kpt.pt.x / ratio;

  // For 2x2 grid, 3x3 grid and 4x4 grid
  const int pattern_size = options_->descriptor_pattern_size;
  CV_Assert((pattern_size & 1) == 0);
  const int sample_step[3] = {
    pattern_size,
    divUp(pattern_size * 2, 3),
    divUp(pattern_size, 2)
  };

  memset(desc, 0, desc_size);

  // For the three grids
  int dcount1 = 0;
  for (int z = 0; z < 3; z++) {
    int dcount2 = 0;
    const int step = sample_step[z];
    for (int i = -pattern_size; i < pattern_size; i += step) {
      for (int j = -pattern_size; j < pattern_size; j += step) {
        float di = 0.0, dx = 0.0, dy = 0.0;

        int nsamples = 0;
        for (int k = 0; k < step; k++) {
          for (int l = 0; l < step; l++) {

            // Get the coordinates of the sample point
            const float sample_y = yf + (l+j)*scale;
            const float sample_x = xf + (k+i)*scale;

            const int y1 = cvRound(sample_y);
            const int x1 = cvRound(sample_x);

            if (y1 < 0 || y1 >= Lt.rows || x1 < 0 || x1 >= Lt.cols)
                continue; // Boundaries

            const float ri = Lt.at<float>(y1, x1);
            const float rx = Lx.at<float>(y1, x1);
            const float ry = Ly.at<float>(y1, x1);

            di += ri;
            dx += rx;
            dy += ry;
            nsamples++;
          }
        }

        if (nsamples > 0)
        {
            const float nsamples_inv = 1.0f / nsamples;
            di *= nsamples_inv;
            dx *= nsamples_inv;
            dy *= nsamples_inv;
        }

        float *val = &values[dcount2*max_channels];
        *(val) = di;
        *(val+1) = dx;
        *(val+2) = dy;
        dcount2++;
      }
    }

    // Do binary comparison
    const int num = (z + 2) * (z + 2);
    for (int i = 0; i < num; i++) {
      for (int j = i + 1; j < num; j++) {
        const float * valI = &values[i*max_channels];
        const float * valJ = &values[j*max_channels];
        for (int k = 0; k < 3; ++k) {
          if (*(valI + k) > *(valJ + k)) {
            desc[dcount1 / 8] |= (1 << (dcount1 % 8));
          }
          dcount1++;
        }
      }
    }

  } // for (int z = 0; z < 3; z++)

  CV_Assert(dcount1 <= desc_size*8);
  CV_Assert(divUp(dcount1, 8) == desc_size);
}

/**
 * @brief Samples the rotated pattern grid of a keypoint once for all three M-LDB grids
 * @details The 2x2, 3x3 and 4x4 grids all sample the same rotated points (k, l) with
 * k, l in [grid_min, grid_min + grid_size). For each point this stores the channel
 * values (Lt, then the rotated derivatives or the gradient magnitude) at
 * grid[((k - grid_min) * grid_size + (l - grid_min)) * chan], and whether it is inside the image.
 */
void MLDB_Full_Descriptor_Invoker::MLDB_Sample_Grid(float* grid, uchar* valid, int grid_min, int grid_size,
                                                    const int level, float xf, float yf, float co, float si,
                                                    float scale) const
{
    const Pyramid& evolution = *evolution_;
    int chan = options_->descriptor_channels;
    const Mat Lx = evolution[level].Lx;
    const Mat Ly = evolution[level].Ly;
    const Mat Lt = evolution[level].Lt;

    const Size size = Lt.size();
    CV_Assert(size == Lx.size());
    CV_Assert(size == Ly.size());

    int pos = 0;
    for (int k = grid_min; k < grid_min + grid_size; k++) {
        for (int l = grid_min; l < grid_min + grid_size; l++, pos++) {
            float sample_y = yf + (l*co * scale + k*si*scale);
            float sample_x = xf + (-l*si * scale + k*co*scale);

            int y1 = cvRound(sample_y);
            int x1 = cvRound(sample_x);

            valid[pos] = !(y1 < 0 || y1 >= Lt.rows || x1 < 0 || x1 >= Lt.cols);
            if (!valid[pos])
                continue; // Boundaries

            float *g = grid + pos * chan;
            g[0] = Lt.at<float>(y1, x1);

            if(chan > 1) {
                float rx = Lx.at<float>(y1, x1);
                float ry = Ly.at<float>(y1, x1);
                if (chan == 2) {
                  g[1] = sqrtf(rx*rx + ry*ry);
                }
                else {
                  float rry = rx*co + ry*si;
                  float rrx = -rx*si + ry*co;
                  g[1] = rrx;
                  g[2] = rry;
                }
            }
        }
    }
}

void MLDB_Full_Descriptor_Invoker::MLDB_Fill_Values(float* values, int sample_step, const float* grid,
                                                    const uchar* valid, int grid_min, int grid_size) const
{
    int pattern_size = options_->descriptor_pattern_size;
    int chan = options_->descriptor_channels;

    int valpos = 0;
    for (int i = -pattern_size; i < pattern_size; i += sample_step) {
        for (int j = -pattern_size; j < pattern_size; j += sample_step) {
            float di = 0.0f, dx = 0.0f, dy = 0.0f;

            int nsamples = 0;
            for (int k = i; k < i + sample_step; k++) {
              const int row = (k - grid_min) * grid_size - grid_min;
              for (int l = j; l < j + sample_step; l++) {
                const int pos = row + l;
                if (!valid[pos])
                    continue; // Boundaries

                const float *g = grid + pos * chan;
                di += g[0];
                if (chan > 1) {
                    dx += g[1];
                    if (chan > 2)
                      dy += g[2];
                }
                nsamples++;
              }
            }

            if (nsamples > 0)
            {
                const float nsamples_inv = 1.0f / nsamples;
                di *= nsamples_inv;
                dx *= nsamples_inv;
                dy *= nsamples_inv;
            }

            values[valpos] = di;
            if (chan > 1) {
                values[valpos + 1] = dx;
            }
            if (chan > 2) {
                values[valpos + 2] = dy;
            }
            valpos += chan;
        }
    }
}

void MLDB_Full_Descriptor_Invoker::MLDB_Binary_Comparisons(float* values, unsigned char* desc,
                                                           int count, int& dpos) const {
    int chan = options_->descriptor_channels;
    int* ivalues = (int*) values;
    for(int i = 0; i < count * chan; i++) {
        ivalues[i] = CV_TOGGLE_FLT(ivalues[i]);
    }

    // bits are accumulated in a register and OR-ed to desc 64 at a time;
    // bit dpos goes to desc[dpos >> 3] at (dpos & 7), as one bit at a time
    int base = dpos & ~7;
    int nbits = dpos - base;
    uint64 acc = 0;
    for(int pos = 0; pos < chan; pos++) {
        for (int i = 0; i < count; i++) {
            int ival = ivalues[chan * i + pos];
            for (int j = i + 1; j < count; j++) {
                acc |= (uint64)(ival > ivalues[chan * j + pos]) << nbits;
                if (++nbits == 64) {
                    for (int b = 0; b < 8; b++)
                        desc[(base >> 3) + b] |= (uchar)(acc >> (8 * b));
                    base += 64;
                    nbits = 0;
                    acc = 0;
                }
            }
        }
    }
    for (int b = 0; b < (nbits + 7) / 8; b++)
        desc[(base >> 3) + b] |= (uchar)(acc >> (8 * b));
    dpos = base + nbits;
}

/* ************************************************************************* */
/**
 * @brief This method computes the descriptor of the provided keypoint given the
 * main orientation of the keypoint
 * @param kpt Input keypoint
 * @param desc Descriptor vector
 */
void MLDB_Full_Descriptor_Invoker::Get_MLDB_Full_Descriptor(const KeyPoint& kpt, unsigned char *desc, int desc_size) const {

  const int max_channels = 3;
  CV_Assert(options_->descriptor_channels <= max_channels);
  const int pattern_size = options_->descriptor_pattern_size;

  float values[16*max_channels];
  CV_Assert((pattern_size & 1) == 0);
  //const double size_mult[3] = {1, 2.0/3.0, 1.0/2.0};
  const int sample_step[3] = { // static_cast<int>(ceil(pattern_size * size_mult[lvl]))
    pattern_size,
    divUp(pattern_size * 2, 3),
    divUp(pattern_size, 2)
  };

  float ratio = (float)(1 << kpt.octave);
  float scale = (float)cvRound(0.5f*kpt.size / ratio);
  float xf = kpt.pt.x / ratio;
  float yf = kpt.pt.y / ratio;
  float angle = kpt.angle * static_cast<float>(CV_PI / 180.f);
  float co = cos(angle);
  float si = sin(angle);

  memset(desc, 0, desc_size);

  // all the grids sample points of the same rotated lattice: sample it once
  int grid_max = pattern_size - 1;
  for (int lvl = 0; lvl < 3; lvl++) {
      int last = -pattern_size;
      while (last + sample_step[lvl] < pattern_size)
          last += sample_step[lvl];
      grid_max = std::max(grid_max, last + sample_step[lvl] - 1);
  }
  const int grid_min = -pattern_size;
  const int grid_size = grid_max - grid_min + 1;
  AutoBuffer<float, 32 * 32 * max_channels> grid(grid_size * grid_size * max_channels);
  AutoBuffer<uchar, 32 * 32> valid(grid_size * grid_size);
  MLDB_Sample_Grid(grid.data(), valid.data(), grid_min, grid_size, kpt.class_id, xf, yf, co, si, scale);

  int dpos = 0;
  for(int lvl = 0; lvl < 3; lvl++)
  {
      int val_count = (lvl + 2) * (lvl + 2);
      MLDB_Fill_Values(values, sample_step[lvl], grid.data(), valid.data(), grid_min, grid_size);
      MLDB_Binary_Comparisons(values, desc, val_count, dpos);
  }

  CV_Assert(dpos == 486);
  CV_Assert(divUp(dpos, 8) == desc_size);
}

/* ************************************************************************* */
/**
 * @brief This method computes the M-LDB descriptor of the provided keypoint given the
 * main orientation of the keypoint. The descriptor is computed based on a subset of
 * the bits of the whole descriptor
 * @param kpt Input keypoint
 * @param desc Descriptor vector
 */
void MLDB_Descriptor_Subset_Invoker::Get_MLDB_Descriptor_Subset(const KeyPoint& kpt, unsigned char *desc, int desc_size) const {

  float rx = 0.f, ry = 0.f;
  float sample_x = 0.f, sample_y = 0.f;

  const AKAZEOptions & options = *options_;
  const Pyramid& evolution = *evolution_;

  // Get the information from the keypoint
  float ratio = (float)(1 << kpt.octave);
  int scale = cvRound(0.5f*kpt.size / ratio);
  float angle = kpt.angle * static_cast<float>(CV_PI / 180.f);
  const int level = kpt.class_id;
  const Mat Lx = evolution[level].Lx;
  const Mat Ly = evolution[level].Ly;
  const Mat Lt = evolution[level].Lt;
  float yf = kpt.pt.y / ratio;
  float xf = kpt.pt.x / ratio;
  float co = cos(angle);
  float si = sin(angle);

  // Allocate memory for the matrix of values
  // Buffer for the M-LDB descriptor
  const int max_channels = 3;
  const int channels = options.descriptor_channels;
  CV_Assert(channels <= max_channels);
  float values[(4 + 9 + 16)*max_channels] = { 0 };

  // Sample everything, but only do the comparisons
  const int pattern_size = options.descriptor_pattern_size;
  CV_Assert((pattern_size & 1) == 0);
  const int sample_steps[3] = {
    pattern_size,
    divUp(pattern_size * 2, 3),
    divUp(pattern_size, 2)
  };

  for (int i = 0; i < descriptorSamples_.rows; i++) {
    const int *coords = descriptorSamples_.ptr<int>(i);
    CV_Assert(coords[0] >= 0 && coords[0] < 3);
    const int sample_step = sample_steps[coords[0]];
    float di = 0.f, dx = 0.f, dy = 0.f;

    for (int k = coords[1]; k < coords[1] + sample_step; k++) {
      for (int l = coords[2]; l < coords[2] + sample_step; l++) {

        // Get the coordinates of the sample point
        sample_y = yf + (l*scale*co + k*scale*si);
        sample_x = xf + (-l*scale*si + k*scale*co);

        const int y1 = cvRound(sample_y);
        const int x1 = cvRound(sample_x);

        if (x1 < 0 || y1 < 0 || x1 >= Lt.cols || y1 >= Lt.rows)
            continue; // Boundaries

        di += Lt.at<float>(y1, x1);

        if (options.descriptor_channels > 1) {
          rx = Lx.at<float>(y1, x1);
          ry = Ly.at<float>(y1, x1);

          if (options.descriptor_channels == 2) {
            dx += sqrtf(rx*rx + ry*ry);
          }
          else if (options.descriptor_channels == 3) {
            // Get the x and y derivatives on the rotated axis
            dx += rx*co + ry*si;
            dy += -rx*si + ry*co;
          }
        }
      }
    }

    float* pValues = &values[channels * i];
    pValues[0] = di;

    if (channels == 2) {
      pValues[1] = dx;
    }
    else if (channels == 3) {
      pValues[1] = dx;
      pValues[2] = dy;
    }
  }

  // Do the comparisons
  const int *comps = descriptorBits_.ptr<int>(0);

  CV_Assert(divUp(descriptorBits_.rows, 8) == desc_size);
  memset(desc, 0, desc_size);

  for (int i = 0; i<descriptorBits_.rows; i++) {
    if (values[comps[2 * i]] > values[comps[2 * i + 1]]) {
      desc[i / 8] |= (1 << (i % 8));
    }
  }
}

/* ************************************************************************* */
/**
 * @brief This method computes the upright (not rotation invariant) M-LDB descriptor
 * of the provided keypoint given the main orientation of the keypoint.
 * The descriptor is computed based on a subset of the bits of the whole descriptor
 * @param kpt Input keypoint
 * @param desc Descriptor vector
 */
void Upright_MLDB_Descriptor_Subset_Invoker::Get_Upright_MLDB_Descriptor_Subset(const KeyPoint& kpt, unsigned char *desc, int desc_size) const {

  float di = 0.0f, dx = 0.0f, dy = 0.0f;
  float rx = 0.0f, ry = 0.0f;
  float sample_x = 0.0f, sample_y = 0.0f;
  int x1 = 0, y1 = 0;

  const AKAZEOptions & options = *options_;
  const Pyramid& evolution = *evolution_;

  // Get the information from the keypoint
  float ratio = (float)(1 << kpt.octave);
  int scale = cvRound(0.5f*kpt.size / ratio);
  const int level = kpt.class_id;
  const Mat Lx = evolution[level].Lx;
  const Mat Ly = evolution[level].Ly;
  const Mat Lt = evolution[level].Lt;
  float yf = kpt.pt.y / ratio;
  float xf = kpt.pt.x / ratio;

  // Allocate memory for the matrix of values
  const int max_channels = 3;
  const int channels = options.descriptor_channels;
  CV_Assert(channels <= max_channels);
  float values[(4 + 9 + 16)*max_channels] = { 0 };

  const int pattern_size = options.descriptor_pattern_size;
  CV_Assert((pattern_size & 1) == 0);
  const int sample_steps[3] = {
    pattern_size,
    divUp(pattern_size * 2, 3),
    divUp(pattern_size, 2)
  };

  for (int i = 0; i < descriptorSamples_.rows; i++) {
    const int *coords = descriptorSamples_.ptr<int>(i);
    CV_Assert(coords[0] >= 0 && coords[0] < 3);
    int sample_step = sample_steps[coords[0]];
    di = 0.0f, dx = 0.0f, dy = 0.0f;

    for (int k = coords[1]; k < coords[1] + sample_step; k++) {
      for (int l = coords[2]; l < coords[2] + sample_step; l++) {

        // Get the coordinates of the sample point
        sample_y = yf + l*scale;
        sample_x = xf + k*scale;

        y1 = cvRound(sample_y);
        x1 = cvRound(sample_x);

        if (x1 < 0 || y1 < 0 || x1 >= Lt.cols || y1 >= Lt.rows)
            continue; // Boundaries

        di += Lt.at<float>(y1, x1);

        if (options.descriptor_channels > 1) {
          rx = Lx.at<float>(y1, x1);
          ry = Ly.at<float>(y1, x1);

          if (options.descriptor_channels == 2) {
            dx += sqrtf(rx*rx + ry*ry);
          }
          else if (options.descriptor_channels == 3) {
            dx += rx;
            dy += ry;
          }
        }
      }
    }

    float* pValues = &values[channels * i];
    pValues[0] = di;

    if (options.descriptor_channels == 2) {
      pValues[1] = dx;
    }
    else if (options.descriptor_channels == 3) {
      pValues[1] = dx;
      pValues[2] = dy;
    }
  }

  // Do the comparisons
  const int *comps = descriptorBits_.ptr<int>(0);

  CV_Assert(divUp(descriptorBits_.rows, 8) == desc_size);
  memset(desc, 0, desc_size);

  for (int i = 0; i<descriptorBits_.rows; i++) {
    if (values[comps[2 * i]] > values[comps[2 * i + 1]]) {
      desc[i / 8] |= (1 << (i % 8));
    }
  }
}

/* ************************************************************************* */
/**
 * @brief This function computes a (quasi-random) list of bits to be taken
 * from the full descriptor. To speed the extraction, the function creates
 * a list of the samples that are involved in generating at least a bit (sampleList)
 * and a list of the comparisons between those samples (comparisons)
 * @param sampleList
 * @param comparisons The matrix with the binary comparisons
 * @param nbits The number of bits of the descriptor
 * @param pattern_size The pattern size for the binary descriptor
 * @param nchannels Number of channels to consider in the descriptor (1-3)
 * @note The function keeps the 18 bits (3-channels by 6 comparisons) of the
 * coarser grid, since it provides the most robust estimations
 */
void generateDescriptorSubsample(Mat& sampleList, Mat& comparisons, int nbits,
                                 int pattern_size, int nchannels) {

  int ssz = 0;
  for (int i = 0; i < 3; i++) {
    int gz = (i + 2)*(i + 2);
    ssz += gz*(gz - 1) / 2;
  }
  ssz *= nchannels;

  CV_Assert(ssz == 162*nchannels);
  CV_Assert(nbits <= ssz && "Descriptor size can't be bigger than full descriptor (486 = 162*3 - 3 channels)");

  // Since the full descriptor is usually under 10k elements, we pick
  // the selection from the full matrix.  We take as many samples per
  // pick as the number of channels. For every pick, we
  // take the two samples involved and put them in the sampling list

  Mat_<int> fullM(ssz / nchannels, 5);
  for (int i = 0, c = 0; i < 3; i++) {
    int gdiv = i + 2; //grid divisions, per row
    int gsz = gdiv*gdiv;
    int psz = divUp(2*pattern_size, gdiv);

    for (int j = 0; j < gsz; j++) {
      for (int k = j + 1; k < gsz; k++, c++) {
        fullM(c, 0) = i;
        fullM(c, 1) = psz*(j % gdiv) - pattern_size;
        fullM(c, 2) = psz*(j / gdiv) - pattern_size;
        fullM(c, 3) = psz*(k % gdiv) - pattern_size;
        fullM(c, 4) = psz*(k / gdiv) - pattern_size;
      }
    }
  }

  RNG rng(1024);
  const int npicks = divUp(nbits, nchannels);
  Mat_<int> comps = Mat_<int>(nchannels * npicks, 2);
  comps = 1000;

  // Select some samples. A sample includes all channels
  int count = 0;
  Mat_<int> samples(29, 3);
  Mat_<int> fullcopy = fullM.clone();
  samples = -1;

  for (int i = 0; i < npicks; i++) {
    int k = rng(fullM.rows - i);
    if (i < 6) {
      // Force use of the coarser grid values and comparisons
      k = i;
    }

    bool n = true;

    for (int j = 0; j < count; j++) {
      if (samples(j, 0) == fullcopy(k, 0) && samples(j, 1) == fullcopy(k, 1) && samples(j, 2) == fullcopy(k, 2)) {
        n = false;
        comps(i*nchannels, 0) = nchannels*j;
        comps(i*nchannels + 1, 0) = nchannels*j + 1;
        comps(i*nchannels + 2, 0) = nchannels*j + 2;
        break;
      }
    }

    if (n) {
      samples(count, 0) = fullcopy(k, 0);
      samples(count, 1) = fullcopy(k, 1);
      samples(count, 2) = fullcopy(k, 2);
      comps(i*nchannels, 0) = nchannels*count;
      comps(i*nchannels + 1, 0) = nchannels*count + 1;
      comps(i*nchannels + 2, 0) = nchannels*count + 2;
      count++;
    }

    n = true;
    for (int j = 0; j < count; j++) {
      if (samples(j, 0) == fullcopy(k, 0) && samples(j, 1) == fullcopy(k, 3) && samples(j, 2) == fullcopy(k, 4)) {
        n = false;
        comps(i*nchannels, 1) = nchannels*j;
        comps(i*nchannels + 1, 1) = nchannels*j + 1;
        comps(i*nchannels + 2, 1) = nchannels*j + 2;
        break;
      }
    }

    if (n) {
      samples(count, 0) = fullcopy(k, 0);
      samples(count, 1) = fullcopy(k, 3);
      samples(count, 2) = fullcopy(k, 4);
      comps(i*nchannels, 1) = nchannels*count;
      comps(i*nchannels + 1, 1) = nchannels*count + 1;
      comps(i*nchannels + 2, 1) = nchannels*count + 2;
      count++;
    }

    Mat tmp = fullcopy.row(k);
    fullcopy.row(fullcopy.rows - i - 1).copyTo(tmp);
  }

  sampleList = samples.rowRange(0, count).clone();
  comparisons = comps.rowRange(0, nbits).clone();
}

}
}
