#ifndef CUDAMAT_CUH
#define CUDAMAT_CUH

#define ERROR_INCOMPATIBLE_DIMENSIONS -1
#define CUBLAS_ERROR -2
#define CUDA_ERROR -3
#define VIEW_ERROR -4
#define ERROR_TRANSPOSED -5
#define ERROR_GENERIC -6
#define ERROR_TRANSPOSEDNESS -7
#define ERROR_NOT_ON_DEVICE -8
#define ERROR_UNSUPPORTED -9

#ifndef DIVUP
#define DIVUP(x, y) (((x) + (y) - 1) / (y))
#endif
#ifndef MIN
#define MIN(x,y) ((x < y) ? x : y)
#endif
#ifndef MAX
#define MAX(x,y) ((x > y) ? x : y)
#endif

#include "cublas.h"
#ifdef __cplusplus
extern "C" {
#endif

struct cudamat {
    float* data_host;
    float* data_device;
    int on_device;
    int on_host;
    int size[2];
    int is_trans; // 0 or 1
    int owns_data;
    cudaTextureObject_t tex_obj;
};

struct rnd_struct {
    unsigned int* dev_mults;
    unsigned long long* dev_words;
};
// bounding boxes.
struct bbox {
    int *seg;   // array of length 'size' + 1.
    int *labels;  // labels[seg[i]:seg[i+1]] are the labels for image i.
    int *boxes;   // boxes[4*seg[i]:4*seg[i+1]] are bounding boxes for image i.
};

struct cudamat_bbox {
    bbox data_host;
    bbox data_device;
    int on_device;
    int on_host;
    int size;  // Number of images in the (mini)batch.
    int numboxes;  // Total number of boxes over all images in the (mini)batch.
};


struct sparse_data {
  int *indices, *indptr;
  float* data;
};

struct cudamat_sparse {
    sparse_data data_host;
    sparse_data data_device;
    int on_device;
    int on_host;
    int size[2];
    int is_trans; // 0 or 1
    int owns_data;
    int nnz;
};

const char* get_last_cuda_error();
int cuda_record_event(cudaEvent_t* t);
int cuda_synchronize_event(cudaEvent_t* t);
int cuda_create_event(cudaEvent_t* t);
int cublas_init();
int cublas_shutdown();
bool cuda_is_fermi(int deviceId);
int cuda_set_device(int deviceId);
int cuda_set_P2P(int gpu1, int gpu2);
int init_random(rnd_struct* rnd_state, int seed, const char* cudamatpath);
int get_rnd_state(rnd_struct* rnd_state, unsigned long long* host_words_out, int *size_out);
int get_leading_dimension(cudamat* mat);
int get_nonleading_dimension(cudamat* mat);
void set_transpose(cudamat* mat, int is_trans);
void cuda_sync_threads();
int allocate_device_memory(cudamat* mat);
int allocate_device_memory_bbox(cudamat_bbox* mat);
int allocate_device_memory_sparse(cudamat_sparse* mat);
int destroy_tex(cudamat* mat);
int copy_to_host(cudamat* mat);
int copy_to_host_slice(cudamat* mat, int start, int end);
int copy_bbox_to_host(cudamat_bbox* mat);
int copy_to_device(cudamat* mat);
int copy_to_device_slice(cudamat* mat, int start, int end);
int copy_bbox_to_device(cudamat_bbox* mat);
int copy_sparse_to_device(cudamat_sparse* mat);
int copy_on_device(cudamat* mat1, cudamat* mat2);
int copy_on_device_p2p_async(cudamat* src, cudamat* dst, int src_dev, int dst_dev);
int get_row_slice(cudamat* source, cudamat* target, unsigned int start, unsigned int end);
int set_row_slice(cudamat* source, cudamat* target, unsigned int start, unsigned int end);
int copy_transpose(cudamat* source, cudamat* target);
int copy_transpose_big_matrix(cudamat* source, cudamat* target);
int free_device_memory(cudamat* mat);
int free_device_memory_bbox(cudamat_bbox* mat);
int set_shape(cudamat* mat, unsigned int m, unsigned int n);
int reshape(cudamat* mat, int m, int n);
int get_slice(cudamat* source, cudamat* target, unsigned int first_col, unsigned int last_col);
int get_vector_slice(cudamat* source, cudamat* target, unsigned int first_ind, unsigned int last_ind);
void init_from_array(cudamat* mat, float* data, int m, int n);
void init_from_sparse_array(cudamat_sparse* mat, float* data, int* indices, int* indptr, int m, int n, int nnz);
void set_on_device(cudamat* mat);
int init_empty(cudamat* mat, int m, int n);
int fill_with_rand(rnd_struct* rnd_state, cudamat* mat);
int fill_with_randn(rnd_struct* rnd_state, cudamat* mat);
int sample_bernoulli(rnd_struct* rnd_state, cudamat* mat, cudamat* target);
int sample_bernoulli_tanh(rnd_struct* rnd_state, cudamat* mat, cudamat* target);
int sample_poisson(rnd_struct* rnd_state, cudamat* mat, cudamat* target);
int sample_gaussian(rnd_struct* rnd_state, cudamat* mat, cudamat* target, float mult);
int perturb_energy(rnd_struct* rnd_state, cudamat* mat, cudamat* target);
int perturb_prob(rnd_struct* rnd_state, cudamat* mat, cudamat* target);
int dropout(rnd_struct* rnd_state, cudamat* mat, float dropprob, float val, float scale);
int gaussian_dropout(rnd_struct* rnd_state, cudamat* mat, float scale);
int add_col_vec(cudamat* mat, cudamat* vec, cudamat* target);
int add_col_mult(cudamat* mat, cudamat* vec, cudamat* target, float mult);
int add_to_each_pixel(cudamat* mat1, cudamat* mat2, cudamat* target, float mult);
int mult_diagonal_scalar(cudamat* mat, float val, cudamat* target);
int add_diagonal_scalar(cudamat* mat, float val, cudamat* target);
int mult_diagonal(cudamat* mat, cudamat* vec, cudamat* target);
int add_diagonal(cudamat* mat, cudamat* vec, cudamat* target);
int add_row_mult(cudamat* mat, cudamat* vec, cudamat* target, float mult);
int add_row_vec(cudamat* mat, cudamat* vec, cudamat* target);
int mult_by_col_vec(cudamat* mat, cudamat* vec, cudamat* target);
int mult_by_row_vec(cudamat* mat, cudamat* vec, cudamat* target);
int div_by_col_vec(cudamat* mat, cudamat* vec, cudamat* target);
int div_by_row_vec(cudamat* mat, cudamat* vec, cudamat* target);
int less_than_eq(cudamat* mat1, cudamat* mat2, cudamat* target);
int less_than(cudamat* mat1, cudamat* mat2, cudamat* target);
int less_than_eq_scalar(cudamat* mat, float val, cudamat* target);
int less_than_scalar(cudamat* mat, float val, cudamat* target);
int greater_than_eq(cudamat* mat1, cudamat* mat2, cudamat* target);
int greater_than(cudamat* mat1, cudamat* mat2, cudamat* target);
int upper_bound(cudamat* mat1, cudamat* mat2, cudamat* target);
int lower_bound(cudamat* mat1, cudamat* mat2, cudamat* target);
int greater_than_eq_scalar(cudamat* mat, float val, cudamat* target);
int greater_than_scalar(cudamat* mat, float val, cudamat* target);
int upper_bound_scalar(cudamat* mat, float val, cudamat* target);
int lower_bound_scalar(cudamat* mat, float val, cudamat* target);
int upper_bound_mod_scalar(cudamat* mat, float val, cudamat* target);
int max_by_axis(cudamat* mat, cudamat* target, int axis);
int choose_max_and_accumulate(cudamat* mat, cudamat* acc);
int choose_max_by_axis(cudamat* mat, cudamat* target, int axis);
int argmax_by_axis(cudamat* mat, cudamat* target, int axis);
int sqsum_by_axis(cudamat* mat, cudamat* target, int axis, float mult, float p);
int normlimit_by_axis(cudamat* mat, cudamat* target, int axis, float norm, int constraint);
int sign(cudamat* mat, cudamat* target);
int apply_cos(cudamat* mat, cudamat* target);
int apply_sin(cudamat* mat, cudamat* target);
int apply_sigmoid(cudamat* mat, cudamat* target);
int apply_tanh(cudamat* mat, cudamat* target);
int apply_abs(cudamat* mat, cudamat* target);
int apply_log_1_plus_exp(cudamat* mat, cudamat* target);
int apply_relu_squash(cudamat* mat, cudamat* target, float lambda);
int apply_log(cudamat* mat, cudamat* target, float tiny);
int apply_exp(cudamat* mat, cudamat* target);
int apply_ceil(cudamat* mat, cudamat* target);
int apply_floor(cudamat* mat, cudamat* target);
int apply_sqrt(cudamat* mat, cudamat* target);
int apply_pow(cudamat* mat, float pow, cudamat* target);
int apply_pow_matrix(cudamat* mat, cudamat* pow, cudamat* target);
int compute_cross_entropy(cudamat* mat, cudamat* pow, cudamat* target, float tiny);
int compute_cross_entropy_bernoulli(cudamat* mat, cudamat* pow, cudamat* target, float tiny);
int correct_preds(cudamat* mat, cudamat* pow, cudamat* target, float cutoff);
int reciprocal(cudamat* mat, cudamat* target);
int dot(cudamat* mat1, cudamat* mat2, cudamat* target, float beta, float alpha);
int sparse_dot(cudamat_sparse* mat1, cudamat* mat2, cudamat* target, float beta, float alpha);
float vdot(cudamat* mat1, cudamat* mat2, int* err_code);
int add_mult(cudamat* mat1, cudamat* mat2, float alpha);
int add_mult_sign(cudamat* mat1, cudamat* mat2, float mult);
int add_elementwise(cudamat* mat1, cudamat* mat2, cudamat* target);
int subtract_elementwise(cudamat* mat1, cudamat* mat2, cudamat* target);
int divide_elementwise(cudamat* mat1, cudamat* mat2, cudamat* target);
int mult_elementwise(cudamat* mat1, cudamat* mat2, cudamat* target);
int apply_sin_deriv(cudamat* mat1, cudamat* mat2, cudamat* target);
int apply_cos_deriv(cudamat* mat1, cudamat* mat2, cudamat* target);
int apply_logistic_deriv(cudamat* mat1, cudamat* mat2, cudamat* target);
int apply_tanh_deriv(cudamat* mat1, cudamat* mat2, cudamat* target);
int apply_rectified_linear_deriv(cudamat* mat1, cudamat* mat2, cudamat* target);
int apply_rectified_linear_smooth_deriv(cudamat* mat1, cudamat* mat2, cudamat* target);
int assign_scalar(cudamat* mat, float alpha);
int mult_by_scalar(cudamat* mat, float alpha, cudamat* target);
int divide_by_scalar(cudamat* mat, float alpha, cudamat* target);
int add_scalar(cudamat* mat, float alpha, cudamat* target);
float euclid_norm(cudamat* mat, int* err_code);
int selectRows(cudamat* source, cudamat* target, cudamat* indices);
int swapColumns(cudamat* source, cudamat* target, cudamat* indices1, cudamat* indices2);
int shuffleColumns(cudamat* source, cudamat* rand_perm_indices);
int setSelectedRows(cudamat* target, cudamat* source, cudamat* indices);
int generate_translations_big_var_off(cudamat* source, cudamat* target, cudamat* off_x, cudamat* off_y, int source_w, int target_w, int num_channels);
int blockify(cudamat* source, cudamat* target, int blocksize);
int softmax(cudamat* mat, cudamat* target); 
int softmax_overwrite(cudamat* mat); 
int softmax_row_major(cudamat* mat); 
int softmax_row_major_multi(cudamat* mat, int numslices);
int apply_logistic_grad(cudamat* mat1, cudamat* mat2, cudamat* out_grad);
int apply_softmax_grad(cudamat* mat, cudamat* labels, cudamat* target); 
int apply_softmax_grad_CLS(cudamat* mat, cudamat_bbox* labels, cudamat* indices, cudamat* target); 
int apply_softmax_grad_row_major(cudamat* mat, cudamat* labels, cudamat* target); 
int get_softmax_correct(cudamat* mat, cudamat* labels, cudamat* target); 
int get_softmax_correct_row_major(cudamat* mat, cudamat* labels, cudamat* target); 
int get_softmax_correct_CLS(cudamat* mat, cudamat_bbox* labels, cudamat* indices, cudamat* target); 
int hinge_loss_row_major(cudamat* mat, cudamat* labels, cudamat* target, int quadratic, float margin);
int accumulate_columns(cudamat* mat, cudamat* indices, cudamat* target, float mult, int avg); 
int get_softmax_cross_entropy(cudamat* mat, cudamat* labels, cudamat* target, float tiny); 
int get_softmax_cross_entropy_row_major(cudamat* mat, cudamat* labels, cudamat* target, float tiny); 
int expand(cudamat* source, cudamat* indices, cudamat* target);
int expand_and_add(cudamat* source, cudamat* mat, cudamat* indices, cudamat* target, float mult);
int extract_patches(cudamat* images, cudamat* patches, cudamat* width_offset,
                    cudamat* height_offset, cudamat* flip, int img_width,
                    int img_height, int patch_width, int patch_height);
int rectify_bounding_boxes(cudamat* boxes, cudamat* width_offset,
                           cudamat* height_offset, cudamat* flip,
                           int patch_width, int patch_height);
int adagrad(cudamat* w, cudamat* grad, cudamat* sum_grad_sq, float decay, float epsilon);
int apply_grad_bbox(
    cudamat* mat, cudamat_bbox* bbox, cudamat* indices, cudamat* width_offset,
    cudamat* height_offset, cudamat* target, int width, int height, int depth,
    float scale_width, float scale_height, int loss_function);
int get_softmax_correct_row_major_bbox(
    cudamat* mat, cudamat_bbox* bbox, cudamat* indices, cudamat* width_offset,
    cudamat* height_offset, cudamat* target, int width, int height, int depth,
    float scale_width, float scale_height);
int get_logistic_correct_row_major_bbox(
    cudamat* mat, cudamat_bbox* bbox, cudamat* indices, cudamat* width_offset,
    cudamat* height_offset, cudamat* target, int width, int height, int depth,
    float scale_width, float scale_height, float cutoff);
int get_logistic_correct_normalized(cudamat* mat1, cudamat* mat2, cudamat* out);
#ifdef __cplusplus
}
#endif
#endif
