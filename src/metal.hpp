#pragma once

#include "nn/context.hpp"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace nn {

// Stuffed MTLBuffer. A handle is not a device address, so callers must not add to it.
using Buffer = void*;

struct ArenaBlock {
  void* ptr = nullptr;
  size_t bytes = 0;
  size_t used = 0;
};

struct ContextAccess {
  static Context::Impl& get(Context& ctx) { return *ctx.impl_; }
  static const Context::Impl& get(const Context& ctx) { return *ctx.impl_; }
};

int agx_device_count();
std::string agx_device_name(int device);

Buffer alloc_mem(Context& ctx, size_t bytes);
void free_mem(Buffer mem) noexcept;
void write_mem(Context& ctx, Buffer mem, const void* src, size_t bytes);
void read_mem(Context& ctx, Buffer mem, void* dst, size_t bytes);
void fill_mem_f(Context& ctx, Buffer mem, float value, size_t count);
void fill_mem_i(Context& ctx, Buffer mem, int value, size_t count);
void fill_zero(Context& ctx, Buffer mem, size_t bytes);
void copy_mem(Context& ctx, Buffer dst, Buffer src, size_t bytes);
void wait_quiet(const Context& ctx) noexcept;

inline Buffer mem_of(const float* p) { return const_cast<float*>(p); }
inline Buffer mem_of(const int64_t* p) { return const_cast<int64_t*>(p); }
inline Buffer mem_of(const int* p) { return const_cast<int*>(p); }
inline Buffer mem_of(const uint16_t* p) { return const_cast<uint16_t*>(p); }
inline float* as_float(Buffer mem) { return static_cast<float*>(mem); }
inline int64_t* as_i64(Buffer mem) { return static_cast<int64_t*>(mem); }
inline int* as_i32(Buffer mem) { return static_cast<int*>(mem); }

void launch_scale(Context& ctx, float* x, int64_t n, float factor);
void launch_add_bias(Context& ctx, float* y, const float* bias, int rows, int cols);
void launch_bias_grad(Context& ctx, const float* grad_y, float* grad_bias, int rows, int cols);
void launch_add2(Context& ctx, float* dst, const float* a, const float* b, int64_t n);
void launch_mul(Context& ctx, float* y, const float* a, const float* b, int64_t n);
void launch_gelu(Context& ctx, float* y, const float* x, int64_t n);
void launch_gelu_backward(Context& ctx, float* grad_x, const float* x, const float* grad_y,
                          int64_t n);
void launch_swiglu(Context& ctx, float* y, const float* x, int rows, int64_t inner);
void launch_swiglu_backward(Context& ctx, float* grad_x, const float* x, const float* grad_y,
                            int rows, int64_t inner);
void launch_tanh(Context& ctx, float* y, const float* x, int64_t n);
void launch_tanh_backward(Context& ctx, float* grad_x, const float* y, const float* grad_y,
                          int64_t n);
void launch_relu(Context& ctx, float* y, const float* x, int64_t n);
void launch_relu_backward(Context& ctx, float* grad_x, const float* x, const float* grad_y,
                          int64_t n);
void launch_gather_time(Context& ctx, float* dst, const float* src, int batch, int time,
                        int features, int index);
void launch_scatter_time(Context& ctx, float* dst, const float* src, int batch, int time,
                         int features, int index);
void launch_sgd(Context& ctx, float* param, float* velocity, const float* grad, int64_t n, float lr,
                float momentum);
void launch_adamw(Context& ctx, float* param, float* m, float* v, const float* grad, int64_t n,
                  float lr, float beta1, float beta2, float eps, float weight_decay, float bc1,
                  float bc2);
double launch_sum_squares(Context& ctx, const float* x, int64_t n);
float launch_reduce_sum(Context& ctx, const float* x, int n);

// C[b,i,j] formulas match src/kernels.metal. alpha scales the product. beta == 0 does not read C.
void gemm_nt(Context& ctx, float* C, const float* A, const float* B, int M, int N, int K,
             int batches, int ldc, int lda, int ldb, int64_t stride_c, int64_t stride_a,
             int64_t stride_b, float alpha, float beta);
void gemm_nn(Context& ctx, float* C, const float* A, const float* B, int M, int N, int K,
             int batches, int ldc, int lda, int ldb, int64_t stride_c, int64_t stride_a,
             int64_t stride_b, float alpha, float beta);
void gemm_tn(Context& ctx, float* C, const float* A, const float* B, int M, int N, int K,
             int batches, int ldc, int lda, int ldb, int64_t stride_c, int64_t stride_a,
             int64_t stride_b, float alpha, float beta);

void launch_embedding_gather(Context& ctx, float* out, const float* weight, const int64_t* indices,
                             int64_t rows, int dim);
void launch_embedding_scatter(Context& ctx, float* grad_weight, const float* grad_output,
                              const int64_t* indices, int64_t rows, int dim);

void launch_layer_norm_forward(Context& ctx, float* y, float* mean, float* inv_std, const float* x,
                               const float* gamma, const float* beta, int rows, int cols, float eps);
void launch_layer_norm_backward_input(Context& ctx, float* dx, const float* dy, const float* x,
                                      const float* mean, const float* inv_std, const float* gamma,
                                      int rows, int cols);
void launch_layer_norm_param_grad(Context& ctx, const float* dy, const float* x, const float* mean,
                                  const float* inv_std, float* grad_gamma, float* grad_beta, int rows,
                                  int cols);

void launch_rms_norm_forward(Context& ctx, float* y, float* inv_rms, const float* x,
                             const float* weight, int rows, int cols, float eps);
void launch_rms_norm_backward_input(Context& ctx, float* dx, const float* dy, const float* x,
                                    const float* inv_rms, const float* weight, int rows, int cols);
void launch_rms_norm_param_grad(Context& ctx, const float* dy, const float* x, const float* inv_rms,
                                float* grad_weight, int rows, int cols);

void launch_split_heads(Context& ctx, float* dst, const float* src, int time, int heads, int dh,
                        int64_t n);
void launch_merge_heads(Context& ctx, float* dst, const float* src, int time, int heads, int dh,
                        int64_t n);
void launch_rope(Context& ctx, float* q, float* k, int time, int dh, int pos0, int64_t npairs,
                 int inverse, float base_theta);
void launch_repeat_kv(Context& ctx, float* dst, const float* src, int time, int kv_heads, int group,
                      int dh, int64_t n);
void launch_sum_kv(Context& ctx, float* dst, const float* src, int time, int kv_heads, int group, int dh,
                   int64_t n);
void launch_causal_mask(Context& ctx, float* scores, int time, int64_t n);
void launch_write_kv(Context& ctx, float* dst, const float* src, int t, int t_max, int dh,
                     int64_t n);
void launch_write_kv_u16(Context& ctx, uint16_t* dst, const uint16_t* src, int t, int t_max, int dh,
                         int64_t n);
void launch_softmax_rows(Context& ctx, float* out, const float* in, int rows, int cols);
void launch_softmax_backward(Context& ctx, float* ds, const float* probs, const float* dp, int rows,
                             int cols, int causal);

void launch_softmax_ce(Context& ctx, const float* logits, const int64_t* labels, float* grad,
                       float* per_sample, int n, int classes, int* bad);
void launch_sequence_label_count(Context& ctx, const int64_t* labels, int rows, int classes,
                                 int64_t ignore_index, int* n_valid, int* bad);
void launch_sequence_ce(Context& ctx, const float* logits, const int64_t* labels, float* grad,
                        float* per_sample, int rows, int classes, int64_t ignore_index, int n_valid,
                        int* bad);

void launch_conv2d_forward(Context& ctx, float* y, const float* x, const float* w, const float* bias,
                           int has_bias, int N, int IC, int OC, int H, int W, int K, int OH, int OW,
                           int stride, int pad);
void launch_conv2d_backward_data(Context& ctx, float* dx, const float* dy, const float* w, int N,
                                 int IC, int OC, int H, int W, int K, int OH, int OW, int stride,
                                 int pad);
void launch_conv2d_backward_filter(Context& ctx, float* dw, const float* x, const float* dy, int N,
                                   int IC, int OC, int H, int W, int K, int OH, int OW, int stride,
                                   int pad);
void launch_conv_bias_grad(Context& ctx, const float* dy, float* grad_bias, int N, int OC, int OH,
                           int OW);

void launch_max_pool_forward(Context& ctx, float* y, int* index, const float* x, int N, int C, int H,
                             int W, int K, int OH, int OW, int stride);
void launch_max_pool_backward(Context& ctx, float* dx, const float* dy, const int* index, int N,
                              int C, int H, int W, int OH, int OW);
void launch_global_avg_forward(Context& ctx, float* y, const float* x, int N, int C, int H, int W);
void launch_global_avg_backward(Context& ctx, float* dx, const float* dy, int N, int C, int H, int W);

void launch_batch_norm_train(Context& ctx, float* y, const float* x, const float* gamma,
                             const float* beta, float* running_mean, float* running_var,
                             float* saved_mean, float* saved_inv, int N, int C, int H, int W,
                             float eps, float factor);
void launch_batch_norm_eval(Context& ctx, float* y, const float* x, const float* gamma,
                            const float* beta, const float* running_mean, const float* running_var,
                            int N, int C, int H, int W, float eps);
void launch_batch_norm_backward(Context& ctx, float* dx, const float* dy, const float* x,
                                const float* gamma, const float* saved_mean, const float* saved_inv,
                                float* grad_gamma, float* grad_beta, int N, int C, int H, int W);

}  // namespace nn
