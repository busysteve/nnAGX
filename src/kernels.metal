// Metal Shading Language kernels for nnAGX.
// Translated from the nnCL OpenCL kernels. One in-order command queue.
// Bfloat16 values are bit patterns. Products widen them and accumulate in float32.
// Fast math is off so divf and the bfloat16 rounding stay exact.
#include <metal_stdlib>
using namespace metal;

#pragma clang fp contract(off)

int agx_cmpxchg(device int* addr, int comparand, int desired) {
  device atomic_int* object = (device atomic_int*)addr;
  int expected = comparand;
  while (!atomic_compare_exchange_weak_explicit(object, &expected, desired, memory_order_relaxed,
                                                memory_order_relaxed)) {
    if (expected != comparand) return expected;
  }
  return comparand;
}

int agx_atomic_add(device int* addr, int value) {
  return atomic_fetch_add_explicit((device atomic_int*)addr, value, memory_order_relaxed);
}

float block_sum(float value, threadgroup float* buffer, uint t_tid, uint t_tgs) {
  const int tid = (int)t_tid;
  const int width = (int)t_tgs;
  buffer[tid] = value;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (int offset = width >> 1; offset > 0; offset >>= 1) {
    if (tid < offset) buffer[tid] += buffer[tid + offset];
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  const float total = buffer[0];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  return total;
}

float block_max(float value, threadgroup float* buffer, uint t_tid, uint t_tgs) {
  const int tid = (int)t_tid;
  const int width = (int)t_tgs;
  buffer[tid] = value;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (int offset = width >> 1; offset > 0; offset >>= 1) {
    if (tid < offset) buffer[tid] = fmax(buffer[tid], buffer[tid + offset]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  const float total = buffer[0];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  return total;
}

// libclc's reciprocal is an ulp low here (12/3 -> 3.99999976). Normalization
// multiplies that error by rsqrt(eps), which misses a 1e-5 check. One FMA
// residual step restores the correctly rounded quotient.
float divf(float a, float b) {
  const float q = a / b;
  const float r = fma(-q, b, a);
  return fma(r, 1.f / b, q);
}

void atomic_add_f(device float* addr, float val) {
  device int* iaddr = (device int*)addr;
  int expected = agx_cmpxchg(iaddr, 0, 0);
  while (1) {
    union {
      int i;
      float f;
    } next;
    next.f = as_type<float>(expected) + val;
    const int old = agx_cmpxchg(iaddr, expected, next.i);
    if (old == expected) return;
    expected = old;
  }
}

float gelu_forward(float x) {
  const float k_inv_sqrt2 = 0.7071067811865476f;
  return 0.5f * x * (1.f + erf(x * k_inv_sqrt2));
}

float gelu_grad(float x) {
  const float k_inv_sqrt2 = 0.7071067811865476f;
  const float k_inv_sqrt2pi = 0.3989422804014327f;
  const float cdf = 0.5f * (1.f + erf(x * k_inv_sqrt2));
  const float pdf = k_inv_sqrt2pi * exp(-0.5f * x * x);
  return cdf + x * pdf;
}

float sigmoid_stable(float x) {
  if (x >= 0.f) {
    const float z = exp(-x);
    return 1.f / (1.f + z);
  }
  const float z = exp(x);
  return z / (1.f + z);
}

ushort f32_to_bf16(float value) {
  uint bits = as_type<uint>(value);
  uint exp = bits & 0x7f800000u;
  if (exp == 0x7f800000u) return (ushort)(bits >> 16);
  uint lsb = (bits >> 16) & 1u;
  bits += 0x7fffu + lsb;
  return (ushort)(bits >> 16);
}

float bf16_to_f32(ushort bits) { return as_type<float>(((uint)bits) << 16); }


kernel void scale_f(
    device float* x [[buffer(0)]],
    constant long& n [[buffer(1)]],
    constant float& factor [[buffer(2)]],
    uint gid [[thread_position_in_grid]]) {
  const long i = (long)gid;
  if (i < n) x[i] *= factor;
}


kernel void add_bias(
    device float* y [[buffer(0)]],
    const device float* bias [[buffer(1)]],
    constant int& rows [[buffer(2)]],
    constant int& cols [[buffer(3)]],
    uint gid [[thread_position_in_grid]]) {
  const long n = (long)rows * cols;
  const long i = (long)gid;
  if (i < n) y[i] += bias[i % cols];
}


kernel void bias_grad(
    const device float* grad_y [[buffer(0)]],
    device float* grad_bias [[buffer(1)]],
    constant int& rows [[buffer(2)]],
    constant int& cols [[buffer(3)]],
    threadgroup float* buffer [[threadgroup(0)]],
    uint t_tid [[thread_position_in_threadgroup]],
    uint t_tgs [[threads_per_threadgroup]],
    uint t_grp [[threadgroup_position_in_grid]]) {
  const int col = (int)t_grp;
  if (col >= cols) return;
  float sum = 0.f;
  for (int row = (int)t_tid; row < rows; row += (int)t_tgs) {
    sum += grad_y[(long)row * cols + col];
  }
  sum = block_sum(sum, buffer, t_tid, t_tgs);
  if (t_tid == 0) grad_bias[col] += sum;
}


kernel void add2(
    device float* dst [[buffer(0)]],
    const device float* a [[buffer(1)]],
    const device float* b [[buffer(2)]],
    constant long& n [[buffer(3)]],
    uint gid [[thread_position_in_grid]]) {
  const long i = (long)gid;
  if (i < n) dst[i] = a[i] + b[i];
}


kernel void mul2(
    device float* y [[buffer(0)]],
    const device float* a [[buffer(1)]],
    const device float* b [[buffer(2)]],
    constant long& n [[buffer(3)]],
    uint gid [[thread_position_in_grid]]) {
  const long i = (long)gid;
  if (i < n) y[i] = a[i] * b[i];
}


kernel void gelu_k(
    device float* y [[buffer(0)]],
    const device float* x [[buffer(1)]],
    constant long& n [[buffer(2)]],
    uint gid [[thread_position_in_grid]]) {
  const long i = (long)gid;
  if (i < n) y[i] = gelu_forward(x[i]);
}


kernel void gelu_backward_k(
    device float* grad_x [[buffer(0)]],
    const device float* x [[buffer(1)]],
    const device float* grad_y [[buffer(2)]],
    constant long& n [[buffer(3)]],
    uint gid [[thread_position_in_grid]]) {
  const long i = (long)gid;
  if (i < n) grad_x[i] = grad_y[i] * gelu_grad(x[i]);
}


kernel void swiglu_k(
    device float* y [[buffer(0)]],
    const device float* x [[buffer(1)]],
    constant int& rows [[buffer(2)]],
    constant long& inner [[buffer(3)]],
    uint gid [[thread_position_in_grid]]) {
  const long n = (long)rows * inner;
  const long i = (long)gid;
  if (i >= n) return;
  const long row = i / inner;
  const long col = i - row * inner;
  const long base = row * (inner * 2) + col;
  y[i] = silu(x[base]) * x[base + inner];
}


kernel void swiglu_backward_k(
    device float* grad_x [[buffer(0)]],
    const device float* x [[buffer(1)]],
    const device float* grad_y [[buffer(2)]],
    constant int& rows [[buffer(3)]],
    constant long& inner [[buffer(4)]],
    uint gid [[thread_position_in_grid]]) {
  const long n = (long)rows * inner;
  const long i = (long)gid;
  if (i >= n) return;
  const long row = i / inner;
  const long col = i - row * inner;
  const long base = row * (inner * 2) + col;
  const float gate = x[base];
  const float value = x[base + inner];
  const float dy = grad_y[i];
  grad_x[base] = dy * value * silu_grad(gate);
  grad_x[base + inner] = dy * silu(gate);
}


kernel void tanh_k(
    device float* y [[buffer(0)]],
    const device float* x [[buffer(1)]],
    constant long& n [[buffer(2)]],
    uint gid [[thread_position_in_grid]]) {
  const long i = (long)gid;
  if (i < n) y[i] = tanh(x[i]);
}


kernel void tanh_backward_k(
    device float* grad_x [[buffer(0)]],
    const device float* y [[buffer(1)]],
    const device float* grad_y [[buffer(2)]],
    constant long& n [[buffer(3)]],
    uint gid [[thread_position_in_grid]]) {
  const long i = (long)gid;
  if (i >= n) return;
  const float h = y[i];
  grad_x[i] = grad_y[i] * (1.f - h * h);
}


kernel void relu_k(
    device float* y [[buffer(0)]],
    const device float* x [[buffer(1)]],
    constant long& n [[buffer(2)]],
    uint gid [[thread_position_in_grid]]) {
  const long i = (long)gid;
  if (i < n) y[i] = x[i] > 0.f ? x[i] : 0.f;
}


kernel void relu_backward_k(
    device float* grad_x [[buffer(0)]],
    const device float* x [[buffer(1)]],
    const device float* grad_y [[buffer(2)]],
    constant long& n [[buffer(3)]],
    uint gid [[thread_position_in_grid]]) {
  const long i = (long)gid;
  if (i < n) grad_x[i] = x[i] > 0.f ? grad_y[i] : 0.f;
}


kernel void gather_time(
    device float* dst [[buffer(0)]],
    const device float* src [[buffer(1)]],
    constant int& batch [[buffer(2)]],
    constant int& time [[buffer(3)]],
    constant int& features [[buffer(4)]],
    constant int& index [[buffer(5)]],
    uint gid [[thread_position_in_grid]]) {
  const long n = (long)batch * features;
  const long i = (long)gid;
  if (i >= n) return;
  const long row = i / features;
  const long col = i - row * features;
  dst[i] = src[(row * time + index) * features + col];
}


kernel void scatter_time(
    device float* dst [[buffer(0)]],
    const device float* src [[buffer(1)]],
    constant int& batch [[buffer(2)]],
    constant int& time [[buffer(3)]],
    constant int& features [[buffer(4)]],
    constant int& index [[buffer(5)]],
    uint gid [[thread_position_in_grid]]) {
  const long n = (long)batch * features;
  const long i = (long)gid;
  if (i >= n) return;
  const long row = i / features;
  const long col = i - row * features;
  dst[(row * time + index) * features + col] = src[i];
}


kernel void sgd_k(
    device float* param [[buffer(0)]],
    device float* velocity [[buffer(1)]],
    const device float* grad [[buffer(2)]],
    constant long& n [[buffer(3)]],
    constant float& lr [[buffer(4)]],
    constant float& momentum [[buffer(5)]],
    uint gid [[thread_position_in_grid]]) {
  const long i = (long)gid;
  if (i >= n) return;
  const float updated = momentum * velocity[i] + grad[i];
  velocity[i] = updated;
  param[i] = param[i] - lr * updated;
}


kernel void adamw_k(
    device float* param [[buffer(0)]],
    device float* m [[buffer(1)]],
    device float* v [[buffer(2)]],
    const device float* grad [[buffer(3)]],
    constant long& n [[buffer(4)]],
    constant float& lr [[buffer(5)]],
    constant float& beta1 [[buffer(6)]],
    constant float& beta2 [[buffer(7)]],
    constant float& eps [[buffer(8)]],
    constant float& weight_decay [[buffer(9)]],
    constant float& bc1 [[buffer(10)]],
    constant float& bc2 [[buffer(11)]],
    uint gid [[thread_position_in_grid]]) {
  const long i = (long)gid;
  if (i >= n) return;
  const float g = grad[i];
  const float m_new = beta1 * m[i] + (1.f - beta1) * g;
  const float v_new = beta2 * v[i] + (1.f - beta2) * g * g;
  m[i] = m_new;
  v[i] = v_new;
  const float m_hat = m_new / bc1;
  const float v_hat = v_new / bc2;
  const float update = m_hat / (sqrt(v_hat) + eps) + weight_decay * param[i];
  param[i] = param[i] - lr * update;
}


kernel void sum_squares_f(
    const device float* x [[buffer(0)]],
    constant long& n [[buffer(1)]],
    device float* partials [[buffer(2)]],
    threadgroup float* buffer [[threadgroup(0)]],
    uint gid [[thread_position_in_grid]],
    uint t_tid [[thread_position_in_threadgroup]],
    uint t_tgs [[threads_per_threadgroup]],
    uint t_grp [[threadgroup_position_in_grid]],
    uint t_ngrp [[threadgroups_per_grid]]) {
  float sum = 0.f;
  const long stride = (long)t_tgs * t_ngrp;
  for (long i = (long)gid; i < n; i += stride) {
    const float value = x[i];
    sum += value * value;
  }
  sum = block_sum(sum, buffer, t_tid, t_tgs);
  if (t_tid == 0) partials[t_grp] = sum;
}


kernel void reduce_sum_f(
    const device float* x [[buffer(0)]],
    constant int& n [[buffer(1)]],
    device float* out [[buffer(2)]],
    threadgroup float* buffer [[threadgroup(0)]],
    uint t_tid [[thread_position_in_threadgroup]],
    uint t_tgs [[threads_per_threadgroup]]) {
  float sum = 0.f;
  for (int i = (int)t_tid; i < n; i += (int)t_tgs) sum += x[i];
  sum = block_sum(sum, buffer, t_tid, t_tgs);
  if (t_tid == 0) *out = sum;
}


kernel void gemm_nt(
    device float* C [[buffer(0)]],
    const device float* A [[buffer(1)]],
    const device float* B [[buffer(2)]],
    constant int& M [[buffer(3)]],
    constant int& N [[buffer(4)]],
    constant int& K [[buffer(5)]],
    constant int& batches [[buffer(6)]],
    constant int& ldc [[buffer(7)]],
    constant int& lda [[buffer(8)]],
    constant int& ldb [[buffer(9)]],
    constant long& stride_c [[buffer(10)]],
    constant long& stride_a [[buffer(11)]],
    constant long& stride_b [[buffer(12)]],
    constant float& alpha [[buffer(13)]],
    constant float& beta [[buffer(14)]],
    uint3 gid [[thread_position_in_grid]]) {
  const int j = (int)gid.x;
  const int i = (int)gid.y;
  const int b = (int)gid.z;
  if (i >= M || j >= N || b >= batches) return;
  const device float* a = A + b * stride_a + (long)i * lda;
  const device float* brow = B + b * stride_b + (long)j * ldb;
  float sum = 0.f;
  for (int p = 0; p < K; ++p) sum += a[p] * brow[p];
  device float* c = C + b * stride_c + (long)i * ldc + j;
  *c = (beta == 0.f) ? alpha * sum : alpha * sum + beta * (*c);
}


kernel void gemm_nn(
    device float* C [[buffer(0)]],
    const device float* A [[buffer(1)]],
    const device float* B [[buffer(2)]],
    constant int& M [[buffer(3)]],
    constant int& N [[buffer(4)]],
    constant int& K [[buffer(5)]],
    constant int& batches [[buffer(6)]],
    constant int& ldc [[buffer(7)]],
    constant int& lda [[buffer(8)]],
    constant int& ldb [[buffer(9)]],
    constant long& stride_c [[buffer(10)]],
    constant long& stride_a [[buffer(11)]],
    constant long& stride_b [[buffer(12)]],
    constant float& alpha [[buffer(13)]],
    constant float& beta [[buffer(14)]],
    uint3 gid [[thread_position_in_grid]]) {
  const int j = (int)gid.x;
  const int i = (int)gid.y;
  const int b = (int)gid.z;
  if (i >= M || j >= N || b >= batches) return;
  const device float* a = A + b * stride_a + (long)i * lda;
  const device float* bbase = B + b * stride_b;
  float sum = 0.f;
  for (int p = 0; p < K; ++p) sum += a[p] * bbase[(long)p * ldb + j];
  device float* c = C + b * stride_c + (long)i * ldc + j;
  *c = (beta == 0.f) ? alpha * sum : alpha * sum + beta * (*c);
}


kernel void gemm_tn(
    device float* C [[buffer(0)]],
    const device float* A [[buffer(1)]],
    const device float* B [[buffer(2)]],
    constant int& M [[buffer(3)]],
    constant int& N [[buffer(4)]],
    constant int& K [[buffer(5)]],
    constant int& batches [[buffer(6)]],
    constant int& ldc [[buffer(7)]],
    constant int& lda [[buffer(8)]],
    constant int& ldb [[buffer(9)]],
    constant long& stride_c [[buffer(10)]],
    constant long& stride_a [[buffer(11)]],
    constant long& stride_b [[buffer(12)]],
    constant float& alpha [[buffer(13)]],
    constant float& beta [[buffer(14)]],
    uint3 gid [[thread_position_in_grid]]) {
  const int j = (int)gid.x;
  const int i = (int)gid.y;
  const int b = (int)gid.z;
  if (i >= M || j >= N || b >= batches) return;
  const device float* abase = A + b * stride_a;
  const device float* bbase = B + b * stride_b;
  float sum = 0.f;
  for (int p = 0; p < K; ++p) sum += abase[(long)p * lda + i] * bbase[(long)p * ldb + j];
  device float* c = C + b * stride_c + (long)i * ldc + j;
  *c = (beta == 0.f) ? alpha * sum : alpha * sum + beta * (*c);
}


kernel void embedding_gather(
    device float* out [[buffer(0)]],
    const device float* weight [[buffer(1)]],
    const device long* indices [[buffer(2)]],
    constant long& rows [[buffer(3)]],
    constant int& dim [[buffer(4)]],
    uint gid [[thread_position_in_grid]]) {
  const long n = rows * (long)dim;
  const long i = (long)gid;
  if (i >= n) return;
  const long row = i / dim;
  const int col = (int)(i - row * dim);
  const long id = indices[row];
  out[i] = weight[id * (long)dim + col];
}


kernel void embedding_scatter(
    device float* grad_weight [[buffer(0)]],
    const device float* grad_output [[buffer(1)]],
    const device long* indices [[buffer(2)]],
    constant long& rows [[buffer(3)]],
    constant int& dim [[buffer(4)]],
    uint gid [[thread_position_in_grid]]) {
  const long n = rows * (long)dim;
  const long i = (long)gid;
  if (i >= n) return;
  const long row = i / dim;
  const int col = (int)(i - row * dim);
  const long id = indices[row];
  atomic_add_f(grad_weight + id * (long)dim + col, grad_output[i]);
}


kernel void layer_norm_forward(
    device float* y [[buffer(0)]],
    device float* mean_out [[buffer(1)]],
    device float* inv_std_out [[buffer(2)]],
    const device float* x [[buffer(3)]],
    const device float* gamma [[buffer(4)]],
    const device float* beta [[buffer(5)]],
    constant int& cols [[buffer(6)]],
    constant float& eps [[buffer(7)]],
    threadgroup float* buffer [[threadgroup(0)]],
    uint t_tid [[thread_position_in_threadgroup]],
    uint t_tgs [[threads_per_threadgroup]],
    uint t_grp [[threadgroup_position_in_grid]]) {
  const int row = (int)t_grp;
  const long base = (long)row * cols;
  const device float* row_x = x + base;
  device float* row_y = y + base;
  float sum = 0.f;
  for (int col = (int)t_tid; col < cols; col += (int)t_tgs) sum += row_x[col];
  const float mean = divf(block_sum(sum, buffer, t_tid, t_tgs), (float)cols);
  float var_sum = 0.f;
  for (int col = (int)t_tid; col < cols; col += (int)t_tgs) {
    const float diff = row_x[col] - mean;
    var_sum += diff * diff;
  }
  var_sum = block_sum(var_sum, buffer, t_tid, t_tgs);
  const float inv_std = rsqrt(divf(var_sum, (float)cols) + eps);
  if (t_tid == 0) {
    mean_out[row] = mean;
    inv_std_out[row] = inv_std;
  }
  for (int col = (int)t_tid; col < cols; col += (int)t_tgs) {
    const float xhat = (row_x[col] - mean) * inv_std;
    row_y[col] = xhat * gamma[col] + beta[col];
  }
}


kernel void layer_norm_backward_input(
    device float* dx [[buffer(0)]],
    const device float* dy [[buffer(1)]],
    const device float* x [[buffer(2)]],
    const device float* mean [[buffer(3)]],
    const device float* inv_std [[buffer(4)]],
    const device float* gamma [[buffer(5)]],
    constant int& cols [[buffer(6)]],
    threadgroup float* buffer [[threadgroup(0)]],
    uint t_tid [[thread_position_in_threadgroup]],
    uint t_tgs [[threads_per_threadgroup]],
    uint t_grp [[threadgroup_position_in_grid]]) {
  const int row = (int)t_grp;
  const long base = (long)row * cols;
  const float mean_v = mean[row];
  const float inv = inv_std[row];
  float sum_dhat = 0.f;
  float sum_dhat_xhat = 0.f;
  for (int col = (int)t_tid; col < cols; col += (int)t_tgs) {
    const float xhat = (x[base + col] - mean_v) * inv;
    const float dhat = dy[base + col] * gamma[col];
    sum_dhat += dhat;
    sum_dhat_xhat += dhat * xhat;
  }
  sum_dhat = block_sum(sum_dhat, buffer, t_tid, t_tgs);
  sum_dhat_xhat = block_sum(sum_dhat_xhat, buffer, t_tid, t_tgs);
  const float mean_dhat = divf(sum_dhat, (float)cols);
  const float mean_dhat_xhat = divf(sum_dhat_xhat, (float)cols);
  for (int col = (int)t_tid; col < cols; col += (int)t_tgs) {
    const float xhat = (x[base + col] - mean_v) * inv;
    const float dhat = dy[base + col] * gamma[col];
    dx[base + col] = inv * (dhat - mean_dhat - xhat * mean_dhat_xhat);
  }
}


kernel void layer_norm_param_grad(
    const device float* dy [[buffer(0)]],
    const device float* x [[buffer(1)]],
    const device float* mean [[buffer(2)]],
    const device float* inv_std [[buffer(3)]],
    device float* grad_gamma [[buffer(4)]],
    device float* grad_beta [[buffer(5)]],
    constant int& rows [[buffer(6)]],
    constant int& cols [[buffer(7)]],
    threadgroup float* buffer [[threadgroup(0)]],
    uint t_tid [[thread_position_in_threadgroup]],
    uint t_tgs [[threads_per_threadgroup]],
    uint t_grp [[threadgroup_position_in_grid]]) {
  const int col = (int)t_grp;
  float sum_gamma = 0.f;
  float sum_beta = 0.f;
  for (int row = (int)t_tid; row < rows; row += (int)t_tgs) {
    const long index = (long)row * cols + col;
    const float xhat = (x[index] - mean[row]) * inv_std[row];
    const float dy_v = dy[index];
    sum_gamma += dy_v * xhat;
    sum_beta += dy_v;
  }
  sum_gamma = block_sum(sum_gamma, buffer, t_tid, t_tgs);
  sum_beta = block_sum(sum_beta, buffer, t_tid, t_tgs);
  if (t_tid == 0) {
    grad_gamma[col] += sum_gamma;
    grad_beta[col] += sum_beta;
  }
}


kernel void rms_norm_forward(
    device float* y [[buffer(0)]],
    device float* inv_rms_out [[buffer(1)]],
    const device float* x [[buffer(2)]],
    const device float* weight [[buffer(3)]],
    constant int& cols [[buffer(4)]],
    constant float& eps [[buffer(5)]],
    threadgroup float* buffer [[threadgroup(0)]],
    uint t_tid [[thread_position_in_threadgroup]],
    uint t_tgs [[threads_per_threadgroup]],
    uint t_grp [[threadgroup_position_in_grid]]) {
  const int row = (int)t_grp;
  const long base = (long)row * cols;
  const device float* row_x = x + base;
  device float* row_y = y + base;
  float sq = 0.f;
  for (int col = (int)t_tid; col < cols; col += (int)t_tgs) {
    const float value = row_x[col];
    sq += value * value;
  }
  sq = block_sum(sq, buffer, t_tid, t_tgs);
  const float inv_rms = rsqrt(divf(sq, (float)cols) + eps);
  if (t_tid == 0) inv_rms_out[row] = inv_rms;
  for (int col = (int)t_tid; col < cols; col += (int)t_tgs) {
    row_y[col] = row_x[col] * inv_rms * weight[col];
  }
}


kernel void rms_norm_backward_input(
    device float* dx [[buffer(0)]],
    const device float* dy [[buffer(1)]],
    const device float* x [[buffer(2)]],
    const device float* inv_rms [[buffer(3)]],
    const device float* weight [[buffer(4)]],
    constant int& cols [[buffer(5)]],
    threadgroup float* buffer [[threadgroup(0)]],
    uint t_tid [[thread_position_in_threadgroup]],
    uint t_tgs [[threads_per_threadgroup]],
    uint t_grp [[threadgroup_position_in_grid]]) {
  const int row = (int)t_grp;
  const long base = (long)row * cols;
  const float inv = inv_rms[row];
  float sum_dhat_xhat = 0.f;
  for (int col = (int)t_tid; col < cols; col += (int)t_tgs) {
    const float xhat = x[base + col] * inv;
    const float dhat = dy[base + col] * weight[col];
    sum_dhat_xhat += dhat * xhat;
  }
  const float mean_dhat_xhat = divf(block_sum(sum_dhat_xhat, buffer, t_tid, t_tgs), (float)cols);
  for (int col = (int)t_tid; col < cols; col += (int)t_tgs) {
    const float xhat = x[base + col] * inv;
    const float dhat = dy[base + col] * weight[col];
    dx[base + col] = inv * (dhat - xhat * mean_dhat_xhat);
  }
}


kernel void rms_norm_param_grad(
    const device float* dy [[buffer(0)]],
    const device float* x [[buffer(1)]],
    const device float* inv_rms [[buffer(2)]],
    device float* grad_weight [[buffer(3)]],
    constant int& rows [[buffer(4)]],
    constant int& cols [[buffer(5)]],
    threadgroup float* buffer [[threadgroup(0)]],
    uint t_tid [[thread_position_in_threadgroup]],
    uint t_tgs [[threads_per_threadgroup]],
    uint t_grp [[threadgroup_position_in_grid]]) {
  const int col = (int)t_grp;
  float sum = 0.f;
  for (int row = (int)t_tid; row < rows; row += (int)t_tgs) {
    const long index = (long)row * cols + col;
    const float xhat = x[index] * inv_rms[row];
    sum += dy[index] * xhat;
  }
  sum = block_sum(sum, buffer, t_tid, t_tgs);
  if (t_tid == 0) grad_weight[col] += sum;
}


kernel void split_heads(
    device float* dst [[buffer(0)]],
    const device float* src [[buffer(1)]],
    constant int& time [[buffer(2)]],
    constant int& heads [[buffer(3)]],
    constant int& dh [[buffer(4)]],
    constant long& n [[buffer(5)]],
    uint gid [[thread_position_in_grid]]) {
  const long i = (long)gid;
  if (i >= n) return;
  const long d = i % dh;
  const long h = (i / dh) % heads;
  const long t = (i / ((long)dh * heads)) % time;
  const long batch = i / ((long)dh * heads * time);
  const long o = ((batch * heads + h) * time + t) * dh + d;
  dst[o] = src[i];
}


kernel void repeat_kv(
    device float* dst [[buffer(0)]],
    const device float* src [[buffer(1)]],
    constant int& time [[buffer(2)]],
    constant int& kv_heads [[buffer(3)]],
    constant int& group [[buffer(4)]],
    constant int& dh [[buffer(5)]],
    constant long& n [[buffer(6)]],
    uint gid [[thread_position_in_grid]]) {
  const long i = (long)gid;
  if (i >= n) return;
  // dst and src are (N, heads, T, Dh). The head axis is outside time.
  const int heads = kv_heads * group;
  const long d = i % dh;
  const long t = (i / dh) % time;
  const long h = (i / ((long)dh * time)) % heads;
  const long batch = i / ((long)dh * time * heads);
  const long hkv = h / group;
  const long s = ((batch * kv_heads + hkv) * time + t) * dh + d;
  dst[i] = src[s];
}


kernel void sum_kv(
    device float* dst [[buffer(0)]],
    const device float* src [[buffer(1)]],
    constant int& time [[buffer(2)]],
    constant int& kv_heads [[buffer(3)]],
    constant int& group [[buffer(4)]],
    constant int& dh [[buffer(5)]],
    constant long& n [[buffer(6)]],
    uint gid [[thread_position_in_grid]]) {
  const long i = (long)gid;
  if (i >= n) return;
  const int heads = kv_heads * group;
  const long d = i % dh;
  const long t = (i / dh) % time;
  const long hkv = (i / ((long)dh * time)) % kv_heads;
  const long batch = i / ((long)dh * time * kv_heads);
  float sum = 0.f;
  for (int g = 0; g < group; ++g) {
    const long h = (long)hkv * group + g;
    const long s = ((batch * heads + h) * time + t) * dh + d;
    sum += src[s];
  }
  dst[i] = sum;
}


kernel void merge_heads(
    device float* dst [[buffer(0)]],
    const device float* src [[buffer(1)]],
    constant int& time [[buffer(2)]],
    constant int& heads [[buffer(3)]],
    constant int& dh [[buffer(4)]],
    constant long& n [[buffer(5)]],
    uint gid [[thread_position_in_grid]]) {
  const long i = (long)gid;
  if (i >= n) return;
  const long d = i % dh;
  const long h = (i / dh) % heads;
  const long t = (i / ((long)dh * heads)) % time;
  const long batch = i / ((long)dh * heads * time);
  const long s = ((batch * heads + h) * time + t) * dh + d;
  dst[i] = src[s];
}


kernel void rope_qk(
    device float* q [[buffer(0)]],
    device float* k [[buffer(1)]],
    constant int& time [[buffer(2)]],
    constant int& dh [[buffer(3)]],
    constant int& pos0 [[buffer(4)]],
    constant long& npairs [[buffer(5)]],
    constant int& inverse [[buffer(6)]],
    constant float& base_theta [[buffer(7)]],
    uint gid [[thread_position_in_grid]]) {
  const long pair_index = (long)gid;
  if (pair_index >= npairs) return;
  const int pairs = dh / 2;
  const int pair = (int)(pair_index % pairs);
  const long vec = pair_index / pairs;
  const int t = (int)(vec % time);
  const long base = vec * dh + (long)pair * 2;
  const float theta = (float)(pos0 + t) *
                      exp(-log(base_theta) * (2.f * divf((float)pair, (float)dh)));
  const float c = cos(theta);
  const float s = sin(theta);
  const float q0 = q[base];
  const float q1 = q[base + 1];
  const float k0 = k[base];
  const float k1 = k[base + 1];
  if (inverse) {
    q[base] = q0 * c + q1 * s;
    q[base + 1] = -q0 * s + q1 * c;
    k[base] = k0 * c + k1 * s;
    k[base + 1] = -k0 * s + k1 * c;
  } else {
    q[base] = q0 * c - q1 * s;
    q[base + 1] = q0 * s + q1 * c;
    k[base] = k0 * c - k1 * s;
    k[base + 1] = k0 * s + k1 * c;
  }
}


kernel void causal_mask(
    device float* scores [[buffer(0)]],
    constant int& time [[buffer(1)]],
    constant long& n [[buffer(2)]],
    uint gid [[thread_position_in_grid]]) {
  const long i = (long)gid;
  if (i >= n) return;
  const int col = (int)(i % time);
  const int row = (int)((i / time) % time);
  if (col > row) scores[i] = -1.0e9f;
}


kernel void write_kv(
    device float* dst [[buffer(0)]],
    const device float* src [[buffer(1)]],
    constant int& t [[buffer(2)]],
    constant int& t_max [[buffer(3)]],
    constant int& dh [[buffer(4)]],
    constant long& n [[buffer(5)]],
    uint gid [[thread_position_in_grid]]) {
  const long i = (long)gid;
  if (i >= n) return;
  const int d = (int)(i % dh);
  const long head = i / dh;
  dst[(head * (long)t_max + t) * dh + d] = src[i];
}


kernel void write_kv_u16(
    device ushort* dst [[buffer(0)]],
    const device ushort* src [[buffer(1)]],
    constant int& t [[buffer(2)]],
    constant int& t_max [[buffer(3)]],
    constant int& dh [[buffer(4)]],
    constant long& n [[buffer(5)]],
    uint gid [[thread_position_in_grid]]) {
  const long i = (long)gid;
  if (i >= n) return;
  const int d = (int)(i % dh);
  const long head = i / dh;
  dst[(head * (long)t_max + t) * dh + d] = src[i];
}


kernel void softmax_rows(
    device float* out [[buffer(0)]],
    const device float* in [[buffer(1)]],
    constant int& cols [[buffer(2)]],
    threadgroup float* buffer [[threadgroup(0)]],
    uint t_tid [[thread_position_in_threadgroup]],
    uint t_tgs [[threads_per_threadgroup]],
    uint t_grp [[threadgroup_position_in_grid]]) {
  const int row = (int)t_grp;
  const long base = (long)row * cols;
  const device float* row_in = in + base;
  device float* row_out = out + base;
  float max_v = -INFINITY;
  for (int col = (int)t_tid; col < cols; col += (int)t_tgs) {
    max_v = fmax(max_v, row_in[col]);
  }
  max_v = block_max(max_v, buffer, t_tid, t_tgs);
  float sum = 0.f;
  for (int col = (int)t_tid; col < cols; col += (int)t_tgs) {
    sum += exp(row_in[col] - max_v);
  }
  sum = block_sum(sum, buffer, t_tid, t_tgs);
  const float inv = divf(1.f, sum);
  for (int col = (int)t_tid; col < cols; col += (int)t_tgs) {
    row_out[col] = exp(row_in[col] - max_v) * inv;
  }
}


kernel void softmax_backward(
    device float* ds [[buffer(0)]],
    const device float* probs [[buffer(1)]],
    const device float* dp [[buffer(2)]],
    constant int& cols [[buffer(3)]],
    constant int& causal [[buffer(4)]],
    threadgroup float* buffer [[threadgroup(0)]],
    uint t_tid [[thread_position_in_threadgroup]],
    uint t_tgs [[threads_per_threadgroup]],
    uint t_grp [[threadgroup_position_in_grid]]) {
  const int row = (int)t_grp;
  const int query = row % cols;
  const long base = (long)row * cols;
  float dot = 0.f;
  for (int col = (int)t_tid; col < cols; col += (int)t_tgs) {
    dot += dp[base + col] * probs[base + col];
  }
  dot = block_sum(dot, buffer, t_tid, t_tgs);
  for (int col = (int)t_tid; col < cols; col += (int)t_tgs) {
    float grad = probs[base + col] * (dp[base + col] - dot);
    if (causal && col > query) grad = 0.f;
    ds[base + col] = grad;
  }
}


kernel void softmax_ce(
    const device float* logits [[buffer(0)]],
    const device long* labels [[buffer(1)]],
    device float* grad [[buffer(2)]],
    device float* per_sample [[buffer(3)]],
    constant int& n [[buffer(4)]],
    constant int& classes [[buffer(5)]],
    device int* bad [[buffer(6)]],
    threadgroup float* scratch [[threadgroup(0)]],
    uint t_tid [[thread_position_in_threadgroup]],
    uint t_tgs [[threads_per_threadgroup]],
    uint t_grp [[threadgroup_position_in_grid]]) {
  const int row = (int)t_grp;
  if (row >= n) return;
  const device float* row_logits = logits + (long)row * classes;
  device float* row_grad = grad + (long)row * classes;
  float thread_max = -1e30f;
  for (int c = (int)t_tid; c < classes; c += (int)t_tgs) {
    thread_max = fmax(thread_max, row_logits[c]);
  }
  const float max_logit = block_max(thread_max, scratch, t_tid, t_tgs);
  float thread_sum = 0.f;
  for (int c = (int)t_tid; c < classes; c += (int)t_tgs) {
    thread_sum += exp(row_logits[c] - max_logit);
  }
  const float sum = block_sum(thread_sum, scratch, t_tid, t_tgs);
  const long label = labels[row];
  if (label < 0 || label >= classes) {
    if (t_tid == 0) agx_cmpxchg(bad, 0, 1);
    return;
  }
  const float inv_sum = divf(1.f, sum);
  const float inv_n = divf(1.f, (float)n);
  for (int c = (int)t_tid; c < classes; c += (int)t_tgs) {
    const float softmax = exp(row_logits[c] - max_logit) * inv_sum;
    const float indicator = ((long)c == label) ? 1.f : 0.f;
    row_grad[c] = (softmax - indicator) * inv_n;
  }
  if (t_tid == 0) per_sample[row] = -(row_logits[label] - (max_logit + log(sum)));
}


kernel void sequence_label_count(
    const device long* labels [[buffer(0)]],
    constant int& rows [[buffer(1)]],
    constant int& classes [[buffer(2)]],
    constant long& ignore_index [[buffer(3)]],
    device int* n_valid [[buffer(4)]],
    device int* bad [[buffer(5)]],
    uint gid [[thread_position_in_grid]]) {
  const long row = (long)gid;
  if (row >= rows) return;
  const long label = labels[row];
  if (label == ignore_index) return;
  if (label < 0 || label >= (long)classes) {
    agx_cmpxchg(bad, 0, 1);
    return;
  }
  agx_atomic_add(n_valid, 1);
}


kernel void sequence_ce(
    const device float* logits [[buffer(0)]],
    const device long* labels [[buffer(1)]],
    device float* grad [[buffer(2)]],
    device float* per_sample [[buffer(3)]],
    constant int& rows [[buffer(4)]],
    constant int& classes [[buffer(5)]],
    constant long& ignore_index [[buffer(6)]],
    constant int& n_valid [[buffer(7)]],
    device int* bad [[buffer(8)]],
    threadgroup float* scratch [[threadgroup(0)]],
    uint t_tid [[thread_position_in_threadgroup]],
    uint t_tgs [[threads_per_threadgroup]],
    uint t_grp [[threadgroup_position_in_grid]]) {
  const int row = (int)t_grp;
  if (row >= rows) return;
  const device float* row_logits = logits + (long)row * classes;
  device float* row_grad = grad + (long)row * classes;
  const long label = labels[row];
  if (label == ignore_index) {
    for (int c = (int)t_tid; c < classes; c += (int)t_tgs) row_grad[c] = 0.f;
    if (t_tid == 0) per_sample[row] = 0.f;
    return;
  }
  if (label < 0 || label >= (long)classes) {
    for (int c = (int)t_tid; c < classes; c += (int)t_tgs) row_grad[c] = 0.f;
    if (t_tid == 0) {
      agx_cmpxchg(bad, 0, 1);
      per_sample[row] = 0.f;
    }
    return;
  }
  float thread_max = -1e30f;
  for (int c = (int)t_tid; c < classes; c += (int)t_tgs) {
    thread_max = fmax(thread_max, row_logits[c]);
  }
  const float max_logit = block_max(thread_max, scratch, t_tid, t_tgs);
  float thread_sum = 0.f;
  for (int c = (int)t_tid; c < classes; c += (int)t_tgs) {
    thread_sum += exp(row_logits[c] - max_logit);
  }
  const float sum = block_sum(thread_sum, scratch, t_tid, t_tgs);
  const float inv_sum = divf(1.f, sum);
  const float inv_n = divf(1.f, (float)n_valid);
  for (int c = (int)t_tid; c < classes; c += (int)t_tgs) {
    const float softmax = exp(row_logits[c] - max_logit) * inv_sum;
    const float indicator = ((long)c == label) ? 1.f : 0.f;
    row_grad[c] = (softmax - indicator) * inv_n;
  }
  if (t_tid == 0) per_sample[row] = -(row_logits[label] - (max_logit + log(sum)));
}


kernel void conv2d_forward(
    device float* y [[buffer(0)]],
    const device float* x [[buffer(1)]],
    const device float* w [[buffer(2)]],
    const device float* bias [[buffer(3)]],
    constant int& has_bias [[buffer(4)]],
    constant int& N [[buffer(5)]],
    constant int& IC [[buffer(6)]],
    constant int& OC [[buffer(7)]],
    constant int& H [[buffer(8)]],
    constant int& W [[buffer(9)]],
    constant int& K [[buffer(10)]],
    constant int& OH [[buffer(11)]],
    constant int& OW [[buffer(12)]],
    constant int& stride [[buffer(13)]],
    constant int& pad [[buffer(14)]],
    uint gid [[thread_position_in_grid]]) {
  const long total = (long)N * OC * OH * OW;
  const long idx = (long)gid;
  if (idx >= total) return;
  const int ow = (int)(idx % OW);
  const int oh = (int)((idx / OW) % OH);
  const int oc = (int)((idx / ((long)OW * OH)) % OC);
  const int n = (int)(idx / ((long)OW * OH * OC));
  float sum = has_bias ? bias[oc] : 0.f;
  const int h0 = oh * stride - pad;
  const int w0 = ow * stride - pad;
  for (int ic = 0; ic < IC; ++ic) {
    for (int kh = 0; kh < K; ++kh) {
      for (int kw = 0; kw < K; ++kw) {
        const int ih = h0 + kh;
        const int iw = w0 + kw;
        if (!in_spatial(ih, H) || !in_spatial(iw, W)) continue;
        const float xv = x[(((long)n * IC + ic) * H + ih) * W + iw];
        const float wv = w[(((long)oc * IC + ic) * K + kh) * K + kw];
        sum += xv * wv;
      }
    }
  }
  y[idx] = sum;
}


kernel void conv2d_backward_data(
    device float* dx [[buffer(0)]],
    const device float* dy [[buffer(1)]],
    const device float* w [[buffer(2)]],
    constant int& N [[buffer(3)]],
    constant int& IC [[buffer(4)]],
    constant int& OC [[buffer(5)]],
    constant int& H [[buffer(6)]],
    constant int& W [[buffer(7)]],
    constant int& K [[buffer(8)]],
    constant int& OH [[buffer(9)]],
    constant int& OW [[buffer(10)]],
    constant int& stride [[buffer(11)]],
    constant int& pad [[buffer(12)]],
    uint gid [[thread_position_in_grid]]) {
  const long total = (long)N * IC * H * W;
  const long idx = (long)gid;
  if (idx >= total) return;
  const int iw = (int)(idx % W);
  const int ih = (int)((idx / W) % H);
  const int ic = (int)((idx / ((long)W * H)) % IC);
  const int n = (int)(idx / ((long)W * H * IC));
  float sum = 0.f;
  for (int oc = 0; oc < OC; ++oc) {
    for (int kh = 0; kh < K; ++kh) {
      const int h_tmp = ih + pad - kh;
      if (h_tmp % stride != 0) continue;
      const int oh = h_tmp / stride;
      if (!in_spatial(oh, OH)) continue;
      for (int kw = 0; kw < K; ++kw) {
        const int w_tmp = iw + pad - kw;
        if (w_tmp % stride != 0) continue;
        const int ow = w_tmp / stride;
        if (!in_spatial(ow, OW)) continue;
        const float g = dy[(((long)n * OC + oc) * OH + oh) * OW + ow];
        const float wv = w[(((long)oc * IC + ic) * K + kh) * K + kw];
        sum += g * wv;
      }
    }
  }
  dx[idx] = sum;
}


kernel void conv2d_backward_filter(
    device float* dw [[buffer(0)]],
    const device float* x [[buffer(1)]],
    const device float* dy [[buffer(2)]],
    constant int& N [[buffer(3)]],
    constant int& IC [[buffer(4)]],
    constant int& OC [[buffer(5)]],
    constant int& H [[buffer(6)]],
    constant int& W [[buffer(7)]],
    constant int& K [[buffer(8)]],
    constant int& OH [[buffer(9)]],
    constant int& OW [[buffer(10)]],
    constant int& stride [[buffer(11)]],
    constant int& pad [[buffer(12)]],
    uint gid [[thread_position_in_grid]]) {
  const long total = (long)OC * IC * K * K;
  const long idx = (long)gid;
  if (idx >= total) return;
  const int kw = (int)(idx % K);
  const int kh = (int)((idx / K) % K);
  const int ic = (int)((idx / ((long)K * K)) % IC);
  const int oc = (int)(idx / ((long)K * K * IC));
  float sum = 0.f;
  for (int n = 0; n < N; ++n) {
    for (int oh = 0; oh < OH; ++oh) {
      const int ih = oh * stride + kh - pad;
      if (!in_spatial(ih, H)) continue;
      for (int ow = 0; ow < OW; ++ow) {
        const int iw = ow * stride + kw - pad;
        if (!in_spatial(iw, W)) continue;
        const float xv = x[(((long)n * IC + ic) * H + ih) * W + iw];
        const float g = dy[(((long)n * OC + oc) * OH + oh) * OW + ow];
        sum += xv * g;
      }
    }
  }
  dw[idx] += sum;
}


kernel void conv_bias_grad(
    const device float* dy [[buffer(0)]],
    device float* grad_bias [[buffer(1)]],
    constant int& N [[buffer(2)]],
    constant int& OC [[buffer(3)]],
    constant int& OH [[buffer(4)]],
    constant int& OW [[buffer(5)]],
    threadgroup float* buffer [[threadgroup(0)]],
    uint t_tid [[thread_position_in_threadgroup]],
    uint t_tgs [[threads_per_threadgroup]],
    uint t_grp [[threadgroup_position_in_grid]]) {
  const int oc = (int)t_grp;
  if (oc >= OC) return;
  const int spatial = OH * OW;
  float sum = 0.f;
  const int threads = (int)t_tgs;
  for (int i = (int)t_tid; i < N * spatial; i += threads) {
    const int n = i / spatial;
    const int hw = i - n * spatial;
    sum += dy[(((long)n * OC + oc) * OH * OW) + hw];
  }
  sum = block_sum(sum, buffer, t_tid, t_tgs);
  if (t_tid == 0) grad_bias[oc] += sum;
}


kernel void max_pool_forward(
    device float* y [[buffer(0)]],
    device int* index [[buffer(1)]],
    const device float* x [[buffer(2)]],
    constant int& N [[buffer(3)]],
    constant int& C [[buffer(4)]],
    constant int& H [[buffer(5)]],
    constant int& W [[buffer(6)]],
    constant int& K [[buffer(7)]],
    constant int& OH [[buffer(8)]],
    constant int& OW [[buffer(9)]],
    constant int& stride [[buffer(10)]],
    uint gid [[thread_position_in_grid]]) {
  const long total = (long)N * C * OH * OW;
  const long idx = (long)gid;
  if (idx >= total) return;
  const int ow = (int)(idx % OW);
  const int oh = (int)((idx / OW) % OH);
  const int c = (int)((idx / ((long)OW * OH)) % C);
  const int n = (int)(idx / ((long)OW * OH * C));
  const int h0 = oh * stride;
  const int w0 = ow * stride;
  float best = -INFINITY;
  int best_i = 0;
  for (int kh = 0; kh < K; ++kh) {
    for (int kw = 0; kw < K; ++kw) {
      const int ih = h0 + kh;
      const int iw = w0 + kw;
      if (!in_spatial(ih, H) || !in_spatial(iw, W)) continue;
      const int flat = ((ih * W) + iw);
      const float value = x[(((long)n * C + c) * H + ih) * W + iw];
      if (value > best) {
        best = value;
        best_i = flat;
      }
    }
  }
  y[idx] = best;
  index[idx] = best_i;
}


kernel void max_pool_backward(
    device float* dx [[buffer(0)]],
    const device float* dy [[buffer(1)]],
    const device int* index [[buffer(2)]],
    constant int& N [[buffer(3)]],
    constant int& C [[buffer(4)]],
    constant int& H [[buffer(5)]],
    constant int& W [[buffer(6)]],
    constant int& OH [[buffer(7)]],
    constant int& OW [[buffer(8)]],
    uint gid [[thread_position_in_grid]]) {
  const long total = (long)N * C * OH * OW;
  const long idx = (long)gid;
  if (idx >= total) return;
  const int c = (int)((idx / ((long)OW * OH)) % C);
  const int n = (int)(idx / ((long)OW * OH * C));
  const int flat = index[idx];
  const int ih = flat / W;
  const int iw = flat - ih * W;
  atomic_add_f(dx + (((long)n * C + c) * H + ih) * W + iw, dy[idx]);
}


kernel void global_avg_forward(
    device float* y [[buffer(0)]],
    const device float* x [[buffer(1)]],
    constant int& N [[buffer(2)]],
    constant int& C [[buffer(3)]],
    constant int& H [[buffer(4)]],
    constant int& W [[buffer(5)]],
    threadgroup float* buffer [[threadgroup(0)]],
    uint t_tid [[thread_position_in_threadgroup]],
    uint t_tgs [[threads_per_threadgroup]],
    uint t_grp [[threadgroup_position_in_grid]]) {
  const int nc = (int)t_grp;
  if (nc >= N * C) return;
  const int c = nc % C;
  const int n = nc / C;
  const int spatial = H * W;
  float sum = 0.f;
  for (int i = (int)t_tid; i < spatial; i += (int)t_tgs) {
    sum += x[(((long)n * C + c) * H * W) + i];
  }
  sum = block_sum(sum, buffer, t_tid, t_tgs);
  if (t_tid == 0) y[nc] = divf(sum, (float)spatial);
}


kernel void global_avg_backward(
    device float* dx [[buffer(0)]],
    const device float* dy [[buffer(1)]],
    constant int& N [[buffer(2)]],
    constant int& C [[buffer(3)]],
    constant int& H [[buffer(4)]],
    constant int& W [[buffer(5)]],
    uint gid [[thread_position_in_grid]]) {
  const long spatial = (long)H * W;
  const long total = (long)N * C * spatial;
  const long idx = (long)gid;
  if (idx >= total) return;
  const int hw = (int)(idx % spatial);
  const int c = (int)((idx / spatial) % C);
  const int n = (int)(idx / (spatial * C));
  (void)hw;
  dx[idx] = divf(dy[(long)n * C + c], (float)spatial);
}


kernel void batch_norm_train(
    device float* y [[buffer(0)]],
    const device float* x [[buffer(1)]],
    const device float* gamma [[buffer(2)]],
    const device float* beta [[buffer(3)]],
    device float* running_mean [[buffer(4)]],
    device float* running_var [[buffer(5)]],
    device float* saved_mean [[buffer(6)]],
    device float* saved_inv [[buffer(7)]],
    constant int& N [[buffer(8)]],
    constant int& C [[buffer(9)]],
    constant int& H [[buffer(10)]],
    constant int& W [[buffer(11)]],
    constant float& eps [[buffer(12)]],
    constant float& factor [[buffer(13)]],
    threadgroup float* buffer [[threadgroup(0)]],
    uint t_tid [[thread_position_in_threadgroup]],
    uint t_tgs [[threads_per_threadgroup]],
    uint t_grp [[threadgroup_position_in_grid]]) {
  const int c = (int)t_grp;
  if (c >= C) return;
  const int spatial = N * H * W;
  const int hw_plane = H * W;
  float sum = 0.f;
  for (int s = (int)t_tid; s < spatial; s += (int)t_tgs) {
    const int n = s / hw_plane;
    const int hw = s - n * hw_plane;
    sum += x[((long)n * C + c) * hw_plane + hw];
  }
  sum = block_sum(sum, buffer, t_tid, t_tgs);
  const float mean = divf(sum, (float)spatial);
  float var_sum = 0.f;
  for (int s = (int)t_tid; s < spatial; s += (int)t_tgs) {
    const int n = s / hw_plane;
    const int hw = s - n * hw_plane;
    const float diff = x[((long)n * C + c) * hw_plane + hw] - mean;
    var_sum += diff * diff;
  }
  var_sum = block_sum(var_sum, buffer, t_tid, t_tgs);
  const float var = divf(var_sum, (float)spatial);
  const float inv = rsqrt(var + eps);
  if (t_tid == 0) {
    saved_mean[c] = mean;
    saved_inv[c] = inv;
    running_mean[c] = (1.f - factor) * running_mean[c] + factor * mean;
    const float unbiased = var * divf((float)spatial, (float)(spatial - 1));
    running_var[c] = (1.f - factor) * running_var[c] + factor * unbiased;
  }
  threadgroup_barrier(mem_flags::mem_device);
  for (int s = (int)t_tid; s < spatial; s += (int)t_tgs) {
    const int n = s / hw_plane;
    const int hw = s - n * hw_plane;
    const long index = ((long)n * C + c) * hw_plane + hw;
    const float xhat = (x[index] - mean) * inv;
    y[index] = xhat * gamma[c] + beta[c];
  }
}


kernel void batch_norm_eval(
    device float* y [[buffer(0)]],
    const device float* x [[buffer(1)]],
    const device float* gamma [[buffer(2)]],
    const device float* beta [[buffer(3)]],
    const device float* running_mean [[buffer(4)]],
    const device float* running_var [[buffer(5)]],
    constant int& N [[buffer(6)]],
    constant int& C [[buffer(7)]],
    constant int& H [[buffer(8)]],
    constant int& W [[buffer(9)]],
    constant float& eps [[buffer(10)]],
    uint gid [[thread_position_in_grid]]) {
  const long spatial = (long)H * W;
  const long total = (long)N * C * spatial;
  const long idx = (long)gid;
  if (idx >= total) return;
  const int hw = (int)(idx % spatial);
  const int c = (int)((idx / spatial) % C);
  const int n = (int)(idx / (spatial * C));
  (void)hw;
  (void)n;
  const float inv = rsqrt(running_var[c] + eps);
  const float xhat = (x[idx] - running_mean[c]) * inv;
  y[idx] = xhat * gamma[c] + beta[c];
}


kernel void batch_norm_backward(
    device float* dx [[buffer(0)]],
    const device float* dy [[buffer(1)]],
    const device float* x [[buffer(2)]],
    const device float* gamma [[buffer(3)]],
    const device float* saved_mean [[buffer(4)]],
    const device float* saved_inv [[buffer(5)]],
    device float* grad_gamma [[buffer(6)]],
    device float* grad_beta [[buffer(7)]],
    constant int& N [[buffer(8)]],
    constant int& C [[buffer(9)]],
    constant int& H [[buffer(10)]],
    constant int& W [[buffer(11)]],
    threadgroup float* buffer [[threadgroup(0)]],
    uint t_tid [[thread_position_in_threadgroup]],
    uint t_tgs [[threads_per_threadgroup]],
    uint t_grp [[threadgroup_position_in_grid]]) {
  const int c = (int)t_grp;
  if (c >= C) return;
  const int spatial = N * H * W;
  const int hw_plane = H * W;
  const float mean = saved_mean[c];
  const float inv = saved_inv[c];
  const float scale = gamma[c];
  float sum_dhat = 0.f;
  float sum_dhat_xhat = 0.f;
  for (int s = (int)t_tid; s < spatial; s += (int)t_tgs) {
    const int n = s / hw_plane;
    const int hw = s - n * hw_plane;
    const long index = ((long)n * C + c) * hw_plane + hw;
    const float xhat = (x[index] - mean) * inv;
    const float dhat = dy[index] * scale;
    sum_dhat += dhat;
    sum_dhat_xhat += dhat * xhat;
  }
  sum_dhat = block_sum(sum_dhat, buffer, t_tid, t_tgs);
  sum_dhat_xhat = block_sum(sum_dhat_xhat, buffer, t_tid, t_tgs);
  const float mean_dhat = divf(sum_dhat, (float)spatial);
  const float mean_dhat_xhat = divf(sum_dhat_xhat, (float)spatial);
  if (t_tid == 0) {
    float g_sum = 0.f;
    float b_sum = 0.f;
    for (int s = 0; s < spatial; ++s) {
      const int n = s / hw_plane;
      const int hw = s - n * hw_plane;
      const long index = ((long)n * C + c) * hw_plane + hw;
      const float xhat = (x[index] - mean) * inv;
      g_sum += dy[index] * xhat;
      b_sum += dy[index];
    }
    grad_gamma[c] += g_sum;
    grad_beta[c] += b_sum;
  }
  for (int s = (int)t_tid; s < spatial; s += (int)t_tgs) {
    const int n = s / hw_plane;
    const int hw = s - n * hw_plane;
    const long index = ((long)n * C + c) * hw_plane + hw;
    const float xhat = (x[index] - mean) * inv;
    const float dhat = dy[index] * scale;
    dx[index] = inv * (dhat - mean_dhat - xhat * mean_dhat_xhat);
  }
}


kernel void cast_f32_to_bf16(
    device ushort* dst [[buffer(0)]],
    const device float* src [[buffer(1)]],
    constant long& n [[buffer(2)]],
    uint gid [[thread_position_in_grid]]) {
  const long i = (long)gid;
  if (i < n) dst[i] = f32_to_bf16(src[i]);
}


kernel void cast_bf16_to_f32(
    device float* dst [[buffer(0)]],
    const device ushort* src [[buffer(1)]],
    constant long& n [[buffer(2)]],
    uint gid [[thread_position_in_grid]]) {
  const long i = (long)gid;
  if (i < n) dst[i] = bf16_to_f32(src[i]);
}


kernel void add_bias_bf16(
    device ushort* y [[buffer(0)]],
    const device float* bias [[buffer(1)]],
    constant int& rows [[buffer(2)]],
    constant int& cols [[buffer(3)]],
    uint gid [[thread_position_in_grid]]) {
  const long n = (long)rows * cols;
  const long i = (long)gid;
  if (i >= n) return;
  y[i] = f32_to_bf16(bf16_to_f32(y[i]) + bias[i % cols]);
}


kernel void bias_grad_bf16(
    const device ushort* grad_y [[buffer(0)]],
    device float* grad_bias [[buffer(1)]],
    constant int& rows [[buffer(2)]],
    constant int& cols [[buffer(3)]],
    threadgroup float* buffer [[threadgroup(0)]],
    uint t_tid [[thread_position_in_threadgroup]],
    uint t_tgs [[threads_per_threadgroup]],
    uint t_grp [[threadgroup_position_in_grid]]) {
  const int col = (int)t_grp;
  if (col >= cols) return;
  float sum = 0.f;
  for (int row = (int)t_tid; row < rows; row += (int)t_tgs) {
    sum += bf16_to_f32(grad_y[(long)row * cols + col]);
  }
  sum = block_sum(sum, buffer, t_tid, t_tgs);
  if (t_tid == 0) grad_bias[col] += sum;
}


kernel void gemm_nt_bf16(
    device ushort* C [[buffer(0)]],
    const device ushort* A [[buffer(1)]],
    const device ushort* B [[buffer(2)]],
    constant int& M [[buffer(3)]],
    constant int& N [[buffer(4)]],
    constant int& K [[buffer(5)]],
    constant int& batches [[buffer(6)]],
    constant int& ldc [[buffer(7)]],
    constant int& lda [[buffer(8)]],
    constant int& ldb [[buffer(9)]],
    constant long& stride_c [[buffer(10)]],
    constant long& stride_a [[buffer(11)]],
    constant long& stride_b [[buffer(12)]],
    constant float& alpha [[buffer(13)]],
    constant float& beta [[buffer(14)]],
    uint3 gid [[thread_position_in_grid]]) {
  const int j = (int)gid.x;
  const int i = (int)gid.y;
  const int b = (int)gid.z;
  if (i >= M || j >= N || b >= batches) return;
  const device ushort* a = A + b * stride_a + (long)i * lda;
  const device ushort* brow = B + b * stride_b + (long)j * ldb;
  float sum = 0.f;
  for (int p = 0; p < K; ++p) sum += bf16_to_f32(a[p]) * bf16_to_f32(brow[p]);
  device ushort* c = C + b * stride_c + (long)i * ldc + j;
  float out = alpha * sum;
  if (beta != 0.f) out += beta * bf16_to_f32(*c);
  *c = f32_to_bf16(out);
}


kernel void gemm_nt_bf16_f32(
    device float* C [[buffer(0)]],
    const device ushort* A [[buffer(1)]],
    const device ushort* B [[buffer(2)]],
    constant int& M [[buffer(3)]],
    constant int& N [[buffer(4)]],
    constant int& K [[buffer(5)]],
    constant int& batches [[buffer(6)]],
    constant int& ldc [[buffer(7)]],
    constant int& lda [[buffer(8)]],
    constant int& ldb [[buffer(9)]],
    constant long& stride_c [[buffer(10)]],
    constant long& stride_a [[buffer(11)]],
    constant long& stride_b [[buffer(12)]],
    constant float& alpha [[buffer(13)]],
    constant float& beta [[buffer(14)]],
    uint3 gid [[thread_position_in_grid]]) {
  const int j = (int)gid.x;
  const int i = (int)gid.y;
  const int b = (int)gid.z;
  if (i >= M || j >= N || b >= batches) return;
  const device ushort* a = A + b * stride_a + (long)i * lda;
  const device ushort* brow = B + b * stride_b + (long)j * ldb;
  float sum = 0.f;
  for (int p = 0; p < K; ++p) sum += bf16_to_f32(a[p]) * bf16_to_f32(brow[p]);
  device float* c = C + b * stride_c + (long)i * ldc + j;
  *c = (beta == 0.f) ? alpha * sum : alpha * sum + beta * (*c);
}


kernel void gemm_nn_bf16(
    device ushort* C [[buffer(0)]],
    const device ushort* A [[buffer(1)]],
    const device ushort* B [[buffer(2)]],
    constant int& M [[buffer(3)]],
    constant int& N [[buffer(4)]],
    constant int& K [[buffer(5)]],
    constant int& batches [[buffer(6)]],
    constant int& ldc [[buffer(7)]],
    constant int& lda [[buffer(8)]],
    constant int& ldb [[buffer(9)]],
    constant long& stride_c [[buffer(10)]],
    constant long& stride_a [[buffer(11)]],
    constant long& stride_b [[buffer(12)]],
    constant float& alpha [[buffer(13)]],
    constant float& beta [[buffer(14)]],
    uint3 gid [[thread_position_in_grid]]) {
  const int j = (int)gid.x;
  const int i = (int)gid.y;
  const int b = (int)gid.z;
  if (i >= M || j >= N || b >= batches) return;
  const device ushort* a = A + b * stride_a + (long)i * lda;
  const device ushort* bbase = B + b * stride_b;
  float sum = 0.f;
  for (int p = 0; p < K; ++p) sum += bf16_to_f32(a[p]) * bf16_to_f32(bbase[(long)p * ldb + j]);
  device ushort* c = C + b * stride_c + (long)i * ldc + j;
  float out = alpha * sum;
  if (beta != 0.f) out += beta * bf16_to_f32(*c);
  *c = f32_to_bf16(out);
}


kernel void gemm_nn_bf16_f32(
    device float* C [[buffer(0)]],
    const device ushort* A [[buffer(1)]],
    const device ushort* B [[buffer(2)]],
    constant int& M [[buffer(3)]],
    constant int& N [[buffer(4)]],
    constant int& K [[buffer(5)]],
    constant int& batches [[buffer(6)]],
    constant int& ldc [[buffer(7)]],
    constant int& lda [[buffer(8)]],
    constant int& ldb [[buffer(9)]],
    constant long& stride_c [[buffer(10)]],
    constant long& stride_a [[buffer(11)]],
    constant long& stride_b [[buffer(12)]],
    constant float& alpha [[buffer(13)]],
    constant float& beta [[buffer(14)]],
    uint3 gid [[thread_position_in_grid]]) {
  const int j = (int)gid.x;
  const int i = (int)gid.y;
  const int b = (int)gid.z;
  if (i >= M || j >= N || b >= batches) return;
  const device ushort* a = A + b * stride_a + (long)i * lda;
  const device ushort* bbase = B + b * stride_b;
  float sum = 0.f;
  for (int p = 0; p < K; ++p) sum += bf16_to_f32(a[p]) * bf16_to_f32(bbase[(long)p * ldb + j]);
  device float* c = C + b * stride_c + (long)i * ldc + j;
  *c = (beta == 0.f) ? alpha * sum : alpha * sum + beta * (*c);
}


kernel void gemm_tn_bf16(
    device ushort* C [[buffer(0)]],
    const device ushort* A [[buffer(1)]],
    const device ushort* B [[buffer(2)]],
    constant int& M [[buffer(3)]],
    constant int& N [[buffer(4)]],
    constant int& K [[buffer(5)]],
    constant int& batches [[buffer(6)]],
    constant int& ldc [[buffer(7)]],
    constant int& lda [[buffer(8)]],
    constant int& ldb [[buffer(9)]],
    constant long& stride_c [[buffer(10)]],
    constant long& stride_a [[buffer(11)]],
    constant long& stride_b [[buffer(12)]],
    constant float& alpha [[buffer(13)]],
    constant float& beta [[buffer(14)]],
    uint3 gid [[thread_position_in_grid]]) {
  const int j = (int)gid.x;
  const int i = (int)gid.y;
  const int b = (int)gid.z;
  if (i >= M || j >= N || b >= batches) return;
  const device ushort* abase = A + b * stride_a;
  const device ushort* bbase = B + b * stride_b;
  float sum = 0.f;
  for (int p = 0; p < K; ++p) {
    sum += bf16_to_f32(abase[(long)p * lda + i]) * bf16_to_f32(bbase[(long)p * ldb + j]);
  }
  device ushort* c = C + b * stride_c + (long)i * ldc + j;
  float out = alpha * sum;
  if (beta != 0.f) out += beta * bf16_to_f32(*c);
  *c = f32_to_bf16(out);
}


kernel void gemm_tn_bf16_f32(
    device float* C [[buffer(0)]],
    const device ushort* A [[buffer(1)]],
    const device ushort* B [[buffer(2)]],
    constant int& M [[buffer(3)]],
    constant int& N [[buffer(4)]],
    constant int& K [[buffer(5)]],
    constant int& batches [[buffer(6)]],
    constant int& ldc [[buffer(7)]],
    constant int& lda [[buffer(8)]],
    constant int& ldb [[buffer(9)]],
    constant long& stride_c [[buffer(10)]],
    constant long& stride_a [[buffer(11)]],
    constant long& stride_b [[buffer(12)]],
    constant float& alpha [[buffer(13)]],
    constant float& beta [[buffer(14)]],
    uint3 gid [[thread_position_in_grid]]) {
  const int j = (int)gid.x;
  const int i = (int)gid.y;
  const int b = (int)gid.z;
  if (i >= M || j >= N || b >= batches) return;
  const device ushort* abase = A + b * stride_a;
  const device ushort* bbase = B + b * stride_b;
  float sum = 0.f;
  for (int p = 0; p < K; ++p) {
    sum += bf16_to_f32(abase[(long)p * lda + i]) * bf16_to_f32(bbase[(long)p * ldb + j]);
  }
  device float* c = C + b * stride_c + (long)i * ldc + j;
  *c = (beta == 0.f) ? alpha * sum : alpha * sum + beta * (*c);
}



kernel void fill_f(device float* dst [[buffer(0)]], constant long& n [[buffer(1)]],
                   constant float& value [[buffer(2)]], uint gid [[thread_position_in_grid]]) {
  if ((long)gid < n) dst[gid] = value;
}

kernel void fill_i(device int* dst [[buffer(0)]], constant long& n [[buffer(1)]],
                   constant int& value [[buffer(2)]], uint gid [[thread_position_in_grid]]) {
  if ((long)gid < n) dst[gid] = value;
}

kernel void fill_u8(device uchar* dst [[buffer(0)]], constant long& n [[buffer(1)]],
                    constant int& value [[buffer(2)]], uint gid [[thread_position_in_grid]]) {
  if ((long)gid < n) dst[gid] = (uchar)value;
}
