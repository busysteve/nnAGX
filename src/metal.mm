#include "metal.hpp"

#include "activate.hpp"
#include "nn/error.hpp"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace nn {
namespace {

std::string clip(std::string text) {
  if (text.size() > 2000) text.resize(2000);
  return text;
}

std::string ns_text(NSString* text) {
  if (text == nil) return {};
  const char* utf = text.UTF8String;
  if (utf == nullptr) return {};
  return utf;
}

bool usable_device(id<MTLDevice> dev) {
  return dev != nil && dev.maxTotalThreadsPerThreadgroup >= 256 &&
         dev.maxThreadgroupMemoryLength >= 1024;
}

id<MTLBuffer> buffer_of(Buffer mem) { return (__bridge id<MTLBuffer>)mem; }

size_t round_up(size_t n, size_t local) { return ((n + local - 1) / local) * local; }

}  // namespace

struct Context::Impl {
  id<MTLDevice> device = nil;
  id<MTLCommandQueue> queue = nil;
  id<MTLLibrary> library = nil;
  NSMutableDictionary<NSString*, id<MTLComputePipelineState>>* pipelines = nil;
  mutable id<MTLCommandBuffer> cmd = nil;
  mutable id<MTLComputeCommandEncoder> encoder = nil;
  Buffer dummy = nullptr;
  size_t gemm_lx = 16;
  size_t gemm_ly = 16;
  DType activation = DType::BFloat16;
  std::vector<ArenaBlock> f32_arena;
  std::vector<ArenaBlock> bf16_arena;

  void open(int index);
  ~Impl();
  void finish(bool report) const;
  id<MTLCommandBuffer> command_buffer() const;
  id<MTLComputeCommandEncoder> compute() const;
  void end_compute() const;
  id<MTLComputePipelineState> pipeline(const char* name);
};

void Context::Impl::finish(bool report) const {
  if (encoder != nil) {
    [encoder endEncoding];
    encoder = nil;
  }
  if (cmd == nil) return;
  [cmd commit];
  [cmd waitUntilCompleted];
  const bool failed = cmd.status == MTLCommandBufferStatusError;
  std::string detail = failed ? ns_text(cmd.error.localizedDescription) : std::string();
  cmd = nil;
  if (failed && report) {
    std::string message = "commandBuffer: MTLCommandBufferStatusError";
    detail = clip(std::move(detail));
    if (!detail.empty()) message += " " + detail;
    throw Error(message);
  }
}

id<MTLCommandBuffer> Context::Impl::command_buffer() const {
  if (cmd == nil) {
    cmd = [queue commandBuffer];
    if (cmd == nil) throw Error("commandBuffer: MTL_OUT_OF_MEMORY");
  }
  return cmd;
}

void Context::Impl::end_compute() const {
  if (encoder == nil) return;
  [encoder endEncoding];
  encoder = nil;
}

id<MTLComputeCommandEncoder> Context::Impl::compute() const {
  (void)command_buffer();
  if (encoder == nil) {
    encoder = [cmd computeCommandEncoder];
    if (encoder == nil) throw Error("computeCommandEncoder: MTL_OUT_OF_MEMORY");
  }
  return encoder;
}

id<MTLComputePipelineState> Context::Impl::pipeline(const char* name) {
  @autoreleasepool {
    NSString* key = [NSString stringWithUTF8String:name];
    id<MTLComputePipelineState> found = pipelines[key];
    if (found != nil) return found;
    id<MTLFunction> function = [library newFunctionWithName:key];
    if (function == nil) throw Error(std::string("newFunctionWithName: INVALID_FUNCTION ") + name);
    NSError* error = nil;
    id<MTLComputePipelineState> created =
        [device newComputePipelineStateWithFunction:function error:&error];
    if (created == nil) {
      throw Error(clip(std::string("newComputePipelineState: MTL_PIPELINE_ERROR ") +
                       ns_text(error.localizedDescription)));
    }
    pipelines[key] = created;
    return created;
  }
}

void Context::Impl::open(int index) {
  @autoreleasepool {
    if (index < 0) throw Error("device");
    NSArray<id<MTLDevice>>* all = MTLCopyAllDevices();
    id<MTLDevice> chosen = nil;
    int count = 0;
    for (id<MTLDevice> dev in all) {
      if (!usable_device(dev)) continue;
      if (count == index) chosen = dev;
      ++count;
    }
    if (chosen == nil || index >= count) throw Error("device");
    device = chosen;
    queue = [device newCommandQueue];
    if (queue == nil) throw Error("newCommandQueue: MTL_OUT_OF_MEMORY");
    pipelines = [NSMutableDictionary dictionary];

    std::ifstream in(NNAGX_KERNELS_FILE);
    if (!in) throw Error("Metal program");
    std::ostringstream source;
    source << in.rdbuf();
    NSString* text = [NSString stringWithUTF8String:source.str().c_str()];
    if (text == nil) throw Error("Metal program");
    MTLCompileOptions* options = [MTLCompileOptions new];
    options.fastMathEnabled = NO;
    NSError* error = nil;
    library = [device newLibraryWithSource:text options:options error:&error];
    if (library == nil) {
      throw Error(clip(std::string("newLibraryWithSource: MTL_LIBRARY_ERROR ") +
                       ns_text(error.localizedDescription)));
    }

    gemm_lx = 16;
    gemm_ly = 16;
    id<MTLBuffer> zero = [device newBufferWithLength:sizeof(float) options:MTLResourceStorageModeShared];
    if (zero == nil || zero.contents == nullptr) throw Error("newBufferWithLength: MTL_OUT_OF_MEMORY");
    *static_cast<float*>(zero.contents) = 0.f;
    dummy = (__bridge_retained void*)zero;
  }
}

Context::Impl::~Impl() {
  try {
    finish(false);
  } catch (...) {
  }
  for (const ArenaBlock& block : f32_arena) free_mem(static_cast<Buffer>(block.ptr));
  for (const ArenaBlock& block : bf16_arena) free_mem(static_cast<Buffer>(block.ptr));
  f32_arena.clear();
  bf16_arena.clear();
  free_mem(dummy);
  dummy = nullptr;
}

namespace {

class KernelCall {
 public:
  KernelCall(Context& ctx, const char* name)
      : impl_(ContextAccess::get(ctx)),
        enc_(impl_.compute()),
        pipe_(impl_.pipeline(name)) {
    [enc_ setComputePipelineState:pipe_];
  }

  KernelCall& mem(Buffer value) { return set_buffer(value); }
  KernelCall& i(int value) { return set_bytes(&value, sizeof(value)); }
  KernelCall& l(int64_t value) { return set_bytes(&value, sizeof(value)); }
  KernelCall& f(float value) { return set_bytes(&value, sizeof(value)); }

  KernelCall& local(size_t bytes) {
    if (local_set_) throw Error("setThreadgroupMemoryLength: INVALID_ARG_INDEX");
    local_set_ = true;
    [enc_ setThreadgroupMemoryLength:bytes atIndex:0];
    return *this;
  }

  void over(int64_t n, int local_size) {
    if (n <= 0) return;
    const size_t local = static_cast<size_t>(local_size);
    const size_t groups = (static_cast<size_t>(n) + local - 1) / local;
    dispatch(MTLSizeMake(groups, 1, 1), MTLSizeMake(local, 1, 1));
  }

  void groups(int n_groups, int local_size) {
    if (n_groups <= 0) return;
    dispatch(MTLSizeMake(static_cast<NSUInteger>(n_groups), 1, 1),
             MTLSizeMake(static_cast<NSUInteger>(local_size), 1, 1));
  }

  void nd(const size_t* global, const size_t* local, unsigned dim) {
    if (dim != 3 || local[0] == 0 || local[1] == 0 || local[2] == 0) {
      throw Error("dispatchThreadgroups: INVALID_DIMENSION");
    }
    dispatch(MTLSizeMake(global[0] / local[0], global[1] / local[1], global[2] / local[2]),
             MTLSizeMake(local[0], local[1], local[2]));
  }

 private:
  KernelCall& set_buffer(Buffer value) {
    if (local_set_) throw Error("setBuffer: INVALID_ARG_INDEX");
    id<MTLBuffer> buf = buffer_of(value);
    if (buf == nil) throw Error("setBuffer: INVALID_BUFFER");
    [enc_ setBuffer:buf offset:0 atIndex:index_++];
    return *this;
  }

  KernelCall& set_bytes(const void* data, size_t bytes) {
    if (local_set_) throw Error("setBytes: INVALID_ARG_INDEX");
    [enc_ setBytes:data length:bytes atIndex:index_++];
    return *this;
  }

  void dispatch(MTLSize groups, MTLSize threads) {
    [enc_ dispatchThreadgroups:groups threadsPerThreadgroup:threads];
  }

  Context::Impl& impl_;
  id<MTLComputeCommandEncoder> enc_ = nil;
  id<MTLComputePipelineState> pipe_ = nil;
  NSUInteger index_ = 0;
  bool local_set_ = false;
};

void enqueue_gemm(Context& ctx, const char* name, float* C, const float* A, const float* B, int M,
                  int N, int K, int batches, int ldc, int lda, int ldb, int64_t stride_c,
                  int64_t stride_a, int64_t stride_b, float alpha, float beta) {
  if (M <= 0 || N <= 0 || batches <= 0) return;
  const auto& impl = ContextAccess::get(ctx);
  KernelCall call(ctx, name);
  call.mem(mem_of(C))
      .mem(mem_of(A))
      .mem(mem_of(B))
      .i(M)
      .i(N)
      .i(K)
      .i(batches)
      .i(ldc)
      .i(lda)
      .i(ldb)
      .l(stride_c)
      .l(stride_a)
      .l(stride_b)
      .f(alpha)
      .f(beta);
  const size_t local[3] = {impl.gemm_lx, impl.gemm_ly, 1};
  const size_t global[3] = {round_up(static_cast<size_t>(N), local[0]),
                            round_up(static_cast<size_t>(M), local[1]),
                            static_cast<size_t>(batches)};
  call.nd(global, local, 3);
}

}  // namespace

int agx_device_count() {
  @autoreleasepool {
    int count = 0;
    NSArray<id<MTLDevice>>* all = MTLCopyAllDevices();
    for (id<MTLDevice> dev in all) {
      if (usable_device(dev)) ++count;
    }
    return count;
  }
}

std::string agx_device_name(int index) {
  @autoreleasepool {
    if (index < 0) throw Error("device");
    int count = 0;
    NSArray<id<MTLDevice>>* all = MTLCopyAllDevices();
    for (id<MTLDevice> dev in all) {
      if (!usable_device(dev)) continue;
      if (count == index) {
        std::string name = ns_text(dev.name);
        if (name.empty()) name = "AGX";
        return name;
      }
      ++count;
    }
    throw Error("device");
  }
}

Buffer alloc_mem(Context& ctx, size_t bytes) {
  if (bytes == 0) bytes = 4;
  @autoreleasepool {
    id<MTLBuffer> buf = [ContextAccess::get(ctx).device newBufferWithLength:bytes
                                                                    options:MTLResourceStorageModeShared];
    if (buf == nil) throw Error("newBufferWithLength: MTL_OUT_OF_MEMORY");
    return (__bridge_retained void*)buf;
  }
}

void free_mem(Buffer mem) noexcept {
  if (mem == nullptr) return;
  id<MTLBuffer> buf = (__bridge_transfer id<MTLBuffer>)mem;
  buf = nil;
}

void write_mem(Context& ctx, Buffer mem, const void* src, size_t bytes) {
  if (bytes == 0) return;
  ctx.synchronize();
  void* dst = [buffer_of(mem) contents];
  if (dst == nullptr) throw Error("contents: MTL_NO_CONTENTS");
  std::memcpy(dst, src, bytes);
}

void read_mem(Context& ctx, Buffer mem, void* dst, size_t bytes) {
  if (bytes == 0) return;
  ctx.synchronize();
  void* src = [buffer_of(mem) contents];
  if (src == nullptr) throw Error("contents: MTL_NO_CONTENTS");
  std::memcpy(dst, src, bytes);
}

void fill_mem_f(Context& ctx, Buffer mem, float value, size_t count) {
  if (count == 0) return;
  KernelCall(ctx, "fill_f").mem(mem).l(static_cast<int64_t>(count)).f(value).over(
      static_cast<int64_t>(count), 256);
}

void fill_mem_i(Context& ctx, Buffer mem, int value, size_t count) {
  if (count == 0) return;
  KernelCall(ctx, "fill_i").mem(mem).l(static_cast<int64_t>(count)).i(value).over(
      static_cast<int64_t>(count), 256);
}

void fill_zero(Context& ctx, Buffer mem, size_t bytes) {
  if (bytes == 0 || mem == nullptr) return;
  KernelCall(ctx, "fill_u8").mem(mem).l(static_cast<int64_t>(bytes)).i(0).over(
      static_cast<int64_t>(bytes), 256);
}

void copy_mem(Context& ctx, Buffer dst, Buffer src, size_t bytes) {
  if (bytes == 0 || dst == src) return;
  auto& impl = ContextAccess::get(ctx);
  (void)impl.command_buffer();
  impl.end_compute();
  @autoreleasepool {
    id<MTLBlitCommandEncoder> blit = [impl.cmd blitCommandEncoder];
    if (blit == nil) throw Error("blitCommandEncoder: MTL_OUT_OF_MEMORY");
    [blit copyFromBuffer:buffer_of(src)
            sourceOffset:0
                toBuffer:buffer_of(dst)
       destinationOffset:0
                    size:bytes];
    [blit endEncoding];
  }
}

void wait_quiet(const Context& ctx) noexcept {
  try {
    ContextAccess::get(ctx).finish(false);
  } catch (...) {
  }
}

Context::Context() : impl_(std::make_unique<Impl>()) { impl_->open(0); }

Context::Context(int device) : impl_(std::make_unique<Impl>()) { impl_->open(device); }

Context::Context(const std::string& choice) : impl_(std::make_unique<Impl>()) {
  impl_->open(select_device(choice));
}

Context::~Context() = default;

void Context::set_activation_dtype(DType dtype) {
  if (dtype != DType::Float32 && dtype != DType::BFloat16) throw Error("dtype");
  impl_->activation = dtype;
}

DType Context::activation_dtype() const { return impl_->activation; }

void Context::synchronize() const { impl_->finish(true); }

#include "metal_launches.inc"

}  // namespace nn
