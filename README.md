# nnAGX

Metal implementation of the [busysteve/nnCUDA](https://github.com/busysteve/nnCUDA) neural-net API for Apple AGX GPUs. The static library target is still named `nn`, and the public headers live in `nn/`. The shared contract is [docs/api.md](docs/api.md). [busysteve/llm](https://github.com/busysteve/llm) links against it by pointing `NN_SOURCE` at this tree, or by running `./build.sh agx`.

Dense float32 parameters, bfloat16 retained activations, ranks 1–4, NCHW convolutions. One in-order Metal command queue per `nn::Context`. A device is used only when its threadgroup holds at least 256 threads and 1024 bytes of threadgroup memory, because the kernels launch 256-wide groups.

This tree configures only on Apple platforms. It needs Xcode or the Metal command-line tools, because the library compiles `src/kernels.metal` at runtime with `newLibraryWithSource`. Fast math is off.

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
./build/nn_tests shapes
./build/nn_tests grads
./build/nn_tests ckpt
```

`Context()` uses device 0 from `MTLCopyAllDevices` after that filter. `Context(int device)` uses that index. `device_count()` and `device_name(int)` report the same list. An index outside the list throws `device`.

`detect_devices()` returns that list. `select_device("auto")` and `Context("auto")` use the first entry. `select_device("1")` and `Context("1")` select that index by hand. A blank choice, a word other than `auto`, or an index past the list throws `device`.

Buffers use shared storage. `synchronize()` ends the open compute encoder, commits the command buffer, and waits. `IndexTensor::to_host()` returns the index buffer. Read tensors with `to_host()` as well. `data()` is a stuffed buffer handle, not a host pointer.

Bfloat16 GEMMs load the narrow bits and accumulate in float32. The conversion is the same round-to-nearest-even rule as the other nn libraries, including infinities and NaNs. Apple's hardware bfloat type is not required, so the path is the same on M1 through M4. Sums of squares use the float reduction and return it as `double`. Apple GPUs have no device float64.

An `nn::Error` from a failed Metal call looks like `newLibraryWithSource: MTL_LIBRARY_ERROR` followed by the compiler log.

To build llm against nnAGX, configure llm with `-DNN_SOURCE=/path/to/nnAGX` on a Mac. llm itself stays C++ (`project(llm LANGUAGES CXX)`). The nested nnAGX build enables Objective-C++.
