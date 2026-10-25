# Qwen3.8-27B on AMD (gfx1100) — HIP port notes

Ported the Xolotl (Lagyu/strata-ds41) DeepSeek/Qwen dense engine onto upstream
Niko1221/Strata 0.2.4.2, built for Windows + ROCm 10.2 (TheRock wheels).

## Result

* `build-hip-win/strata-dense.exe` — 9.4 MB, links and runs.
* `strata-dense --selftest` — **36/36 passed**, max rel err ~2e-4 (gdn chunk final state),
  everything else <= 1e-5.

## Files brought over from Xolotl

| file | lines | notes |
| --- | --- | --- |
| `src/core/qwen35.cpp` | 788 | 27B dense runtime (GDN + full attention) |
| `src/kernels/cuda/qwen35_attention.cu` | 223 | attention kernels |
| `src/core/dense_model.cpp` | 1419 | GGUF load, weight placement, prefill/decode |
| `src/kernels/cuda/dense_kernels.cu` | 1137 | rms_norm, swiglu, gdn, gemv, kv cache |
| `src/program/dense_main.cpp` | 1204 | `strata-dense --serve/--selftest/--check` |
| headers | 178+125+112+37 | dense_model.hpp, dense_kernels.hpp, qwen35.hpp, qwen35_attention.hpp |

## Changes needed

### 1. CUDA -> HIP (mechanical)

~4,771 lines, zero `cuda*` left. `cudaMalloc/cudaMemcpy/cudaStream_t/cudaEvent/cudaHostAlloc`
-> `hip*`; `__shfl_*_sync(mask, ...)` -> `__shfl_*(...)` (AMD has no mask operand);
`<cuda_runtime.h>` -> `<hip/hip_runtime.h>`. Kernel keywords (`__global__`, `__shared__`,
`__syncthreads`) needed no change at all.

### 2. `nvcuda::wmma` is not portable

`dense_kernels.cu` used the NVIDIA WMMA API in `attn_chunk_tc_kernel`. Wrapped the whole
tensor-core chunk-attention block (kernel + `launch_attn_chunk_tc` + its dispatch branch)
in `#if !defined(STRATA_NO_WMMA)`, and forced `use_tc = false` when `STRATA_NO_WMMA`.
The file already had a scalar fallback (`attn_chunk_g_kernel`, documented as bit-identical),
so this only costs speed, not correctness — `selftest` confirms.

### 3. `std::regex` does not link through clang++ with the static CRT

`dense_model.cpp` used `std::regex` to expand `-00001-of-0000N.gguf` shard names; it pulled
`__std_regex_transform_primary_char`, which `lld-link` cannot resolve with
`CMAKE_MSVC_RUNTIME_LIBRARY=MultiThreaded`. Rewrote `shard_paths()` as plain string parsing
(no `<regex>` anywhere in the tree now).

### 4. Missing symbol on the upstream tree

`layer_set_qwen35_gdn` / `g_qwen35_gdn` / the `qwen35_gdn_out_norm` dispatch exist only in
Xolotl. Added: the declaration in `layer.hpp`, the flag in `layer.cpp`'s anonymous namespace,
the `else if (g_qwen35_gdn) qwen35_gdn_out_norm(...)` branch before the generic `gdn_out_norm`,
the setter, and `#include "strata/kernels/qwen35_attention.hpp"`.

### 5. CMake

* `strata_kernels`: added `dense_kernels.cu`, `qwen35_attention.cu`.
* `strata_engine`: added `dense_model.cpp`, `qwen35.cpp`.
* New target `strata-dense` (guarded by `STRATA_ENABLE_HIP OR STRATA_ENABLE_CUDA`), linking
  `strata_engine strata_mmq ${_strata_gpu_runtime_target}`, with `STRATA_NO_WMMA=1` propagated
  to `strata-dense`, `strata_engine` and `strata_kernels` when HIP.
* `tools/hip/build_windows.bat` only builds `strata strata-device`, so build the target by hand:
  `cmake --build build-hip-win --target strata-dense`.

## Build recipe (what actually worked)

```bat
rem ROCm from TheRock wheels, VS2022 BuildTools required (14.44, NOT the VS18 14.51 toolset)
set STRATA_HIP_ARCHS=gfx1100
set STRATA_NO_VISION=1
tools\hip\build_windows.bat
cmake --build build-hip-win --target strata-dense
```

### Why MSVC matters

With the **VS18 / MSVC 14.51** toolset every `hipcc` compile fails before reaching our code:

```
error: __device__ function 'isgreater' cannot overload
       __host__ __device__ function 'isgreater'
  __clang_cuda_math_forward_declares.h:97
  note: previous declaration is here
  ...MSVC\14.51.36231\include\cmath:688  _CLANG_BUILTIN2(isgreater)
```

MSVC 14.51's `<cmath>` now defines `isgreater`/`isless`/… as `constexpr` functions under
`#ifdef __clang__`, which collides with ROCm's own `__device__` overloads. **VS2022's 14.44
toolset does not do this**, so it builds. Guarding the ROCm headers or defining
`__CLANG_HIP_CMATH_H__` only moves the failure to `<complex>`; the toolset is the real fix.

## Not done

* `--check` / real inference not yet run (model downloading).
* WMMA disabled -> the tensor-core chunk attention path is off; speed will be below the CUDA build.
* `dense_main.cpp`'s `--serve` server path untested.

## Measured (2026-10-25, gfx1100 / W7800 32GB)

Model: `unsloth/Qwen3.8-27B-UD-Q4_K_M.gguf` (15.33 GB on disk, 14.32 GiB resident)

```
INFO engine=dense-0.2 model=qwen35 layers=64 n_embd=5120
     weights_mib=14664 host_mib=0 kv_mib=208 context=8192 kv=k8v4
strata-dense: 14.32 GiB of weights, 0.20 GiB of KV cache (k8v4)
```

`host_mib=0` — the whole model fits in VRAM, zero offload.

| metric | result |
| --- | --- |
| decode | **19.9 tok/s** (128 tok in 6.43 s) |
| prefill | 55 tok/s (fell back to one token at a time — see below) |
| output quality | correct, fluent English |

### Why decode is ~20 tok/s and not more

The model is **dense** (27B active = 27B read per token), not MoE:

```
14.66 GB per token / 640 GB/s  =  22.9 ms  =>  43.6 tok/s ceiling
measured 19.9  =  46% of ceiling
```

An MoE model at the same VRAM reads only the routed experts per token, which is why
Strata's Qwen3.8-Flash-Next reaches ~90 tok/s here. **This is a property of the
architecture, not of the port.**

### prefill is off, and why

```
dense model: prefill cannot multiply layer 0 ffn_up of GGUF type 11
dense model: prefill cannot multiply layer 0 attn_qkv of GGUF type 13
```

`unsloth`'s **UD** (dynamic) quants mix types per tensor. `mmq::supported()` in
`src/prefill/moe_mmq.cu` covers Q2_0 / IQ2 / IQ3 / IQ4 / Q8_0, and Q4_K/Q5_K/Q5_1/Q6_K
only when `kq_ok()` — which needs `STRATA_MMQ_KQ=1` at run time. **Q3_K (type 11) is not in
the list at all**, so `prefill()` refuses `ffn_up` and the prompt is read one token at a
time through `run()`.

Options, in order of effort:

1. run with `STRATA_MMQ_KQ=1` and a **non-UD** quant (plain `Q4_0` / `Q5_K_M`) whose every
   projection is a type `supported()` covers;
2. add `GGML_TYPE_Q3_K` to `supported()` (needs a matching MMQ kernel — real work);
3. leave it; a 27-token prompt already reads at 55 tok/s.

### MTP (speculative decoding)

`dense_main.cpp` implements MTP (`--mtp --draft-max N`), which needs the separate head
`MTP/mtp-Qwen3.8-27B-Q4_0.gguf` (1.3 GB) from the same repo. Expected 1.5-2x on decode.
Not yet measured.

### MTP sweep (draft-max, --mtp --mtp-force)

Acceptance falls faster than the chain grows, so 1-2 drafts win:

| draft-max | decode | drafts accepted |
| --- | --- | --- |
| off | 19.9 tok/s | - |
| **1** | **26.7 tok/s** | 51/75 (68%) |
| **2** | **26.8 tok/s** | 65/121 (54%) |
| 3 | 24.2 tok/s | 67/174 (39%) |
| 4 | 21.5 tok/s | 70/218 (32%) |
| 5 | 19.3 tok/s | 71/265 (27%) |

Per cycle at draft-max 3: draft 10.5 ms, verify 77.6 ms, ingest 0.1 ms.

Final: **26.8 tok/s** with `--mtp --mtp-force --draft-max 2`, vs 19.9 without MTP (+35%).

### MTP head discovery

`DenseModel::load` looks the nextn block up as `blk.<n_layer>.` across the shard list, so the
head has to be in that list. Added: when `--mtp` is set, scan the trunk's directory for an
`mtp-*.gguf` / `MTP-*.gguf` / `mtp_*.gguf` and append it to the shards. Two heads in one
directory is an error rather than a guess.

## Final numbers (W7800, 32 GB)

| | |
| --- | --- |
| model | Qwen3.8-27B UD-Q4_K_M (14.3 GiB, `host_mib=0`) |
| decode | **26.8 tok/s** (MTP draft-max 2) |
| prefill | 55.8 tok/s (falls back to `run()`, MMQ rejects the UD mix) |
| quality | correct, fluent |

Orchestrate it with `E:\Strata-main\run-dense-27b.bat`.
