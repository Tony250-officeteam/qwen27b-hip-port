# Qwen3.8-27B dense on AMD RDNA3 (gfx1100), Windows + ROCm

A working HIP port of the dense Qwen3.8-27B engine, so it runs on a **Radeon
W7800 / 7900 XTX class card** — hardware the upstream dense path never targeted
(it was written for CUDA).

**Upstream:** [Niko1221/Strata](https://github.com/Niko1221/Strata) (MIT) 0.2.4.2
**Engine source:** the Xolotl fork,
[Lagyu/strata-ds41](https://github.com/Lagyu/strata-ds41) (MIT), which added the
`strata-dense` 27B engine on top of upstream.

The engine is Xolotl's, not mine. This patch makes it build and run on
HIP/Windows.

---

## Results

AMD Radeon PRO W7800 32 GB, gfx1100, Windows 11, ROCm 10.2 (TheRock wheels),
VS2022 BuildTools 14.44.

Model: `unsloth/Qwen3.8-27B-UD-Q4_K_M.gguf` (15.33 GB on disk, 14.3 GiB resident).

```
INFO engine=dense-0.2 model=qwen35 layers=64 n_embd=5120
     weights_mib=14664 host_mib=0 kv_mib=208 context=8192 kv=k8v4
strata-dense: 14.32 GiB of weights, 0.20 GiB of KV cache (k8v4)
```

`host_mib=0` — the whole model fits in VRAM, **zero offload**.

| | |
| --- | --- |
| `--selftest` | **36/36 passed** |
| decode | **26.8 tok/s** (`--mtp --draft-max 2`) |
| prefill | 55.8 tok/s |
| output | correct, fluent English |

### Against llama.cpp

`llama.cpp` v0.7.0 (b11669) with its Vulkan backend, same model, same card:

| | this port | llama.cpp Vulkan |
| --- | --- | --- |
| **decode** | **26.8 tok/s** | 21.2 tok/s |
| **prefill** | 55.8 tok/s | **484 tok/s** |

Decode is ~26% faster here; prefill is ~8.7x slower. The decode win comes from
MTP speculative decoding, which llama.cpp's Vulkan path does not apply to this
family. The prefill gap is because this port's MMQ prompt path rejects the `UD`
(dynamic) quant's mixed tensor types, so the prompt is read one token at a time.

### MTP draft sweep

| draft-max | decode | accepted |
| --- | --- | --- |
| off | 19.9 tok/s | — |
| **1** | **26.7 tok/s** | 51/75 (68%) |
| **2** | **26.8 tok/s** | 65/121 (54%) |
| 3 | 24.2 tok/s | 67/174 (39%) |
| 4 | 21.5 tok/s | 70/218 (32%) |
| 5 | 19.3 tok/s | 71/265 (27%) |

Use 1 or 2.

### Why ~27 tok/s

This model is **dense** — 27B parameters active, so every weight is read for
every token:

```
14.66 GB/token ÷ 640 GB/s = 22.9 ms  →  43.6 tok/s ceiling
measured 26.8 tok/s = 61% of ceiling
```

An MoE model at the same VRAM reads only its routed experts per token; that is
why Strata's Qwen3.8-Flash-Next reaches ~90 tok/s on this card.

### Hybrid placement measured

`STRATA_DENSE_GPU_MIB=8000` forces part of the weights into host memory:

| | decode |
| --- | --- |
| all in VRAM | 26.8 tok/s |
| 8 GB budget | **2.2 tok/s** |

12x slower — every out-of-VRAM weight goes over PCIe on every token.

---

## The changes

Full detail in [HIP-PORT-NOTES.md](HIP-PORT-NOTES.md).

**1. MSVC 14.51 breaks every HIP compile.** With the VS18 toolset, `hipcc` fails
before it reaches any of this code:

```
error: __device__ function 'isgreater' cannot overload
       __host__ __device__ function 'isgreater'
  __clang_cuda_math_forward_declares.h:97
  note: previous declaration is here
  ...\MSVC\14.51.36231\include\cmath:688  _CLANG_BUILTIN2(isgreater)
```

MSVC 14.51's `<cmath>` defines `isgreater`/`isless`/... as `constexpr` functions
under `#ifdef __clang__`, colliding with ROCm's `__device__` overloads. VS2022's
14.44 toolset does not do this.

**2. `nvcuda::wmma` is not portable.** `attn_chunk_tc_kernel` used NVIDIA's WMMA
API, which has no drop-in AMD equivalent. The file already had a scalar fallback
(`attn_chunk_g_kernel`, documented as bit-identical by its author), so the
tensor-core block is wrapped in `#if !defined(STRATA_NO_WMMA)` and `use_tc` is
forced false. `--selftest` confirms the fallback is numerically correct. This
costs speed, not correctness.

**3. `std::regex` does not link with the static CRT.** Pulls
`__std_regex_transform_primary_char`, which `lld-link` cannot resolve under
`CMAKE_MSVC_RUNTIME_LIBRARY=MultiThreaded`. `shard_paths()` is now plain string
parsing.

**4. Missing symbols on upstream.** `layer_set_qwen35_gdn` / `g_qwen35_gdn` / the
`qwen35_gdn_out_norm` dispatch only exist in Xolotl; added them.

**5. MTP head discovery.** `DenseModel::load` looks the nextn block up as
`blk.<n_layer>.` in the shard list, so with `--mtp` it now scans the trunk's
directory for an `mtp-*.gguf` and appends it. Two heads in one directory is an
error, not a guess.

---

## Build

```bat
rem ROCm from TheRock wheels; VS2022 BuildTools required (14.44)
set STRATA_HIP_ARCHS=gfx1100
set STRATA_NO_VISION=1
tools\hip\build_windows.bat

rem that script only builds `strata` and `strata-device`
cmake --build build-hip-win --target strata-dense
```

Run:

```bat
build-hip-win\strata-dense.exe --serve --native <27B.gguf> ^
    --context 8192 --kv k8v4 --mtp --mtp-force --draft-max 2
```

`STRATA_MMQ_KQ=1` enables the K-quant MMQ path at run time (`kq_ok()` in
`moe_mmq.cu` reads it). It only helps for types `supported()` covers.

---

## Licence

The engine and upstream tree are MIT. This patch is MIT too.
Model weights are not distributed here.

---

## A note on the work

Four hours of porting, and the honest result is "faster at decode, much slower at
prefill than llama.cpp". If it saves someone else the four hours - especially the
MSVC 14.51 discovery, which is invisible until you try - that is the point of
publishing it.

If it happened to be useful to you:

<img src="wechat-pay.png" alt="WeChat Pay" width="220">

WeChat Pay - CALY(**Hui)
