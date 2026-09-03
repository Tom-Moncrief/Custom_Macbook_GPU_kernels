# Custom MacBook GPU kernels

Metal GPU kernels for AlphaEarth Foundation embedding workflows on Apple Silicon.

## AlphaEarth dequantisation

The released embeddings use 64 signed 8-bit channels per pixel. Values are mapped to analysis-ready values with:

```
sign(q) * (abs(q) / 127.5)^2
```

The raw value `-128` is reserved for no-data.

## Included kernels

- `aef_gpu_benchmark/Dequantize.metal`: standalone int8-to-float32 dequantisation.
- `aef_gpu_benchmark/FusedCosine.metal`: fused dequantisation, dot product, norm, and cosine similarity.
- Swift runners compile the Metal source at runtime and benchmark the Apple GPU.
- `aef_gpu_benchmark/gpu_worker.swift`: persistent worker that loads Metal once and processes multiple tile jobs.
- Python scripts extract a COG overview and provide NumPy CPU baselines.

## Reproducing the test

Install Python dependencies:

```bash
python3 -m pip install numpy rasterio
```

Download a public AlphaEarth COG tile, then extract its 1024x1024 overview:

```python
python3 aef_gpu_benchmark/extract_and_cpu.py \
  aef_2019_1S_tile.tiff \
  --overview 3 \
  --out overview_1024x1024x64.raw
```

Run standalone dequantisation:

```bash
swiftc aef_gpu_benchmark/benchmark.swift -framework Metal \
  -o aef_gpu_benchmark/benchmark
./aef_gpu_benchmark/benchmark \
  overview_1024x1024x64.raw \
  overview_1024x1024x64.gpu.f32
```

Run fused cosine similarity:

```bash
python3 aef_gpu_benchmark/cpu_fused_cosine.py
swiftc aef_gpu_benchmark/benchmark_fused.swift -framework Metal \
  -o aef_gpu_benchmark/benchmark_fused
./aef_gpu_benchmark/benchmark_fused \
  overview_1024x1024x64.raw \
  query.f32 \
  overview_1024x1024x64.gpu_cosine.f32
```

Run the persistent worker:

```bash
swiftc aef_gpu_benchmark/gpu_worker.swift -framework Metal \
  -o aef_gpu_benchmark/gpu_worker
python3 aef_gpu_benchmark/test_persistent_worker.py
```

The worker accepts tab-separated `input.raw`, `query.f32`, and `output.f32` paths, one job per line, and exits on `QUIT`.

## Initial Apple M1 results

On a 1024x1024 overview containing 67,108,864 int8 values:

| Operation | CPU | GPU | Speedup |
|---|---:|---:|---:|
| Dequantisation | 310 ms | 7.1 ms | 43.6x |
| Fused cosine similarity | 604 ms | 10.6 ms | 57.1x |
| Persistent worker job | 754 ms | 50 ms | 15.1x |

The GPU and CPU fused cosine outputs agreed within 1.2e-7 maximum absolute error.

The full 8192x8192 tile must be processed in spatial chunks: expanding all 64 channels to float32 would require approximately 16 GiB.

## Random forest inference

The forest engine compiles scikit-learn random forests into flat Metal buffers and traverses them directly over the quantised int8 AlphaEarth channels. It supports classification and regression, packed feature/threshold metadata, threadgroup feature caching, pixel-tree parallelism, and batched command submission.

Build and run the representative test forest:

```bash
python3 aef_gpu_benchmark/build_rf_model.py
python3 aef_gpu_benchmark/benchmark_cpu_forest.py
swiftc aef_gpu_benchmark/benchmark_forest_parallel.swift -framework Metal \\
  -o aef_gpu_benchmark/benchmark_forest_parallel
./aef_gpu_benchmark/benchmark_forest_parallel \\
  aef_gpu_benchmark/rf_regression \\
  aef_gpu_benchmark/overview_1024x1024x64.raw \\
  rf_regression.gpu.f32 4 packed cached
```

On the Apple M1 test machine, a 256-tree, 702,512-node regression forest scored 1,048,576 pixels in about 270 ms, versus 2.11 s with scikit-learn. The GPU result had mean absolute error about 1.5e-7 relative to the CPU reference.
