# cudual — CUDA hyper‑dual & multi‑dual arithmetic

Header‑only CUDA AD types and kernels: Dual, HyperDual, MultiDual, MultiDual2.
Gradients, Hessians (hyper‑dual + one‑pass), Jacobians, vector‑output per‑output Hessians.
AoS/SoA layouts, streaming, tiled Hessians, packed storage, plans/pipelines, tests. 

This project is probably altogether useless except to help me learn CUDA.

## Build
```bash
cmake -S . -B build -DCMAKE_CUDA_ARCHITECTURES=80 \
                 -DCUDUAL_BUILD_TESTS=ON -DCUDUAL_BUILD_EXAMPLES=ON
cmake --build build -j
ctest --test-dir build -V
```

## Updates
### Math additions
- `lgamma` with correct AD via `digamma` & `trigamma` (reflection + asymptotics).
- `erfcx` (scaled complementary error) with analytic d/dx and d²/dx².
- Normal helpers: `normal_cdf`, `normal_logcdf`, `normal_logpdf(x,mu,sigma)`.
- Utilities: `sinc`, `poly_eval`, `softmax` (AD‑safe, numerically stable).

### Benchmark suite
- `examples/benchmark_suite.cu` times common kernels using CUDA events.\
  Compile‑time params: `DEF_N`, `DEF_M`, `DEF_K`, `REAL_T` (defaults 64,2,16,double).
  Runtime flags:
  ```bash
  ./build/benchmark_suite --kernel vec_plan --batch 8192 --layout AoS --chunkS 1024 --chunkP 128 --streams 1
  ./build/benchmark_suite --kernel vec_tiled --batch 4096 --layout SoA --chunkS 512 --chunkP 64
  ./build/benchmark_suite --kernel grad_hess --batch 32768
  ./build/benchmark_suite --kernel hess_hyper --batch 8192
  ./build/benchmark_suite --kernel vec_packed --batch 4096
  ```

### Notes
- The digamma/trigamma approximations use reflection for non‑positive inputs and a short asymptotic series for `x≥8`, with recurrence to shift small `x` into that region. This is accurate to ~1e‑12 for typical ML/stats ranges in double precision (a bit looser for float).

