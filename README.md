## cuHyperDual

Library for automatic differentiation on GPUs using hyper-dual and multi-dual numbers. Work in progress.

## Features

- Hyper-dual arithmetic for exact first and mixed second derivatives.
- Multi-dual arithmetic (arbitrary K directions) for full Jacobians.
- Batched GPU kernels (millions of samples per second).
- Generic K×M Jacobian runner for any number of inputs/outputs.
- Mixed-precision mode: primals in `__half`, derivatives in `float`.
- Python binding (via PyBind11) – call it from Jupyter in a few lines.
- 
## Sample Benchmarks
