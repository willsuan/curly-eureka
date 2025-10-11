#include "hyperdual.cuh"
#include "jacobian_kernels.cuh"

// ===== Existing HD batch kernel (value + dx,dy,dxy) =====

template <typename T>
__global__ void eval_batch_kernel(
    const T* __restrict__ x,
    const T* __restrict__ y,
    T* __restrict__ out_r,
    T* __restrict__ out_dx,
    T* __restrict__ out_dy,
    T* __restrict__ out_dxy,
    int n)
{
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i >= n) return;

  HD<T> hx = make_seed_x<T>(x[i]);
  HD<T> hy = make_seed_y<T>(y[i]);
  HD<T> h = f_xy(hx, hy);

  out_r  [i] = h.r;
  out_dx [i] = h.e1;
  out_dy [i] = h.e2;
  out_dxy[i] = h.e12;
}

extern "C" void launch_eval_batch_float(
    const float* x, const float* y,
    float* r, float* dx, float* dy, float* dxy, int n,
    cudaStream_t stream)
{
  dim3 blk(256); dim3 grd((n + blk.x - 1) / blk.x);
  eval_batch_kernel<float><<<grd, blk, 0, stream>>>(x, y, r, dx, dy, dxy, n);
}

extern "C" void launch_eval_batch_double(
    const double* x, const double* y,
    double* r, double* dx, double* dy, double* dxy, int n,
    cudaStream_t stream)
{
  dim3 blk(256); dim3 grd((n + blk.x - 1) / blk.x);
  eval_batch_kernel<double><<<grd, blk, 0, stream>>>(x, y, r, dx, dy, dxy, n);
}

// Mixed-precision HD path (primals half, derivatives float) 

__global__ void eval_batch_mixed_kernel(
    const __half* __restrict__ x,
    const __half* __restrict__ y,
    __half* __restrict__ out_r,
    float* __restrict__ out_dx,
    float* __restrict__ out_dy,
    float* __restrict__ out_dxy,
    int n)
{
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i >= n) return;
  HD_mixed hx = make_seed_x_mixed(x[i]);
  HD_mixed hy = make_seed_y_mixed(y[i]);
  HD_mixed h  = hsin(hx * hy) + hexp(hx) / hlog( HD_mixed(1.f) + hy );
  out_r  [i] = __float2half(h.r);
  out_dx [i] = h.e1;
  out_dy [i] = h.e2;
  out_dxy[i] = h.e12;
}

extern "C" void launch_eval_batch_mixed(
    const __half* x, const __half* y,
    __half* r, float* dx, float* dy, float* dxy, int n, cudaStream_t stream)
{
  dim3 blk(256); dim3 grd((n + blk.x - 1) / blk.x);
  eval_batch_mixed_kernel<<<grd, blk, 0, stream>>>(x, y, r, dx, dy, dxy, n);
}

