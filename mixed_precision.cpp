// this mixed-precision HD path (primals half, derivatives float) is incomplete
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
  // same f(x,y) as before: sin(x*y) + exp(x)/log(1+y)
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
