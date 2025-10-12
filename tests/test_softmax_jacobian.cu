// tests/test_softmax_jacobian.cu
#include <cstdio>
#include <array>
#include <vector>
#include <cmath>
#include "cudual/kernels.cuh"
#include "cudual/cudual.cuh"
using namespace cudadual;

// Verify Jacobian of softmax matches analytical J = diag(s) - s s^T for N=4
template <int N>
struct FSoftmax {
  template <class Num>
  CDUAL_HD std::array<Num,N> operator()(const std::array<Num,N>& x) const {
    std::array<Num,N> y;
    softmax(&x[0], N, &y[0]);
    return y;
  }
};

int main(){
  using T=double; constexpr int N=4; const int batch=1;
  // create one sample
  std::vector<T> X(batch*N); for (int i=0;i<N;++i) X[i] = 0.1*(i+1) - 0.2;
  T *dX=nullptr,*dy=nullptr,*dJ=nullptr; cudaMalloc(&dX,sizeof(T)*batch*N); cudaMalloc(&dy,sizeof(T)*batch*N); cudaMalloc(&dJ,sizeof(T)*batch*N*N);
  cudaMemcpy(dX,X.data(),sizeof(T)*batch*N,cudaMemcpyHostToDevice);

  FSoftmax<N> f;
  launch_jacobian<T,N,N>(f, dX, batch, dy, dJ);
  cudaDeviceSynchronize();
  std::vector<T> y(N), J(N*N);
  cudaMemcpy(y.data(), dy, sizeof(T)*N, cudaMemcpyDeviceToHost);
  cudaMemcpy(J.data(), dJ, sizeof(T)*N*N, cudaMemcpyDeviceToHost);

  // Analytical Jacobian
  std::vector<T> Jref(N*N,0.0);
  for (int i=0;i<N;++i){
    for (int j=0;j<N;++j){
      Jref[i*N + j] = (i==j ? y[i]*(1.0 - y[j]) : -y[i]*y[j]);
    }
  }

  double err=0; for (int i=0;i<N*N;++i) err = std::max(err, std::fabs(J[i]-Jref[i]));
  std::printf("softmax jacobian: max diff = %.3e\n", err);
  bool ok = (err < 1e-12);
  std::printf("RESULT: %s\n", ok ? "OK" : "FAIL");

  cudaFree(dX); cudaFree(dy); cudaFree(dJ);
  return ok?0:1;
}
