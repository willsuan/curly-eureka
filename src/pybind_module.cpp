#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cuda_runtime.h>
#include "runner.cuh"
#include "jacobian_kernels.cuh"

namespace py = pybind11;

py::tuple jacobian_2x2_double(py::array_t<double, py::array::c_style|py::array::forcecast> x,
                              py::array_t<double, py::array::c_style|py::array::forcecast> y) {
  if (x.ndim()!=1 || y.ndim()!=1 || x.shape(0)!=y.shape(0))
    throw std::runtime_error("x and y must be 1D arrays of equal length");
  int N = static_cast<int>(x.shape(0));

  const double* hx = x.data();
  const double* hy = y.data();

  double *dx_x,*dx_y,*f0,*f1,*J00,*J01,*J10,*J11;
  cudaMalloc(&dx_x, N*sizeof(double));
  cudaMalloc(&dx_y, N*sizeof(double));
  cudaMalloc(&f0,   N*sizeof(double));
  cudaMalloc(&f1,   N*sizeof(double));
  cudaMalloc(&J00,  N*sizeof(double));
  cudaMalloc(&J01,  N*sizeof(double));
  cudaMalloc(&J10,  N*sizeof(double));
  cudaMalloc(&J11,  N*sizeof(double));
  cudaMemcpy(dx_x, hx, N*sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(dx_y, hy, N*sizeof(double), cudaMemcpyHostToDevice);

  std::array<double*,2> inputs = { dx_x, dx_y };
  std::array<double*,2> outF   = { f0, f1 };
  std::array<double*,4> outJ   = { J00, J01, J10, J11 };
  cudaStream_t s; cudaStreamCreate(&s);
  Runner<2,2,double,Fun2From2>::run(inputs, outF, outJ, N, s);
  cudaStreamSynchronize(s);

  auto of0 = py::array_t<double>(N);
  auto of1 = py::array_t<double>(N);
  auto o00 = py::array_t<double>(N);
  auto o01 = py::array_t<double>(N);
  auto o10 = py::array_t<double>(N);
  auto o11 = py::array_t<double>(N);
  cudaMemcpy(of0.mutable_data(), f0,  N*sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpy(of1.mutable_data(), f1,  N*sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpy(o00.mutable_data(), J00, N*sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpy(o01.mutable_data(), J01, N*sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpy(o10.mutable_data(), J10, N*sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpy(o11.mutable_data(), J11, N*sizeof(double), cudaMemcpyDeviceToHost);

  cudaFree(dx_x); cudaFree(dx_y); cudaFree(f0); cudaFree(f1);
  cudaFree(J00); cudaFree(J01); cudaFree(J10); cudaFree(J11);
  cudaStreamDestroy(s);

  return py::make_tuple(of0, of1, o00, o01, o10, o11);
}

PYBIND11_MODULE(cuhd, m) {
  m.doc() = "cuHyperDual Python bindings (demo)";
  m.def("jacobian_2x2_double", &jacobian_2x2_double,
        "Compute two outputs and 2x2 Jacobian columns for f via multi-dual");
}
