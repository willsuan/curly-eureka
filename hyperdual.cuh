#pragma once
#include <cuda_runtime.h>
#include <math.h>
#include <type_traits>

// Hyper-dual (2-dir, 2nd order mixed) type

template <typename T>
struct HD {
  T r, e1, e2, e12;
  __host__ __device__ HD() = default;
  __host__ __device__ HD(T r_, T e1_=T(0), T e2_=T(0), T e12_=T(0))
      : r(r_), e1(e1_), e2(e2_), e12(e12_) {}
};

template <typename T>
__host__ __device__ inline HD<T> make_seed_x(T x){ return HD<T>(x, T(1), T(0), T(0)); }

template <typename T>
__host__ __device__ inline HD<T> make_seed_y(T y){ return HD<T>(y, T(0), T(1), T(0)); }

// Basic arithmetic (HD)

template <typename T>
__host__ __device__ inline HD<T> operator+(const HD<T>& a, const HD<T>& b){
  return HD<T>(a.r+b.r, a.e1+b.e1, a.e2+b.e2, a.e12+b.e12);
}

template <typename T>
__host__ __device__ inline HD<T> operator-(const HD<T>& a, const HD<T>& b){
  return HD<T>(a.r-b.r, a.e1-b.e1, a.e2-b.e2, a.e12-b.e12);
}

template <typename T>
__host__ __device__ inline HD<T> operator*(const HD<T>& u, const HD<T>& v){
  return HD<T>(
    u.r * v.r,
    u.e1 * v.r + u.r * v.e1,
    u.e2 * v.r + u.r * v.e2,
    u.e12 * v.r + u.r * v.e12 + u.e1 * v.e2 + u.e2 * v.e1
  );
}

template <typename T>
__host__ __device__ inline HD<T> inv(const HD<T>& v){
  const T vr = v.r;
  const T vr2 = vr * vr;
  const T vr3 = vr2 * vr;
  return HD<T>(
    T(1)/vr,
    -v.e1/vr2,
    -v.e2/vr2,
    (T(2) * v.e1 * v.e2) / vr3 - v.e12 / vr2
  );
}

template <typename T>
__host__ __device__ inline HD<T> operator/(const HD<T>& u, const HD<T>& v){
  return u * inv(v);
}

// Unary lifts (HD)

template <typename T>
__host__ __device__ inline HD<T> hsin(const HD<T>& x){
  const T r  = sin(x.r);
  const T fr = cos(x.r);
  const T f2 = -sin(x.r);
  return HD<T>(r, fr * x.e1, fr * x.e2, f2 * (x.e1 * x.e2) + fr * x.e12);
}

template <typename T>
__host__ __device__ inline HD<T> hcos(const HD<T>& x){
  const T r  = cos(x.r);
  const T fr = -sin(x.r);
  const T f2 = -cos(x.r);
  return HD<T>(r, fr * x.e1, fr * x.e2, f2 * (x.e1 * x.e2) + fr * x.e12);
}

template <typename T>
__host__ __device__ inline HD<T> hexp(const HD<T>& x){
  const T r  = exp(x.r);
  const T fr = r;
  const T f2 = r;
  return HD<T>(r, fr * x.e1, fr * x.e2, f2 * (x.e1 * x.e2) + fr * x.e12);
}

template <typename T>
__host__ __device__ inline HD<T> hlog(const HD<T>& x){
  const T r  = log(x.r);
  const T fr = T(1)/x.r;
  const T f2 = -T(1)/(x.r * x.r);
  return HD<T>(r, fr * x.e1, fr * x.e2, f2 * (x.e1 * x.e2) + fr * x.e12);
}

template <typename T>
__host__ __device__ inline HD<T> hpow(const HD<T>& x, T a){
  const T r  = pow(x.r, a);
  const T fr = a * pow(x.r, a - T(1));
  const T f2 = a * (a - T(1)) * pow(x.r, a - T(2));
  return HD<T>(r, fr * x.e1, fr * x.e2, f2 * (x.e1 * x.e2) + fr * x.e12);
}

// Extra unary lifts (HD): tanh, erf, log1p, expm1

template <typename T>
__host__ __device__ inline HD<T> htanh(const HD<T>& x){
  const T t  = tanh(x.r);
  const T fr = T(1) - t*t;             // sech^2
  const T f2 = -T(2) * t * fr;         // d/dx(sech^2) = -2 tanh * sech^2
  return HD<T>(t, fr*x.e1, fr*x.e2, f2*(x.e1*x.e2) + fr*x.e12);
}

template <typename T>
__host__ __device__ inline HD<T> herf(const HD<T>& x){
  const T r  = erf(x.r);
  const T fr = T(2.0/1.7724538509055160273) * exp(-x.r*x.r); // 2/sqrt(pi)
  const T f2 = -T(4.0/1.7724538509055160273) * x.r * exp(-x.r*x.r);
  return HD<T>(r, fr*x.e1, fr*x.e2, f2*(x.e1*x.e2) + fr*x.e12);
}

template <typename T>
__host__ __device__ inline HD<T> hlog1p(const HD<T>& x){
  const T r  = log1p(x.r);
  const T fr = T(1)/(T(1)+x.r);
  const T f2 = -fr*fr;
  return HD<T>(r, fr*x.e1, fr*x.e2, f2*(x.e1*x.e2) + fr*x.e12);
}

template <typename T>
__host__ __device__ inline HD<T> hexpm1(const HD<T>& x){
  const T r  = expm1(x.r);
  const T fr = r + T(1);
  const T f2 = fr;
  return HD<T>(r, fr*x.e1, fr*x.e2, f2*(x.e1*x.e2) + fr*x.e12);
}

// Example target function f(x,y) using HD

template <typename T>
__host__ __device__ inline HD<T> f_xy(const HD<T>& x, const HD<T>& y){
  HD<T> term1 = hsin(x * y);
  HD<T> denom = hlog( HD<T>(T(1)) + y );
  HD<T> term2 = hexp(x) / denom;
  return term1 + term2;
}

// Multi-dual (K-dir, 1st order) type for full Jacobian columns

template <int K, typename T>
struct MD {
  T r;            // primal
  T e[K];         // first-order components along K seed directions
  __host__ __device__ MD() = default;
  __host__ __device__ explicit MD(T r_): r(r_) { for(int i=0;i<K;++i) e[i]=T(0); }
  __host__ __device__ MD(T r_, const T* v){ r=r_; for(int i=0;i<K;++i) e[i]=v[i]; }
};

template <int K, typename T>
__host__ __device__ inline MD<K,T> make_seed(T x, int j){
  MD<K,T> a; a.r = x; for(int i=0;i<K;++i) a.e[i]=T(0); if(j>=0 && j<K) a.e[j]=T(1); return a;
}

// MD ops

template <int K, typename T>
__host__ __device__ inline MD<K,T> operator+(const MD<K,T>& a, const MD<K,T>& b){
  MD<K,T> c; c.r = a.r + b.r; for(int i=0;i<K;++i) c.e[i]=a.e[i]+b.e[i]; return c;
}

template <int K, typename T>
__host__ __device__ inline MD<K,T> operator-(const MD<K,T>& a, const MD<K,T>& b){
  MD<K,T> c; c.r = a.r - b.r; for(int i=0;i<K;++i) c.e[i]=a.e[i]-b.e[i]; return c;
}

template <int K, typename T>
__host__ __device__ inline MD<K,T> operator*(const MD<K,T>& u, const MD<K,T>& v){
  MD<K,T> c; c.r = u.r * v.r; for(int i=0;i<K;++i) c.e[i] = u.e[i]*v.r + u.r*v.e[i]; return c;
}

template <int K, typename T>
__host__ __device__ inline MD<K,T> inv(const MD<K,T>& v){
  MD<K,T> w; const T vr=v.r; const T vr2=vr*vr; w.r = T(1)/vr; for(int i=0;i<K;++i) w.e[i] = -v.e[i]/vr2; return w;
}

template <int K, typename T>
__host__ __device__ inline MD<K,T> operator/(const MD<K,T>& u, const MD<K,T>& v){ return u*inv(v); }

// Unary lifts (MD)

template <int K, typename T>
__host__ __device__ inline MD<K,T> md_sin(const MD<K,T>& x){ MD<K,T> y; y.r=sin(x.r); const T fr=cos(x.r); for(int i=0;i<K;++i) y.e[i]=fr*x.e[i]; return y; }

template <int K, typename T>
__host__ __device__ inline MD<K,T> md_exp(const MD<K,T>& x){ MD<K,T> y; y.r=exp(x.r); for(int i=0;i<K;++i) y.e[i]=y.r*x.e[i]; return y; }

template <int K, typename T>
__host__ __device__ inline MD<K,T> md_log(const MD<K,T>& x){ MD<K,T> y; y.r=log(x.r); const T fr=T(1)/x.r; for(int i=0;i<K;++i) y.e[i]=fr*x.e[i]; return y; }

template <int K, typename T>
__host__ __device__ inline MD<K,T> md_tanh(const MD<K,T>& x){ MD<K,T> y; const T t=tanh(x.r); const T fr=T(1)-t*t; y.r=t; for(int i=0;i<K;++i) y.e[i]=fr*x.e[i]; return y; }

template <int K, typename T>
__host__ __device__ inline MD<K,T> md_log1p(const MD<K,T>& x){ MD<K,T> y; y.r=log1p(x.r); const T fr=T(1)/(T(1)+x.r); for(int i=0;i<K;++i) y.e[i]=fr*x.e[i]; return y; }

template <int K, typename T>
__host__ __device__ inline MD<K,T> md_expm1(const MD<K,T>& x){ MD<K,T> y; y.r=expm1(x.r); const T fr=y.r+T(1); for(int i=0;i<K;++i) y.e[i]=fr*x.e[i]; return y; }

// Binary lifts (MD)

template <int K, typename T>
__host__ __device__ inline MD<K,T> md_hypot(const MD<K,T>& x, const MD<K,T>& y){
  MD<K,T> z; z.r = hypot(x.r, y.r); if(z.r==T(0)){ for(int i=0;i<K;++i) z.e[i]=T(0); return z; }
  const T inv = T(1)/z.r; // ∂r/∂x = x/r, ∂r/∂y = y/r
  #pragma unroll
  for(int i=0;i<K;++i) z.e[i] = inv*( x.r*x.e[i] + y.r*y.e[i] );
  return z;
}

template <int K, typename T>
__host__ __device__ inline MD<K,T> md_atan2(const MD<K,T>& y, const MD<K,T>& x){
  MD<K,T> z; z.r = atan2(y.r, x.r); const T den = x.r*x.r + y.r*y.r; const T inv = T(1)/den;
  #pragma unroll
  for(int i=0;i<K;++i) z.e[i] = (-y.r*inv)*x.e[i] + (x.r*inv)*y.e[i];
  return z;
}

// Example f(x,y) using MD (K>=2)

template <int K, typename T>
__host__ __device__ inline MD<K,T> f_xy_md(const MD<K,T>& x, const MD<K,T>& y){
  MD<K,T> term1 = md_sin<K,T>(x * y);
  MD<K,T> denom = md_log<K,T>( MD<K,T>(T(1)) + y );
  MD<K,T> term2 = md_exp<K,T>(x) / denom;
  return term1 + term2;
}

