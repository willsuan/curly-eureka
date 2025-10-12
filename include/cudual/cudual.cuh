// include/cudual/cudual.cuh
#pragma once
#include <cuda_runtime.h>
#include <math_constants.h>
#include <math_functions.h>
#include <cuda_fp16.h>
#include <cmath>
#include <type_traits>

#ifndef CDUAL_HD
#define CDUAL_HD __host__ __device__ __forceinline__
#endif

namespace cudadual {

// Bring device-friendly overloads into scope
using ::sin; using ::cos; using ::tan; using ::exp; using ::log; using ::sqrt;
using ::sinh; using ::cosh; using ::tanh; using ::asin; using ::acos; using ::atan;
using ::erf; using ::erfc; using ::expm1; using ::log1p; using ::exp2; using ::log2; using ::log10;

// ---------- Relational operators (compare by value .f) ----------
template <typename T> CDUAL_HD inline bool operator>(const struct Dual<T>& a, const struct Dual<T>& b){ return a.f > b.f; }
template <typename T> CDUAL_HD inline bool operator<(const struct Dual<T>& a, const struct Dual<T>& b){ return a.f < b.f; }
template <typename T> CDUAL_HD inline bool operator>=(const struct Dual<T>& a, const struct Dual<T>& b){ return a.f >= b.f; }
template <typename T> CDUAL_HD inline bool operator<=(const struct Dual<T>& a, const struct Dual<T>& b){ return a.f <= b.f; }
template <typename T> CDUAL_HD inline bool operator>(const struct HyperDual<T>& a, const struct HyperDual<T>& b){ return a.f > b.f; }
template <typename T> CDUAL_HD inline bool operator<(const struct HyperDual<T>& a, const struct HyperDual<T>& b){ return a.f < b.f; }
template <typename T> CDUAL_HD inline bool operator>=(const struct HyperDual<T>& a, const struct HyperDual<T>& b){ return a.f >= b.f; }
template <typename T> CDUAL_HD inline bool operator<=(const struct HyperDual<T>& a, const struct HyperDual<T>& b){ return a.f <= b.f; }
template <typename T, int N> CDUAL_HD inline bool operator>(const struct MultiDual<T,N>& a, const struct MultiDual<T,N>& b){ return a.f > b.f; }
template <typename T, int N> CDUAL_HD inline bool operator<(const struct MultiDual<T,N>& a, const struct MultiDual<T,N>& b){ return a.f < b.f; }
template <typename T, int N> CDUAL_HD inline bool operator>=(const struct MultiDual<T,N>& a, const struct MultiDual<T,N>& b){ return a.f >= b.f; }
template <typename T, int N> CDUAL_HD inline bool operator<=(const struct MultiDual<T,N>& a, const struct MultiDual<T,N>& b){ return a.f <= b.f; }
template <typename T, int N> CDUAL_HD inline bool operator>(const struct MultiDual2<T,N>& a, const struct MultiDual2<T,N>& b){ return a.f > b.f; }
template <typename T, int N> CDUAL_HD inline bool operator<(const struct MultiDual2<T,N>& a, const struct MultiDual2<T,N>& b){ return a.f < b.f; }
template <typename T, int N> CDUAL_HD inline bool operator>=(const struct MultiDual2<T,N>& a, const struct MultiDual2<T,N>& b){ return a.f >= b.f; }
template <typename T, int N> CDUAL_HD inline bool operator<=(const struct MultiDual2<T,N>& a, const struct MultiDual2<T,N>& b){ return a.f <= b.f; }
// max/min for AD numbers (choose by .f)
template <class Num, typename Enable = std::enable_if_t<!std::is_arithmetic<Num>::value>>
CDUAL_HD Num max(const Num& a, const Num& b){ return (a > b) ? a : b; }
template <class Num, typename Enable = std::enable_if_t<!std::is_arithmetic<Num>::value>>
CDUAL_HD Num min(const Num& a, const Num& b){ return (a < b) ? a : b; }

// ============================================================================
// Dual<T>
// ============================================================================
template <typename T>
struct Dual { T f, d; CDUAL_HD Dual(T fv=T(0), T dv=T(0)) : f(fv), d(dv) {} };
template <typename T> CDUAL_HD Dual<T> operator+(const Dual<T>& x, const Dual<T>& y){ return Dual<T>(x.f+y.f, x.d+y.d); }
template <typename T> CDUAL_HD Dual<T> operator-(const Dual<T>& x, const Dual<T>& y){ return Dual<T>(x.f-y.f, x.d-y.d); }
template <typename T> CDUAL_HD Dual<T> operator*(const Dual<T>& x, const Dual<T>& y){ return Dual<T>(x.f*y.f, x.f*y.d + x.d*y.f); }
template <typename T> CDUAL_HD Dual<T> inv(const Dual<T>& y){ T a=y.f, b=y.d; return Dual<T>(T(1)/a, -b/(a*a)); }
template <typename T> CDUAL_HD Dual<T> operator/(const Dual<T>& x, const Dual<T>& y){ return x*inv(y); }
template <typename T> CDUAL_HD Dual<T> operator+(const Dual<T>& x, T s){ return Dual<T>(x.f+s, x.d); }
template <typename T> CDUAL_HD Dual<T> operator+(T s, const Dual<T>& x){ return x+s; }
template <typename T> CDUAL_HD Dual<T> operator-(const Dual<T>& x, T s){ return Dual<T>(x.f-s, x.d); }
template <typename T> CDUAL_HD Dual<T> operator-(T s, const Dual<T>& x){ return Dual<T>(s-x.f, -x.d); }
template <typename T> CDUAL_HD Dual<T> operator*(const Dual<T>& x, T s){ return Dual<T>(x.f*s, x.d*s); }
template <typename T> CDUAL_HD Dual<T> operator*(T s, const Dual<T>& x){ return x*s; }
template <typename T> CDUAL_HD Dual<T> operator/(const Dual<T>& x, T s){ return Dual<T>(x.f/s, x.d/s); }
template <typename T> CDUAL_HD Dual<T> operator/(T s, const Dual<T>& x){ return Dual<T>(s,0)/x; }
template <typename T> CDUAL_HD Dual<T> sin(const Dual<T>& x){ T a=x.f; return Dual<T>(::sin(a), ::cos(a)*x.d); }
template <typename T> CDUAL_HD Dual<T> cos(const Dual<T>& x){ T a=x.f; return Dual<T>(::cos(a), -::sin(a)*x.d); }
template <typename T> CDUAL_HD Dual<T> tan(const Dual<T>& x){ T a=x.f; T t=::tan(a); return Dual<T>(t, (T(1)+t*t)*x.d); }
template <typename T> CDUAL_HD Dual<T> exp(const Dual<T>& x){ T a=x.f; T fa=::exp(a); return Dual<T>(fa, fa*x.d); }
template <typename T> CDUAL_HD Dual<T> log(const Dual<T>& x){ T a=x.f; return Dual<T>(::log(a), x.d/a); }
template <typename T> CDUAL_HD Dual<T> sqrt(const Dual<T>& x){ T a=x.f; T r=::sqrt(a); return Dual<T>(r, x.d/(T(2)*r)); }
template <typename T> CDUAL_HD Dual<T> pow(const Dual<T>& x, T c){ T a=x.f; T fa=::pow(a,c); return Dual<T>(fa, (c*::pow(a,c-T(1)))*x.d); }
template <typename T> CDUAL_HD Dual<T> sinh(const Dual<T>& x){ T a=x.f; return Dual<T>(::sinh(a), ::cosh(a)*x.d); }
template <typename T> CDUAL_HD Dual<T> cosh(const Dual<T>& x){ T a=x.f; return Dual<T>(::cosh(a), ::sinh(a)*x.d); }
template <typename T> CDUAL_HD Dual<T> tanh(const Dual<T>& x){ T a=x.f; T th=::tanh(a); return Dual<T>(th, (T(1)-th*th)*x.d); }
template <typename T> CDUAL_HD Dual<T> asin(const Dual<T>& x){ T a=x.f; T s=::sqrt(T(1)-a*a); return Dual<T>(::asin(a), x.d/s); }
template <typename T> CDUAL_HD Dual<T> acos(const Dual<T>& x){ T a=x.f; T s=::sqrt(T(1)-a*a); return Dual<T>(::acos(a), -x.d/s); }
template <typename T> CDUAL_HD Dual<T> atan(const Dual<T>& x){ T a=x.f; return Dual<T>(::atan(a), x.d/(T(1)+a*a)); }
template <typename T> CDUAL_HD Dual<T> expm1(const Dual<T>& x){ T a=x.f; T fa=::expm1(a); return Dual<T>(fa, ::exp(a)*x.d); }
template <typename T> CDUAL_HD Dual<T> log1p(const Dual<T>& x){ T a=x.f; return Dual<T>(::log1p(a), x.d/(T(1)+a)); }
template <typename T> CDUAL_HD Dual<T> erf(const Dual<T>& x){ T a=x.f; T d1=T(2)/CUDART_SQRT_PI*::exp(-a*a); return Dual<T>(::erf(a), d1*x.d); }
template <typename T> CDUAL_HD Dual<T> erfc(const Dual<T>& x){ T a=x.f; T d1=-T(2)/CUDART_SQRT_PI*::exp(-a*a); return Dual<T>(::erfc(a), d1*x.d); }
template <typename T> CDUAL_HD Dual<T> exp2(const Dual<T>& x){ T a=x.f; T fa=::exp2(a); const T ln2=T(0.6931471805599453094); return Dual<T>(fa, ln2*fa*x.d); }
template <typename T> CDUAL_HD Dual<T> log2(const Dual<T>& x){ T a=x.f; const T invln2=T(1.4426950408889634074); return Dual<T>(::log2(a), invln2*x.d/a); }
template <typename T> CDUAL_HD Dual<T> log10(const Dual<T>& x){ T a=x.f; const T invln10=T(0.43429448190325182765); return Dual<T>(::log10(a), invln10*x.d/a); }

// ============================================================================
// HyperDual<T>
// ============================================================================
template <typename T>
struct HyperDual { T f, e1, e2, e12; CDUAL_HD HyperDual(T fv=T(0), T e1v=T(0), T e2v=T(0), T e12v=T(0)) : f(fv), e1(e1v), e2(e2v), e12(e12v) {} };
template <typename T> CDUAL_HD HyperDual<T> operator+(const HyperDual<T>& x, const HyperDual<T>& y){ return HyperDual<T>(x.f+y.f, x.e1+y.e1, x.e2+y.e2, x.e12+y.e12); }
template <typename T> CDUAL_HD HyperDual<T> operator-(const HyperDual<T>& x, const HyperDual<T>& y){ return HyperDual<T>(x.f-y.f, x.e1-y.e1, x.e2-y.e2, x.e12-y.e12); }
template <typename T> CDUAL_HD HyperDual<T> operator*(const HyperDual<T>& x, const HyperDual<T>& y){
  T f  = x.f*y.f; T e1 = x.f*y.e1 + x.e1*y.f; T e2 = x.f*y.e2 + x.e2*y.f;
  T e12= x.f*y.e12 + x.e1*y.e2 + x.e2*y.e1 + x.e12*y.f; return HyperDual<T>(f,e1,e2,e12);
}
template <typename T> CDUAL_HD HyperDual<T> inv(const HyperDual<T>& y){ T A=y.f, B=y.e1, C=y.e2, D=y.e12; T invA=T(1)/A; return HyperDual<T>(invA, -B/(A*A), -C/(A*A), (T(2)*B*C)/(A*A*A) - D/(A*A)); }
template <typename T> CDUAL_HD HyperDual<T> operator/(const HyperDual<T>& x, const HyperDual<T>& y){ return x*inv(y); }
template <typename T> CDUAL_HD HyperDual<T> operator+(const HyperDual<T>& x, T s){ return HyperDual<T>(x.f+s,x.e1,x.e2,x.e12); }
template <typename T> CDUAL_HD HyperDual<T> operator+(T s, const HyperDual<T>& x){ return x+s; }
template <typename T> CDUAL_HD HyperDual<T> operator-(const HyperDual<T>& x, T s){ return HyperDual<T>(x.f-s,x.e1,x.e2,x.e12); }
template <typename T> CDUAL_HD HyperDual<T> operator-(T s, const HyperDual<T>& x){ return HyperDual<T>(s-x.f,-x.e1,-x.e2,-x.e12); }
template <typename T> CDUAL_HD HyperDual<T> operator*(const HyperDual<T>& x, T s){ return HyperDual<T>(x.f*s,x.e1*s,x.e2*s,x.e12*s); }
template <typename T> CDUAL_HD HyperDual<T> operator/(const HyperDual<T>& x, T s){ return HyperDual<T>(x.f/s,x.e1/s,x.e2/s,x.e12/s); }
template <typename T> CDUAL_HD HyperDual<T> sin(const HyperDual<T>& x){ T a=x.f; T fa=::sin(a); T d1=::cos(a); T d2=-fa; return HyperDual<T>(fa, d1*x.e1, d1*x.e2, d2*x.e1*x.e2 + d1*x.e12); }
template <typename T> CDUAL_HD HyperDual<T> cos(const HyperDual<T>& x){ T a=x.f; T fa=::cos(a); T d1=-::sin(a); T d2=-fa; return HyperDual<T>(fa, d1*x.e1, d1*x.e2, d2*x.e1*x.e2 + d1*x.e12); }
template <typename T> CDUAL_HD HyperDual<T> tan(const HyperDual<T>& x){ T a=x.f; T t=::tan(a); T d1=T(1)+t*t; T d2=T(2)*t*d1; return HyperDual<T>(t, d1*x.e1, d1*x.e2, d2*x.e1*x.e2 + d1*x.e12); }
template <typename T> CDUAL_HD HyperDual<T> exp(const HyperDual<T>& x){ T a=x.f; T fa=::exp(a); return HyperDual<T>(fa, fa*x.e1, fa*x.e2, fa*x.e1*x.e2 + fa*x.e12); }
template <typename T> CDUAL_HD HyperDual<T> log(const HyperDual<T>& x){ T a=x.f; T d1=T(1)/a, d2=-T(1)/(a*a); return HyperDual<T>(::log(a), d1*x.e1, d1*x.e2, d2*x.e1*x.e2 + d1*x.e12); }
template <typename T> CDUAL_HD HyperDual<T> sqrt(const HyperDual<T>& x){ T a=x.f; T r=::sqrt(a); T d1=T(0.5)/r; T d2=T(-0.25)/(r*r*r); return HyperDual<T>(r, d1*x.e1, d1*x.e2, d2*x.e1*x.e2 + d1*x.e12); }
template <typename T> CDUAL_HD HyperDual<T> sinh(const HyperDual<T>& x){ T a=x.f; T fa=::sinh(a); T d1=::cosh(a); T d2=fa; return HyperDual<T>(fa, d1*x.e1, d1*x.e2, d2*x.e1*x.e2 + d1*x.e12); }
template <typename T> CDUAL_HD HyperDual<T> cosh(const HyperDual<T>& x){ T a=x.f; T fa=::cosh(a); T d1=::sinh(a); T d2=fa; return HyperDual<T>(fa, d1*x.e1, d1*x.e2, d2*x.e1*x.e2 + d1*x.e12); }
template <typename T> CDUAL_HD HyperDual<T> tanh(const HyperDual<T>& x){ T a=x.f; T th=::tanh(a); T d1=T(1)-th*th; T d2=-T(2)*th*d1; return HyperDual<T>(th, d1*x.e1, d1*x.e2, d2*x.e1*x.e2 + d1*x.e12); }
template <typename T> CDUAL_HD HyperDual<T> asin(const HyperDual<T>& x){ T a=x.f; T den=::sqrt(T(1)-a*a); T d1=T(1)/den; T d2=a/(den*den*den); return HyperDual<T>(::asin(a), d1*x.e1, d1*x.e2, d2*x.e1*x.e2 + d1*x.e12); }
template <typename T> CDUAL_HD HyperDual<T> acos(const HyperDual<T>& x){ T a=x.f; T den=::sqrt(T(1)-a*a); T d1=-T(1)/den; T d2=-a/(den*den*den); return HyperDual<T>(::acos(a), d1*x.e1, d1*x.e2, d2*x.e1*x.e2 + d1*x.e12); }
template <typename T> CDUAL_HD HyperDual<T> atan(const HyperDual<T>& x){ T a=x.f; T den=T(1)+a*a; T d1=T(1)/den; T d2=-T(2)*a/(den*den); return HyperDual<T>(::atan(a), d1*x.e1, d1*x.e2, d2*x.e1*x.e2 + d1*x.e12); }
template <typename T> CDUAL_HD HyperDual<T> pow(const HyperDual<T>& x, T c){ T a=x.f; T fa=::pow(a,c); T d1=c*::pow(a, c-T(1)); T d2=c*(c-T(1))*::pow(a, c-T(2)); return HyperDual<T>(fa, d1*x.e1, d1*x.e2, d2*x.e1*x.e2 + d1*x.e12); }
template <typename T> CDUAL_HD HyperDual<T> expm1(const HyperDual<T>& x){ T a=x.f; T fa=::expm1(a); T d1=::exp(a); T d2=d1; return HyperDual<T>(fa, d1*x.e1, d1*x.e2, d2*x.e1*x.e2 + d1*x.e12); }
template <typename T> CDUAL_HD HyperDual<T> log1p(const HyperDual<T>& x){ T a=x.f; T fa=::log1p(a); T d1=T(1)/(T(1)+a); T d2=-d1*d1; return HyperDual<T>(fa, d1*x.e1, d1*x.e2, d2*x.e1*x.e2 + d1*x.e12); }
template <typename T> CDUAL_HD HyperDual<T> erf(const HyperDual<T>& x){ T a=x.f; T fa=::erf(a); T d1=T(2)/CUDART_SQRT_PI*::exp(-a*a); T d2=-T(4)*a/(CUDART_SQRT_PI)*::exp(-a*a); return HyperDual<T>(fa, d1*x.e1, d1*x.e2, d2*x.e1*x.e2 + d1*x.e12); }
template <typename T> CDUAL_HD HyperDual<T> erfc(const HyperDual<T>& x){ T a=x.f; T fa=::erfc(a); T d1=-T(2)/CUDART_SQRT_PI*::exp(-a*a); T d2=T(4)*a/(CUDART_SQRT_PI)*::exp(-a*a); return HyperDual<T>(fa, d1*x.e1, d1*x.e2, d2*x.e1*x.e2 + d1*x.e12); }
template <typename T> CDUAL_HD HyperDual<T> exp2(const HyperDual<T>& x){ T a=x.f; T fa=::exp2(a); const T ln2=T(0.6931471805599453094); const T ln22=ln2*ln2; return HyperDual<T>(fa, ln2*fa*x.e1, ln2*fa*x.e2, ln22*fa*x.e1*x.e2 + ln2*fa*x.e12); }
template <typename T> CDUAL_HD HyperDual<T> log2(const HyperDual<T>& x){ T a=x.f; const T invln2=T(1.4426950408889634074); const T d1=invln2/a; const T d2=-invln2/(a*a); return HyperDual<T>(::log2(a), d1*x.e1, d1*x.e2, d2*x.e1*x.e2 + d1*x.e12); }
template <typename T> CDUAL_HD HyperDual<T> log10(const HyperDual<T>& x){ T a=x.f; const T invln10=T(0.43429448190325182765); const T d1=invln10/a; const T d2=-invln10/(a*a); return HyperDual<T>(::log10(a), d1*x.e1, d1*x.e2, d2*x.e1*x.e2 + d1*x.e12); }

// ============================================================================
// MultiDual<T,N>  (1st order vector forward mode)
// ============================================================================
template <typename T, int N>
struct MultiDual {
  T f; T d[N];
  CDUAL_HD MultiDual(T fv=T(0)) : f(fv){ for(int i=0;i<N;++i) d[i]=T(0); }
  CDUAL_HD MultiDual(T fv, const T (&dv)[N]) : f(fv){ for(int i=0;i<N;++i) d[i]=dv[i]; }
};
template <typename T, int N> CDUAL_HD MultiDual<T,N> operator+(const MultiDual<T,N>& x, const MultiDual<T,N>& y){ MultiDual<T,N> r; r.f=x.f+y.f; for(int i=0;i<N;++i) r.d[i]=x.d[i]+y.d[i]; return r; }
template <typename T, int N> CDUAL_HD MultiDual<T,N> operator-(const MultiDual<T,N>& x, const MultiDual<T,N>& y){ MultiDual<T,N> r; r.f=x.f-y.f; for(int i=0;i<N;++i) r.d[i]=x.d[i]-y.d[i]; return r; }
template <typename T, int N> CDUAL_HD MultiDual<T,N> operator*(const MultiDual<T,N>& x, const MultiDual<T,N>& y){ MultiDual<T,N> r; r.f=x.f*y.f; for(int i=0;i<N;++i) r.d[i]=x.f*y.d[i]+y.f*x.d[i]; return r; }
template <typename T, int N> CDUAL_HD MultiDual<T,N> inv(const MultiDual<T,N>& y){ MultiDual<T,N> r; T a=y.f; r.f=T(1)/a; for(int i=0;i<N;++i) r.d[i]=-y.d[i]/(a*a); return r; }
template <typename T, int N> CDUAL_HD MultiDual<T,N> operator/(const MultiDual<T,N>& x, const MultiDual<T,N>& y){ return x*inv(y); }
template <typename T, int N> CDUAL_HD MultiDual<T,N> operator+(const MultiDual<T,N>& x, T s){ MultiDual<T,N> r=x; r.f+=s; return r; }
template <typename T, int N> CDUAL_HD MultiDual<T,N> operator-(const MultiDual<T,N>& x, T s){ MultiDual<T,N> r=x; r.f-=s; return r; }
template <typename T, int N> CDUAL_HD MultiDual<T,N> operator*(const MultiDual<T,N>& x, T s){ MultiDual<T,N> r=x; r.f*=s; for(int i=0;i<N;++i) r.d[i]*=s; return r; }
template <typename T, int N> CDUAL_HD MultiDual<T,N> operator/(const MultiDual<T,N>& x, T s){ MultiDual<T,N> r=x; r.f/=s; for(int i=0;i<N;++i) r.d[i]/=s; return r; }
template <typename T, int N> CDUAL_HD MultiDual<T,N> sin(const MultiDual<T,N>& x){ MultiDual<T,N> r; T a=x.f; r.f=::sin(a); T f1=::cos(a); for(int i=0;i<N;++i) r.d[i]=f1*x.d[i]; return r; }
template <typename T, int N> CDUAL_HD MultiDual<T,N> cos(const MultiDual<T,N>& x){ MultiDual<T,N> r; T a=x.f; r.f=::cos(a); T f1=-::sin(a); for(int i=0;i<N;++i) r.d[i]=f1*x.d[i]; return r; }
template <typename T, int N> CDUAL_HD MultiDual<T,N> tan(const MultiDual<T,N>& x){ MultiDual<T,N> r; T a=x.f; T t=::tan(a); r.f=t; T f1=T(1)+t*t; for(int i=0;i<N;++i) r.d[i]=f1*x.d[i]; return r; }
template <typename T, int N> CDUAL_HD MultiDual<T,N> exp(const MultiDual<T,N>& x){ MultiDual<T,N> r; T a=x.f; T fa=::exp(a); r.f=fa; for(int i=0;i<N;++i) r.d[i]=fa*x.d[i]; return r; }
template <typename T, int N> CDUAL_HD MultiDual<T,N> log(const MultiDual<T,N>& x){ MultiDual<T,N> r; T a=x.f; r.f=::log(a); for(int i=0;i<N;++i) r.d[i]=x.d[i]/a; return r; }
template <typename T, int N> CDUAL_HD MultiDual<T,N> sqrt(const MultiDual<T,N>& x){ MultiDual<T,N> r; T a=x.f; T ra=::sqrt(a); r.f=ra; for(int i=0;i<N;++i) r.d[i]=x.d[i]/(T(2)*ra); return r; }
template <typename T, int N> CDUAL_HD MultiDual<T,N> pow(const MultiDual<T,N>& x, T c){ MultiDual<T,N> r; T a=x.f; r.f=::pow(a,c); T f1=c*::pow(a, c-T(1)); for(int i=0;i<N;++i) r.d[i]=f1*x.d[i]; return r; }
template <typename T, int N> CDUAL_HD MultiDual<T,N> sinh(const MultiDual<T,N>& x){ MultiDual<T,N> r; T a=x.f; r.f=::sinh(a); T f1=::cosh(a); for(int i=0;i<N;++i) r.d[i]=f1*x.d[i]; return r; }
template <typename T, int N> CDUAL_HD MultiDual<T,N> cosh(const MultiDual<T,N>& x){ MultiDual<T,N> r; T a=x.f; r.f=::cosh(a); T f1=::sinh(a); for(int i=0;i<N;++i) r.d[i]=f1*x.d[i]; return r; }
template <typename T, int N> CDUAL_HD MultiDual<T,N> tanh(const MultiDual<T,N>& x){ MultiDual<T,N> r; T a=x.f; T th=::tanh(a); r.f=th; T f1=T(1)-th*th; for(int i=0;i<N;++i) r.d[i]=f1*x.d[i]; return r; }
template <typename T, int N> CDUAL_HD MultiDual<T,N> asin(const MultiDual<T,N>& x){ MultiDual<T,N> r; T a=x.f; T s=::sqrt(T(1)-a*a); r.f=::asin(a); for(int i=0;i<N;++i) r.d[i]=x.d[i]/s; return r; }
template <typename T, int N> CDUAL_HD MultiDual<T,N> acos(const MultiDual<T,N>& x){ MultiDual<T,N> r; T a=x.f; T s=::sqrt(T(1)-a*a); r.f=::acos(a); for(int i=0;i<N;++i) r.d[i]=-x.d[i]/s; return r; }
template <typename T, int N> CDUAL_HD MultiDual<T,N> atan(const MultiDual<T,N>& x){ MultiDual<T,N> r; T a=x.f; r.f=::atan(a); T f1=T(1)/(T(1)+a*a); for(int i=0;i<N;++i) r.d[i]=f1*x.d[i]; return r; }
template <typename T, int N> CDUAL_HD MultiDual<T,N> expm1(const MultiDual<T,N>& x){ MultiDual<T,N> r; T a=x.f; T fa=::expm1(a); r.f=fa; for(int i=0;i<N;++i) r.d[i]=::exp(a)*x.d[i]; return r; }
template <typename T, int N> CDUAL_HD MultiDual<T,N> log1p(const MultiDual<T,N>& x){ MultiDual<T,N> r; T a=x.f; r.f=::log1p(a); for(int i=0;i<N;++i) r.d[i]=x.d[i]/(T(1)+a); return r; }
template <typename T, int N> CDUAL_HD MultiDual<T,N> erf(const MultiDual<T,N>& x){ MultiDual<T,N> r; T a=x.f; r.f=::erf(a); T f1=T(2)/CUDART_SQRT_PI*::exp(-a*a); for(int i=0;i<N;++i) r.d[i]=f1*x.d[i]; return r; }
template <typename T, int N> CDUAL_HD MultiDual<T,N> erfc(const MultiDual<T,N>& x){ MultiDual<T,N> r; T a=x.f; r.f=::erfc(a); T f1=-T(2)/CUDART_SQRT_PI*::exp(-a*a); for(int i=0;i<N;++i) r.d[i]=f1*x.d[i]; return r; }
template <typename T, int N> CDUAL_HD MultiDual<T,N> exp2(const MultiDual<T,N>& x){ MultiDual<T,N> r; T a=x.f; T fa=::exp2(a); const T ln2=T(0.6931471805599453094); r.f=fa; for(int i=0;i<N;++i) r.d[i]=ln2*fa*x.d[i]; return r; }
template <typename T, int N> CDUAL_HD MultiDual<T,N> log2(const MultiDual<T,N>& x){ MultiDual<T,N> r; T a=x.f; const T invln2=T(1.4426950408889634074); r.f=::log2(a); for(int i=0;i<N;++i) r.d[i]=invln2*x.d[i]/a; return r; }
template <typename T, int N> CDUAL_HD MultiDual<T,N> log10(const MultiDual<T,N>& x){ MultiDual<T,N> r; T a=x.f; const T invln10=T(0.43429448190325182765); r.f=::log10(a); for(int i=0;i<N;++i) r.d[i]=invln10*x.d[i]/a; return r; }

// ============================================================================
// MultiDual2<T,N> (2nd order forward mode: value + gradient + full Hessian)
// ============================================================================
template <typename T, int N>
struct MultiDual2 {
  T f; T g[N]; T H[N*N];
  CDUAL_HD MultiDual2(T fv=T(0)) : f(fv){ for(int i=0;i<N;++i) g[i]=T(0); for(int i=0;i<N*N;++i) H[i]=T(0); }
};
template <typename T, int N> CDUAL_HD MultiDual2<T,N> operator+(const MultiDual2<T,N>& x, const MultiDual2<T,N>& y){
  MultiDual2<T,N> r; r.f=x.f+y.f; for(int i=0;i<N;++i) r.g[i]=x.g[i]+y.g[i]; for(int i=0;i<N*N;++i) r.H[i]=x.H[i]+y.H[i]; return r;
}
template <typename T, int N> CDUAL_HD MultiDual2<T,N> operator-(const MultiDual2<T,N>& x, const MultiDual2<T,N>& y){
  MultiDual2<T,N> r; r.f=x.f-y.f; for(int i=0;i<N;++i) r.g[i]=x.g[i]-y.g[i]; for(int i=0;i<N*N;++i) r.H[i]=x.H[i]-y.H[i]; return r;
}
template <typename T, int N> CDUAL_HD MultiDual2<T,N> operator*(const MultiDual2<T,N>& x, const MultiDual2<T,N>& y){
  MultiDual2<T,N> r; r.f=x.f*y.f; for(int i=0;i<N;++i) r.g[i]=x.g[i]*y.f + y.g[i]*x.f;
  for(int i=0;i<N;++i) for(int j=0;j<N;++j) r.H[i*N+j] = x.H[i*N+j]*y.f + y.H[i*N+j]*x.f + x.g[i]*y.g[j] + y.g[i]*x.g[j];
  return r;
}
template <typename T, int N> CDUAL_HD MultiDual2<T,N> inv(const MultiDual2<T,N>& y){
  MultiDual2<T,N> r; T a=y.f; T a2=a*a, a3=a2*a; r.f=T(1)/a;
  for(int i=0;i<N;++i) r.g[i] = -y.g[i]/a2;
  for(int i=0;i<N;++i) for(int j=0;j<N;++j) r.H[i*N+j] = (T(2)*y.g[i]*y.g[j])/a3 - y.H[i*N+j]/a2;
  return r;
}
template <typename T, int N> CDUAL_HD MultiDual2<T,N> operator/(const MultiDual2<T,N>& x, const MultiDual2<T,N>& y){ return x*inv(y); }
template <typename T, int N> CDUAL_HD MultiDual2<T,N> operator+(const MultiDual2<T,N>& x, T s){ MultiDual2<T,N> r=x; r.f+=s; return r; }
template <typename T, int N> CDUAL_HD MultiDual2<T,N> operator-(const MultiDual2<T,N>& x, T s){ MultiDual2<T,N> r=x; r.f-=s; return r; }
template <typename T, int N> CDUAL_HD MultiDual2<T,N> operator*(const MultiDual2<T,N>& x, T s){ MultiDual2<T,N> r=x; r.f*=s; for(int i=0;i<N;++i) r.g[i]*=s; for(int i=0;i<N*N;++i) r.H[i]*=s; return r; }
template <typename T, int N> CDUAL_HD MultiDual2<T,N> operator/(const MultiDual2<T,N>& x, T s){ MultiDual2<T,N> r=x; r.f/=s; for(int i=0;i<N;++i) r.g[i]/=s; for(int i=0;i<N*N;++i) r.H[i]/=s; return r; }

template <typename T, int N, typename DF, typename D2F>
CDUAL_HD MultiDual2<T,N> _lift_unary_2(const MultiDual2<T,N>& x, T fa, DF df, D2F d2f){
  MultiDual2<T,N> r; r.f=fa; T d1=df(x.f), d2=d2f(x.f);
  for(int i=0;i<N;++i) r.g[i]=d1*x.g[i];
  for(int i=0;i<N;++i) for(int j=0;j<N;++j) r.H[i*N+j] = d2*x.g[i]*x.g[j] + d1*x.H[i*N+j];
  return r;
}
template <typename T, int N> CDUAL_HD MultiDual2<T,N> sin(const MultiDual2<T,N>& x){ T a=x.f; T fa=::sin(a); auto df=[] CDUAL_HD (T v){return ::cos(v);}; auto d2=[] CDUAL_HD (T v){return -::sin(v);}; return _lift_unary_2<T,N>(x,fa,df,d2); }
template <typename T, int N> CDUAL_HD MultiDual2<T,N> cos(const MultiDual2<T,N>& x){ T a=x.f; T fa=::cos(a); auto df=[] CDUAL_HD (T v){return -::sin(v);}; auto d2=[] CDUAL_HD (T v){return -::cos(v);}; return _lift_unary_2<T,N>(x,fa,df,d2); }
template <typename T, int N> CDUAL_HD MultiDual2<T,N> tan(const MultiDual2<T,N>& x){ T a=x.f; T t=::tan(a); T fa=t; auto df=[] CDUAL_HD (T v){ T t=::tan(v); return T(1)+t*t; }; auto d2=[] CDUAL_HD (T v){ T t=::tan(v); T s=T(1)+t*t; return T(2)*t*s; }; return _lift_unary_2<T,N>(x,fa,df,d2); }
template <typename T, int N> CDUAL_HD MultiDual2<T,N> exp(const MultiDual2<T,N>& x){ T a=x.f; T fa=::exp(a); auto df=[] CDUAL_HD (T v){return ::exp(v);}; auto d2=[] CDUAL_HD (T v){return ::exp(v);}; return _lift_unary_2<T,N>(x,fa,df,d2); }
template <typename T, int N> CDUAL_HD MultiDual2<T,N> log(const MultiDual2<T,N>& x){ T a=x.f; T fa=::log(a); auto df=[] CDUAL_HD (T v){return T(1)/v;}; auto d2=[] CDUAL_HD (T v){return -T(1)/(v*v);}; return _lift_unary_2<T,N>(x,fa,df,d2); }
template <typename T, int N> CDUAL_HD MultiDual2<T,N> sqrt(const MultiDual2<T,N>& x){ T a=x.f; T fa=::sqrt(a); auto df=[] CDUAL_HD (T v){return T(0.5)/::sqrt(v);}; auto d2=[] CDUAL_HD (T v){ T r=::sqrt(v); return T(-0.25)/(r*r*r); }; return _lift_unary_2<T,N>(x,fa,df,d2); }
template <typename T, int N> CDUAL_HD MultiDual2<T,N> sinh(const MultiDual2<T,N>& x){ T a=x.f; T fa=::sinh(a); auto df=[] CDUAL_HD (T v){return ::cosh(v);}; auto d2=[] CDUAL_HD (T v){return ::sinh(v);}; return _lift_unary_2<T,N>(x,fa,df,d2); }
template <typename T, int N> CDUAL_HD MultiDual2<T,N> cosh(const MultiDual2<T,N>& x){ T a=x.f; T fa=::cosh(a); auto df=[] CDUAL_HD (T v){return ::sinh(v);}; auto d2=[] CDUAL_HD (T v){return ::cosh(v);}; return _lift_unary_2<T,N>(x,fa,df,d2); }
template <typename T, int N> CDUAL_HD MultiDual2<T,N> tanh(const MultiDual2<T,N>& x){ T a=x.f; T th=::tanh(a); T fa=th; auto df=[] CDUAL_HD (T v){ T th=::tanh(v); return T(1)-th*th; }; auto d2=[] CDUAL_HD (T v){ T th=::tanh(v); T s=T(1)-th*th; return -T(2)*th*s; }; return _lift_unary_2<T,N>(x,fa,df,d2); }
template <typename T, int N> CDUAL_HD MultiDual2<T,N> asin(const MultiDual2<T,N>& x){ T a=x.f; T fa=::asin(a); auto df=[] CDUAL_HD (T v){ return T(1)/::sqrt(T(1)-v*v); }; auto d2=[] CDUAL_HD (T v){ T den=::sqrt(T(1)-v*v); return v/(den*den*den); }; return _lift_unary_2<T,N>(x,fa,df,d2); }
template <typename T, int N> CDUAL_HD MultiDual2<T,N> acos(const MultiDual2<T,N>& x){ T a=x.f; T fa=::acos(a); auto df=[] CDUAL_HD (T v){ return -T(1)/::sqrt(T(1)-v*v); }; auto d2=[] CDUAL_HD (T v){ T den=::sqrt(T(1)-v*v); return -v/(den*den*den); }; return _lift_unary_2<T,N>(x,fa,df,d2); }
template <typename T, int N> CDUAL_HD MultiDual2<T,N> atan(const MultiDual2<T,N>& x){ T a=x.f; T fa=::atan(a); auto df=[] CDUAL_HD (T v){ return T(1)/(T(1)+v*v); }; auto d2=[] CDUAL_HD (T v){ T den=T(1)+v*v; return -T(2)*v/(den*den); }; return _lift_unary_2<T,N>(x,fa,df,d2); }
template <typename T, int N> CDUAL_HD MultiDual2<T,N> pow(const MultiDual2<T,N>& x, T c){ T a=x.f; T fa=::pow(a,c); auto df=[c] CDUAL_HD (T v){ return c*::pow(v, c-T(1)); }; auto d2=[c] CDUAL_HD (T v){ return c*(c-T(1))*::pow(v, c-T(2)); }; return _lift_unary_2<T,N>(x,fa,df,d2); }
template <typename T, int N> CDUAL_HD MultiDual2<T,N> expm1(const MultiDual2<T,N>& x){ T a=x.f; T fa=::expm1(a); auto df=[] CDUAL_HD (T v){ return ::exp(v); }; auto d2=[] CDUAL_HD (T v){ return ::exp(v); }; return _lift_unary_2<T,N>(x,fa,df,d2); }
template <typename T, int N> CDUAL_HD MultiDual2<T,N> log1p(const MultiDual2<T,N>& x){ T a=x.f; T fa=::log1p(a); auto df=[] CDUAL_HD (T v){ return T(1)/(T(1)+v); }; auto d2=[] CDUAL_HD (T v){ T d=T(1)/(T(1)+v); return -d*d; }; return _lift_unary_2<T,N>(x,fa,df,d2); }
template <typename T, int N> CDUAL_HD MultiDual2<T,N> exp2(const MultiDual2<T,N>& x){ T a=x.f; T fa=::exp2(a); auto df=[] CDUAL_HD (T v){return T(0.6931471805599453094)*::exp2(v);}; auto d2=[] CDUAL_HD (T v){ T t=::exp2(v); return T(0.48045301391820142466)*t; }; return _lift_unary_2<T,N>(x, fa, df, d2); }
template <typename T, int N> CDUAL_HD MultiDual2<T,N> log2(const MultiDual2<T,N>& x){ T a=x.f; T fa=::log2(a); auto df=[] CDUAL_HD (T v){ return T(1.4426950408889634074)/v; }; auto d2=[] CDUAL_HD (T v){ return -T(1.4426950408889634074)/(v*v); }; return _lift_unary_2<T,N>(x, fa, df, d2); }
template <typename T, int N> CDUAL_HD MultiDual2<T,N> log10(const MultiDual2<T,N>& x){ T a=x.f; T fa=::log10(a); auto df=[] CDUAL_HD (T v){ return T(0.43429448190325182765)/v; }; auto d2=[] CDUAL_HD (T v){ return -T(0.43429448190325182765)/(v*v); }; return _lift_unary_2<T,N>(x, fa, df, d2); }

// Variable makers and generic ops
template <typename T, int N> CDUAL_HD MultiDual<T,N> make_variable(T xi, int i){ MultiDual<T,N> x(xi); if(i>=0 && i<N) x.d[i]=T(1); return x; }
template <typename T> CDUAL_HD HyperDual<T> make_hyper(T xi, bool e1=false, bool e2=false){ return HyperDual<T>(xi, e1 ? T(1) : T(0), e2 ? T(1) : T(0), T(0)); }
template <typename T, int N> CDUAL_HD MultiDual2<T,N> make_variable2(T xi, int i){ MultiDual2<T,N> x(xi); if(i>=0 && i<N) x.g[i]=T(1); return x; }

template <class Num, typename Enable = std::enable_if_t<!std::is_arithmetic<Num>::value>>
CDUAL_HD Num pow(const Num& x, const Num& y){ return exp(y * log(x)); }
template <class Num, typename Enable = std::enable_if_t<!std::is_arithmetic<Num>::value>>
CDUAL_HD Num hypot(const Num& x, const Num& y){ return sqrt(x*x + y*y); }
template <class Num> CDUAL_HD Num logaddexp(const Num& a, const Num& b){
  Num m = a > b ? a : b;
  Num d = (a > b ? (b - a) : (a - b));
  return m + log1p(exp(d));
}
template <class Num> CDUAL_HD Num logsumexp(const Num* xs, int n){
  if (n<=0) return Num(0);
  Num m = xs[0];
  for (int i=1;i<n;++i) m = (xs[i] > m) ? xs[i] : m;
  Num s = exp(xs[0] - m);
  for (int i=1;i<n;++i) s = s + exp(xs[i] - m);
  return m + log(s);
}

// ---------- Additional generic math (AD-safe) ----------
template <class Num, typename Enable = std::enable_if_t<!std::is_arithmetic<Num>::value>>
CDUAL_HD Num asinh(const Num& x){ return log(x + sqrt(x*x + Num(1))); }
template <class Num, typename Enable = std::enable_if_t<!std::is_arithmetic<Num>::value>>
CDUAL_HD Num acosh(const Num& x){ return log(x + sqrt(x - Num(1)) * sqrt(x + Num(1))); }
template <class Num, typename Enable = std::enable_if_t<!std::is_arithmetic<Num>::value>>
CDUAL_HD Num atanh(const Num& x){ return Num(0.5) * log((Num(1) + x) / (Num(1) - x)); }
template <class Num, typename Enable = std::enable_if_t<!std::is_arithmetic<Num>::value>>
CDUAL_HD Num sigmoid(const Num& x){
  if (x.f >= 0){ Num e = exp(-x); return Num(1) / (Num(1) + e); }
  else { Num e = exp(x); return e / (Num(1) + e); }
}
template <class Num, typename Enable = std::enable_if_t<!std::is_arithmetic<Num>::value>>
CDUAL_HD Num softplus(const Num& x){ return (x.f > 0) ? x + log1p(exp(-x)) : log1p(exp(x)); }
template <class Num, typename Enable = std::enable_if_t<!std::is_arithmetic<Num>::value>>
CDUAL_HD Num logit(const Num& p, const Num& eps = Num(0)){
  Num pp = p;
  if (eps.f > 0){
    if (pp.f < eps.f) pp = eps;
    if (pp.f > (Num(1)-eps).f) pp = Num(1)-eps;
  }
  return log(pp / (Num(1) - pp));
}

// ---------------- Digamma / Trigamma + lgamma AD + erfcx + normals + misc ----------------
namespace detail {
template <typename T> CDUAL_HD inline T cot(T x){ return ::cos(x) / ::sin(x); }
template <typename T>
CDUAL_HD T digamma_scalar(T x){
  if (x <= T(0)){ T pix=CUDART_PI*x; return digamma_scalar(T(1)-x) - CUDART_PI * cot(pix); }
  T res=T(0); while(x<T(8)){ res -= T(1)/x; x += T(1); }
  T inv=T(1)/x, inv2=inv*inv;
  T t2 = - T(1)/T(12) * inv2;
  T t4 = + T(1)/T(120) * inv2*inv2;
  T t6 = - T(1)/T(252) * inv2*inv2*inv2;
  T t8 = + T(1)/T(240) * inv2*inv2*inv2*inv2;
  T t10= - T(1)/T(132) * inv2*inv2*inv2*inv2*inv2;
  return res + ::log(x) - T(0.5)*inv + t2 + t4 + t6 + t8 + t10;
}
template <typename T>
CDUAL_HD T trigamma_scalar(T x){
  if (x <= T(0)){ T pix=CUDART_PI*x; T s=::sin(pix); T csc2=T(1)/(s*s); return CUDART_PI*CUDART_PI*csc2 - trigamma_scalar(T(1)-x); }
  T res=T(0); while(x<T(8)){ res += T(1)/(x*x); x+=T(1); }
  T inv=T(1)/x, inv2=inv*inv, inv3=inv2*inv, inv5=inv3*inv2*inv, inv7=inv5*inv2*inv, inv9=inv7*inv2*inv, inv11=inv9*inv2*inv;
  return res + inv + T(0.5)*inv2 + T(1.0/6.0)*inv3 - T(1.0/30.0)*inv5 + T(1.0/42.0)*inv7 - T(1.0/30.0)*inv9 + T(5.0/66.0)*inv11;
}
template <typename T> CDUAL_HD T erfcx_scalar(T x){ return ::exp(x*x) * ::erfc(x); }
} // namespace detail

template <typename T> CDUAL_HD Dual<T> lgamma(const Dual<T>& x){ Dual<T> r; r.f=::lgamma(x.f); T psi=detail::digamma_scalar(x.f); r.d=psi*x.d; return r; }
template <typename T> CDUAL_HD HyperDual<T> lgamma(const HyperDual<T>& x){ HyperDual<T> r; r.f=::lgamma(x.f); T psi=detail::digamma_scalar(x.f); T tri=detail::trigamma_scalar(x.f); r.e1=psi*x.e1; r.e2=psi*x.e2; r.e12=tri*x.e1*x.e2 + psi*x.e12; return r; }
template <typename T, int N> CDUAL_HD MultiDual<T,N> lgamma(const MultiDual<T,N>& x){ MultiDual<T,N> r; r.f=::lgamma(x.f); T psi=detail::digamma_scalar(x.f); for(int i=0;i<N;++i) r.d[i]=psi*x.d[i]; return r; }
template <typename T, int N> CDUAL_HD MultiDual2<T,N> lgamma(const MultiDual2<T,N>& x){ T fa=::lgamma(x.f); auto df=[] CDUAL_HD (T v){ return detail::digamma_scalar(v); }; auto d2=[] CDUAL_HD (T v){ return detail::trigamma_scalar(v); }; return _lift_unary_2<T,N>(x,fa,df,d2); }

template <typename T> CDUAL_HD Dual<T> erfcx(const Dual<T>& x){ Dual<T> r; T a=x.f; r.f=detail::erfcx_scalar(a); T d1=T(2)*a*r.f - T(2)/CUDART_SQRT_PI; r.d=d1*x.d; return r; }
template <typename T> CDUAL_HD HyperDual<T> erfcx(const HyperDual<T>& x){ HyperDual<T> r; T a=x.f; r.f=detail::erfcx_scalar(a); T d1=T(2)*a*r.f - T(2)/CUDART_SQRT_PI; T d2=(T(2)+T(4)*a*a)*r.f - T(4)*a/CUDART_SQRT_PI; r.e1=d1*x.e1; r.e2=d1*x.e2; r.e12=d2*x.e1*x.e2 + d1*x.e12; return r; }
template <typename T, int N> CDUAL_HD MultiDual<T,N> erfcx(const MultiDual<T,N>& x){ MultiDual<T,N> r; T a=x.f; r.f=detail::erfcx_scalar(a); T d1=T(2)*a*r.f - T(2)/CUDART_SQRT_PI; for(int i=0;i<N;++i) r.d[i]=d1*x.d[i]; return r; }
template <typename T, int N> CDUAL_HD MultiDual2<T,N> erfcx(const MultiDual2<T,N>& x){ T fa=detail::erfcx_scalar(x.f); auto df=[] CDUAL_HD (T v){ return T(2)*v*detail::erfcx_scalar(v) - T(2)/CUDART_SQRT_PI; }; auto d2=[] CDUAL_HD (T v){ T rf=detail::erfcx_scalar(v); return (T(2)+T(4)*v*v)*rf - T(4)*v/CUDART_SQRT_PI; }; return _lift_unary_2<T,N>(x,fa,df,d2); }

template <class Num> CDUAL_HD Num normal_cdf(const Num& x){ const Num inv_sqrt2 = Num(0.70710678118654752440084436210485); return Num(0.5) * erfc(-x * inv_sqrt2); }
template <class Num> CDUAL_HD Num normal_logcdf(const Num& x){ const Num inv_sqrt2 = Num(0.70710678118654752440084436210485); return log(Num(0.5) * erfc(-x * inv_sqrt2)); }
template <class Num> CDUAL_HD Num normal_logpdf(const Num& x, const Num& mu, const Num& sigma){ const Num log_sqrt2pi = Num(0.91893853320467274178032973640562); Num z=(x-mu)/sigma; return -log_sqrt2pi - log(sigma) - Num(0.5)*z*z; }

template <class Num> CDUAL_HD Num sinc(const Num& x){ Num ax=x; if(ax.f<0) ax=-ax; if(ax.f<1e-4){ Num x2=x*x; return Num(1) - x2/Num(6) + x2*x2/Num(120); } else { return sin(x)/x; } }
template <class Num> CDUAL_HD Num poly_eval(const Num* c, int n, const Num& x){ Num r=Num(0); for(int i=n-1;i>=0;--i) r=r*x + c[i]; return r; }
template <class Num> CDUAL_HD void softmax(const Num* xs, int n, Num* out){ if(n<=0) return; Num m=xs[0]; for(int i=1;i<n;++i) m=(xs[i]>m)?xs[i]:m; Num s=exp(xs[0]-m); for(int i=1;i<n;++i) s=s+exp(xs[i]-m); for(int i=0;i<n;++i) out[i]=exp(xs[i]-m)/s; }

} // namespace cudadual



// ============================================================================
// Extra math: beta/lbeta, atan2 (binary AD), sinc/sinhc, log1mexp/log1pexp/logsigmoid,
// softmax/log_softmax (vector helpers)
// ============================================================================

// ---- beta & lbeta
template <class Num, typename Enable = std::enable_if_t<!std::is_arithmetic<Num>::value>>
CDUAL_HD Num lbeta(const Num& a, const Num& b){ return lgamma(a) + lgamma(b) - lgamma(a+b); }
template <class Num, typename Enable = std::enable_if_t<!std::is_arithmetic<Num>::value>>
CDUAL_HD Num beta (const Num& a, const Num& b){ return exp(lbeta(a,b)); }

// ---- atan2(y, x) for AD types (value-stable across quadrants)
// z = atan2(y, x); r2 = x^2 + y^2; zx = -y/r2; zy = x/r2
// Hessian uses: zxx = 2xy/r^4; zyy = -2xy/r^4; zxy = (y^2 - x^2)/r^4
template <typename T> CDUAL_HD Dual<T> atan2(const Dual<T>& y, const Dual<T>& x){
  T xf=x.f, yf=y.f, r2=xf*xf + yf*yf;
  T zx = -yf / r2, zy = xf / r2;
  return Dual<T>(::atan2(yf, xf), zx*x.d + zy*y.d);
}
template <typename T> CDUAL_HD HyperDual<T> atan2(const HyperDual<T>& y, const HyperDual<T>& x){
  T xf=x.f, yf=y.f, r2=xf*xf + yf*yf, r4=r2*r2;
  T zx = -yf / r2, zy = xf / r2;
  T zxx = (2*xf*yf) / r4;
  T zyy = (-2*xf*yf) / r4;
  T zxy = (yf*yf - xf*xf) / r4;
  HyperDual<T> z;
  z.f  = ::atan2(yf, xf);
  z.e1 = zx*x.e1 + zy*y.e1;
  z.e2 = zx*x.e2 + zy*y.e2;
  z.e12= zx*x.e12 + zy*y.e12 + zxx*x.e1*x.e2 + zyy*y.e1*y.e2 + zxy*(x.e1*y.e2 + y.e1*x.e2);
  return z;
}
template <typename T, int N> CDUAL_HD MultiDual<T,N> atan2(const MultiDual<T,N>& y, const MultiDual<T,N>& x){
  T xf=x.f, yf=y.f, r2=xf*xf + yf*yf;
  T zx = -yf / r2, zy = xf / r2;
  MultiDual<T,N> z; z.f=::atan2(yf,xf);
  for(int i=0;i<N;++i) z.d[i] = zx*x.d[i] + zy*y.d[i];
  return z;
}
template <typename T, int N> CDUAL_HD MultiDual2<T,N> atan2(const MultiDual2<T,N>& y, const MultiDual2<T,N>& x){
  T xf=x.f, yf=y.f, r2=xf*xf + yf*yf, r4=r2*r2;
  T zx = -yf / r2, zy = xf / r2;
  T zxx = (2*xf*yf) / r4;
  T zyy = (-2*xf*yf) / r4;
  T zxy = (yf*yf - xf*xf) / r4;
  MultiDual2<T,N> z; z.f=::atan2(yf,xf);
  // gradient
  for(int i=0;i<N;++i) z.g[i] = zx*x.g[i] + zy*y.g[i];
  // Hessian
  for(int i=0;i<N;++i){
    for(int j=0;j<N;++j){
      z.H[i*N+j] = zx*x.H[i*N+j] + zy*y.H[i*N+j]
                 + zxx * x.g[i]*x.g[j] + zyy * y.g[i]*y.g[j]
                 + zxy * (x.g[i]*y.g[j] + y.g[i]*x.g[j]);
    }
  }
  return z;
}

// ---- sinc / sinhc (stable near 0 using series)
template <class Num, typename Enable = std::enable_if_t<!std::is_arithmetic<Num>::value>>
CDUAL_HD Num sinc(const Num& x){
    const double th = 1e-4;
  if (fabs(x.f) > th) return sin(x)/x;
  // series: 1 - x^2/6 + x^4/120 - x^6/5040
  Num x2 = x*x;
  return Num(1) - x2*(Num(1)/Num(6)) + (x2*x2)*(Num(1)/Num(120)) - (x2*x2*x2)*(Num(1)/Num(5040));
}
template <class Num, typename Enable = std::enable_if_t<!std::is_arithmetic<Num>::value>>
CDUAL_HD Num sinhc(const Num& x){
  const double th = 1e-4;
  if (fabs(x.f) > th) return sinh(x)/x;
  Num x2 = x*x;
  return Num(1) + x2*(Num(1)/Num(6)) + (x2*x2)*(Num(1)/Num(120)) + (x2*x2*x2)*(Num(1)/Num(5040));
}

// ---- log1mexp / log1pexp / logsigmoid (stable)
template <class Num, typename Enable = std::enable_if_t<!std::is_arithmetic<Num>::value>>
CDUAL_HD Num log1mexp(const Num& x){
  // returns log(1 - exp(-x)) for x>0
  const double ln2 = 0.6931471805599453094;
  if (x.f <= ln2) return log(-expm1(-x)); // use expm1 for small x
  else            return log1p(-exp(-x));
}
template <class Num, typename Enable = std::enable_if_t<!std::is_arithmetic<Num>::value>>
CDUAL_HD Num log1pexp(const Num& x){
  // log(1+exp(x)) with stable branching
  if (x.f > 0) return x + log1p(exp(-x));
  else         return log1p(exp(x));
}
template <class Num, typename Enable = std::enable_if_t<!std::is_arithmetic<Num>::value>>
CDUAL_HD Num logsigmoid(const Num& x){
  if (x.f >= 0) return -log1p(exp(-x));
  else          return x - log1p(exp(x));
}

// ---- array helpers: softmax / log_softmax (stable)
// out[i] = exp(x[i]-m) / sum_j exp(x[j]-m)
template <class Num>
CDUAL_HD void softmax(const Num* __restrict__ xs, int n, Num* __restrict__ out){
  if (n<=0) return;
  Num m = xs[0];
  for (int i=1;i<n;++i) m = (xs[i] > m) ? xs[i] : m;
  Num s = exp(xs[0] - m);
  for (int i=1;i<n;++i) s = s + exp(xs[i] - m);
  for (int i=0;i<n;++i) out[i] = exp(xs[i] - m) / s;
}
template <class Num>
CDUAL_HD void log_softmax(const Num* __restrict__ xs, int n, Num* __restrict__ out){
  if (n<=0) return;
  Num m = xs[0];
  for (int i=1;i<n;++i) m = (xs[i] > m) ? xs[i] : m;
  Num s = exp(xs[0] - m);
  for (int i=1;i<n;++i) s = s + exp(xs[i] - m);
  Num logZ = m + log(s);
  for (int i=0;i<n;++i) out[i] = xs[i] - logZ;
}
