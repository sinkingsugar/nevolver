#ifdef NEVOLVER_WIDE

/*
Any size is fine, perf gains are ensured but:
Overall -march=sandybridge, 16 seems to be the best in terms of pure SIMD ops
*/

#include <array>
#include <cmath>
#include <cstdint>
#include <functional>
#include <iostream>
#include <limits>

template <typename T, typename = void> struct is_iterable : std::false_type {};

template <typename T>
struct is_iterable<T, std::void_t<decltype(std::begin(std::declval<T>()))>>
    : std::true_type {};

template <int A, int W, typename T> struct alignas(A) Vector {
  using Vec = Vector<A, W, T>;
  using ValueType = T;
  constexpr static auto Width = W;

  Vector() = default;

  Vector(T repeat) {
    for (auto i = 0; i < W; i++) {
      vec[i] = repeat;
    }
  }

  template <typename Iterable,
            std::enable_if_t<is_iterable<Iterable>::value, int> = 0>
  Vector(Iterable iter) {
    auto idx = 0;
    while (true) {
      for (auto &v : iter) {
        vec[idx++] = T(v);
        if (idx == W)
          return;
      }
    }
  }

  typedef T VT __attribute__((vector_size(A)));

  VT vec;

  Vec operator+(const Vec &other) const {
    Vec res;
    res.vec = vec + other.vec;
    return res;
  }

  Vec operator+(const T &other) const {
    Vec res;
    res.vec = vec + other;
    return res;
  }

  Vec operator*(const Vec &other) const {
    Vec res;
    res.vec = vec * other.vec;
    return res;
  }

  Vec operator*(const T &other) const {
    Vec res;
    res.vec = vec * other;
    return res;
  }

  Vec operator-() const {
    Vec res;
    res.vec = -vec;
    return res;
  }

  Vec &operator+=(const Vec &other) {
    vec += other.vec;
    return *this;
  }

  Vec &operator/=(const Vec &other) {
    vec /= other.vec;
    return *this;
  }

  template <class Archive>
  void save(Archive &ar, std::uint32_t const version) const {
    std::array<T, W> vals;
    for (auto i = 0; i < W; i++) {
      vals[i] = vec[i];
    }
    ar(vals);
  }

  template <class Archive> void load(Archive &ar, std::uint32_t const version) {
    std::array<T, W> vals;
    ar(vals);
    for (auto i = 0; i < W; i++) {
      vec[i] = vals[i];
    }
  }
};

namespace std {
#define NFLOAT_UNARY(__op__)                                                   \
  template <typename TVec> TVec __op__(const TVec x) {                         \
    TVec res;                                                                  \
    for (auto i = 0; i < TVec::Width; i++) {                                   \
      res.vec[i] = __builtin_##__op__(x.vec[i]);                               \
    }                                                                          \
    return res;                                                                \
  }

NFLOAT_UNARY(sqrt);
NFLOAT_UNARY(log);
NFLOAT_UNARY(sin);
NFLOAT_UNARY(cos);
NFLOAT_UNARY(exp);
NFLOAT_UNARY(tanh);
NFLOAT_UNARY(fabs);

template <typename TVec, typename P> TVec pow(const TVec x, P p) {
  TVec res;
  for (auto i = 0; i < TVec::Width; i++) {
    res.vec[i] = __builtin_pow(x.vec[i], p);
  }
  return res;
}
} // namespace std

constexpr static auto vector_width = NEVOLVER_WIDE;
constexpr static auto vector_align = vector_width * 4;
using NeuroFloat = Vector<vector_align, vector_width, float>;
using NeuroInt = Vector<vector_align, vector_width, int>;

template <class T>
typename std::enable_if<!std::numeric_limits<T>::is_integer, NeuroInt>::type
almost_equal(T x, T y, int ulp) {
  // the machine epsilon has to be scaled to the magnitude of the values used
  // and multiplied by the desired precision in ULPs (units in the last place)
  return std::fabs(x - y) <=
             std::numeric_limits<T>::epsilon() * std::fabs(x + y) * ulp
         // unless the result is subnormal
         || std::fabs(x - y) < std::numeric_limits<T>::min();
}

inline std::ostream &operator<<(std::ostream &os, const NeuroFloat &f) {
  os << "[";
  for (auto i = 0; i < NeuroFloat::Width; i++) {
    os << f.vec[i] << " ";
  }
  os << "]";
  return os;
}

inline std::ostream &operator<<(std::ostream &os, const NeuroInt &f) {
  os << "[";
  for (auto i = 0; i < NeuroFloat::Width; i++) {
    os << f.vec[i] << " ";
  }
  os << "]";
  return os;
}

inline NeuroInt operator&&(const NeuroInt &a, const NeuroInt &b) {
  NeuroInt res;
  res.vec = a.vec && b.vec;
  return res;
}

inline NeuroInt operator||(const NeuroInt &a, const NeuroInt &b) {
  NeuroInt res;
  res.vec = a.vec && b.vec;
  return res;
}

inline NeuroInt operator==(const NeuroFloat &a, const NeuroFloat &b) {
  return almost_equal(a, b, 2);
}

inline NeuroInt operator<(const NeuroFloat &a, const NeuroFloat &b) {
  NeuroInt res;
  res.vec = a.vec < b.vec;
  return res;
}

inline NeuroInt operator<=(const NeuroFloat &a, const NeuroFloat &b) {
  NeuroInt res;
  res.vec = a.vec <= b.vec;
  return res;
}

inline NeuroInt operator>(const NeuroFloat &a, const NeuroFloat &b) {
  return b < a;
}

inline NeuroInt operator>=(const NeuroFloat &a, const NeuroFloat &b) {
  return b <= a;
}

inline NeuroInt isnan(const NeuroFloat &a) {
  NeuroInt res;
  for (auto i = 0; i < NeuroFloat::Width; i++) {
    if (std::isnan(a.vec[i]))
      res.vec[i] = -1;
    else
      res.vec[i] = 0;
  }
  return res;
}

inline NeuroInt isinf(const NeuroFloat &a) {
  NeuroInt res;
  for (auto i = 0; i < NeuroFloat::Width; i++) {
    if (std::isinf(a.vec[i]))
      res.vec[i] = -1;
    else
      res.vec[i] = 0;
  }
  return res;
}

inline NeuroFloat operator-(const NeuroFloat &a, const NeuroFloat &b) {
  NeuroFloat res;
  res.vec = a.vec - b.vec;
  return res;
}

inline NeuroFloat operator/(const NeuroFloat &a, const NeuroFloat &b) {
  NeuroFloat res;
  res.vec = a.vec / b.vec;
  return res;
}

inline NeuroFloat operator+(const double &a, const NeuroFloat &b) {
  NeuroFloat res;
  res = NeuroFloat(a) + b;
  return res;
}

inline NeuroFloat operator*(const double &a, const NeuroFloat &b) {
  NeuroFloat res;
  res = NeuroFloat(a) * b;
  return res;
}

inline NeuroFloat operator-(const double &a, const NeuroFloat &b) {
  NeuroFloat res;
  res = NeuroFloat(a) - b;
  return res;
}

inline NeuroFloat operator/(const double &a, const NeuroFloat &b) {
  NeuroFloat res;
  res = NeuroFloat(a) / b;
  return res;
}

inline NeuroFloat either(NeuroInt pred, NeuroFloat positive,
                         NeuroFloat negative) {
  NeuroFloat res;
  for (auto i = 0; i < vector_width; i++) {
    if (pred.vec[i] < 0)
      res.vec[i] = positive.vec[i];
    else
      res.vec[i] = negative.vec[i];
  }
  return res;
}

inline bool any(NeuroInt pred) {
  for (auto i = 0; i < vector_width; i++) {
    if (pred.vec[i] < 0)
      return true;
  }
  return false;
}

inline bool all(NeuroInt pred) {
  for (auto i = 0; i < vector_width; i++) {
    if (pred.vec[i] != 0)
      return false;
  }
  return true;
}

template <class F>
inline NeuroFloat::ValueType reduce(F &&f, const NeuroFloat &vec,
                                    const NeuroFloat::ValueType init = 0.0) {
  NeuroFloat::ValueType res = init;
  for (auto i = 0; i < NeuroFloat::Width; i++) {
    auto v = f(res, vec.vec[i]);
    // trick to avoid nans
    // TODO profile if it's worth it
    res = v == v ? v : res;
  }
  return res;
}

inline NeuroFloat softmax(const NeuroFloat &vec) {
  auto e = std::exp(vec);
  auto s = reduce(std::plus<NeuroFloat::ValueType>(), e);
  return e / s;
}

inline NeuroFloat::ValueType mean(const NeuroFloat &vec) {
  return reduce(std::plus<NeuroFloat::ValueType>(), vec) / NeuroFloat::Width;
}

inline NeuroFloat::ValueType logSumExp(const NeuroFloat &vec) {
  auto e = std::exp(vec);
  auto s = reduce(std::plus<NeuroFloat::ValueType>(), e);
  return std::log(s);
}

inline NeuroFloat::ValueType meanGeometric(const NeuroFloat &vec) {
  return std::pow(reduce(std::multiplies<NeuroFloat::ValueType>(), vec, 1.0),
                  NeuroFloat::ValueType(1.0) /
                      NeuroFloat::ValueType(NeuroFloat::Width));
}

inline NeuroFloat::ValueType meanHarmonic(const NeuroFloat &vec) {
  auto r = 1.0 / vec;
  auto x = reduce(std::plus<NeuroFloat::ValueType>(), r);
  return NeuroFloat::ValueType(NeuroFloat::Width) / x;
}

#else

using NeuroFloat = float;
using NeuroInt = int;

inline NeuroFloat either(bool pred, NeuroFloat positive, NeuroFloat negative) {
  if (pred)
    return positive;
  else
    return negative;
}

inline NeuroFloat mean(const NeuroFloat &single) { return single; }

inline NeuroFloat logSumExp(const NeuroFloat &single) { return single; }

#endif

#if 0

int main(int argc, char** argv) {
  std::vector<float> xx = {0.7, 0.7, 0.99, 0.97};
  NeuroFloat x(xx);
  std::cout << mean(x) << "\n";
  std::cout << meanGeometric(x) << "\n";
  std::cout << meanHarmonic(x) << "\n";
}

#endif
