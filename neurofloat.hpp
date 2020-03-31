#ifdef NEVOLVER_WIDE

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
    std::vector<ValueType> vals;
    for (auto i = 0; i < W; i++) {
      vals.emplace_back(vec[i]);
    }
    ar(vals);
  }

  template <class Archive> void load(Archive &ar, std::uint32_t const version) {
    std::vector<ValueType> vals;
    ar(vals);
    for (auto i = 0; i < W; i++) {
      vec[i] = vals[i];
    }
  }
};

namespace std {
template <typename TVec> TVec sqrt(const TVec x) {
  TVec res;
  for (auto i = 0; i < TVec::Width; i++) {
    res.vec[i] = __builtin_sqrt(x.vec[i]);
  }
  return res;
}

template <typename TVec> TVec log(const TVec x) {
  TVec res;
  for (auto i = 0; i < TVec::Width; i++) {
    res.vec[i] = __builtin_log(x.vec[i]);
  }
  return res;
}

template <typename TVec> TVec sin(const TVec x) {
  TVec res;
  for (auto i = 0; i < TVec::Width; i++) {
    res.vec[i] = __builtin_sin(x.vec[i]);
  }
  return res;
}

template <typename TVec> TVec cos(const TVec x) {
  TVec res;
  for (auto i = 0; i < TVec::Width; i++) {
    res.vec[i] = __builtin_cos(x.vec[i]);
  }
  return res;
}

template <typename TVec> TVec exp(const TVec x) {
  TVec res;
  for (auto i = 0; i < TVec::Width; i++) {
    res.vec[i] = __builtin_exp(x.vec[i]);
  }
  return res;
}

template <typename TVec> TVec tanh(const TVec x) {
  TVec res;
  for (auto i = 0; i < TVec::Width; i++) {
    res.vec[i] = __builtin_tanh(x.vec[i]);
  }
  return res;
}

template <typename TVec, typename P> TVec pow(const TVec x, P p) {
  TVec res;
  for (auto i = 0; i < TVec::Width; i++) {
    res.vec[i] = __builtin_pow(x.vec[i], p);
  }
  return res;
}

template <typename TVec> TVec fabs(const TVec x) {
  TVec res;
  for (auto i = 0; i < TVec::Width; i++) {
    res.vec[i] = __builtin_fabs(x.vec[i]);
  }
  return res;
}
} // namespace std

constexpr static auto vector_width = 128;
constexpr static auto vector_align = vector_width * 4;
using NeuroFloat = Vector<vector_align, vector_width, float>;
using NeuroInt = Vector<vector_align, vector_width, int>;

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

inline NeuroInt operator<(const NeuroFloat &a, const NeuroFloat &b) {
  NeuroInt res;
  res.vec = a.vec < b.vec;
  return res;
}

inline NeuroInt operator>(const NeuroFloat &a, const NeuroFloat &b) {
  return b < a;
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
    if (pred.vec[i] == 0)
      res.vec[i] = positive.vec[i];
    else
      res.vec[i] = negative.vec[i];
  }
  return res;
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

#endif
