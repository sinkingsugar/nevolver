#include <array>
#include <iostream>

template<int A, int W, typename T>
struct alignas(A) Vector {
    using Vec = Vector<A, W, T>;
    constexpr static auto Width = W;

    Vector() = default;

    Vector(T repeat) {
        for(auto  i = 0; i < W; i++) {
            vec[i] = repeat;
        }
    }

    void print() {
        for(auto  i = 0; i < W; i++) {
            std::cout << vec[i] << " ";
        }
    }

    typedef T VT __attribute__((vector_size(A)));
    
    VT vec;

    Vec operator+(const Vec &other) {
        Vec res;
        res.vec = vec + other.vec;
        return res;
    }

    Vec operator+(const T &other) {
        Vec res;
        res.vec = vec + other;
        return res;
    }

    Vec operator*(const Vec &other) {
        Vec res;
        res.vec = vec * other.vec;
        return res;
    }

    Vec operator*(const T &other) {
        Vec res;
        res.vec = vec * other;
        return res;
    }

    Vec operator-() const {
        Vec res;
        res.vec = -vec;
        return res;
    }
};

namespace std {
    template<int A, int W, typename T>
    Vector<A, W, T> sqrt(const Vector<A, W, T> x) {
        Vector<A, W, T> res;
        for(auto i = 0; i <  Vector<A, W, T>::Width; i++) {
            res.vec[i] = __builtin_sqrt(x.vec[i]);
        }
        return res;
    }

    template<int A, int W, typename T>
    Vector<A, W, T> log(const Vector<A, W, T> x) {
        Vector<A, W, T> res;
        for(auto i = 0; i <  Vector<A, W, T>::Width; i++) {
            res.vec[i] = __builtin_log(x.vec[i]);
        }
        return res;
    }

    template<int A, int W, typename T>
    Vector<A, W, T> sin(const Vector<A, W, T> x) {
        Vector<A, W, T> res;
        for(auto i = 0; i <  Vector<A, W, T>::Width; i++) {
            res.vec[i] = __builtin_sin(x.vec[i]);
        }
        return res;
    }

    template<int A, int W, typename T>
    Vector<A, W, T> cos(const Vector<A, W, T> x) {
        Vector<A, W, T> res;
        for(auto i = 0; i <  Vector<A, W, T>::Width; i++) {
            res.vec[i] = __builtin_cos(x.vec[i]);
        }
        return res;
    }

    template<int A, int W, typename T>
    Vector<A, W, T> exp(const Vector<A, W, T> x) {
        Vector<A, W, T> res;
        for(auto i = 0; i <  Vector<A, W, T>::Width; i++) {
            res.vec[i] = __builtin_exp(x.vec[i]);
        }
        return res;
    }

    template<int A, int W, typename T>
    Vector<A, W, T> tanh(const Vector<A, W, T> x) {
        Vector<A, W, T> res;
        for(auto i = 0; i <  Vector<A, W, T>::Width; i++) {
            res.vec[i] = __builtin_tanh(x.vec[i]);
        }
        return res;
    }

    template<int A, int W, typename T, typename P>
    Vector<A, W, T> pow(const Vector<A, W, T> x, P p) {
        Vector<A, W, T> res;
        for(auto i = 0; i <  Vector<A, W, T>::Width; i++) {
            res.vec[i] = __builtin_pow(x.vec[i], p);
        }
        return res;
    }

    template<int A, int W, typename T>
    Vector<A, W, T> fabs(const Vector<A, W, T> x) {
        Vector<A, W, T> res;
        for(auto i = 0; i <  Vector<A, W, T>::Width; i++) {
            res.vec[i] = __builtin_fabs(x.vec[i]);
        }
        return res;
    }
}

#ifdef NEVOLVER_WIDE

constexpr static auto vector_width = 128;
constexpr static auto vector_align = vector_width * 4;
using NeuroFloat = Vector<vector_align, vector_width, float>;

inline std::ostream &operator<<(std::ostream &os,
                                const NeuroFloat &f) {
  os << "[";
  for (int i = 0; i < NeuroFloat::Width; i++) {
    os << f.vec[i] << " ";
  }
  os << "]";
  return os;
}

template<typename T>
inline NeuroFloat operator+(const T &a, const NeuroFloat &b) {
    return a + b.vec;
}

template<typename T>
inline NeuroFloat operator-(const T &a, const NeuroFloat &b) {
    return a - b.vec;
}

template<typename T>
inline NeuroFloat operator/(const T &a, const NeuroFloat &b) {
    return a / b.vec;
}

#else

using NeuroFloat = float;

#endif
