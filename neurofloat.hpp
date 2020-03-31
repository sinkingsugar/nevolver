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
}

constexpr static auto vector_width = 128;
constexpr static auto vector_align = vector_width * 4;
using NeuroFloat = Vector<vector_align, vector_width, float>;
