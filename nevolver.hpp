#ifndef NEVOLVER_H
#define NEVOLVER_H

#include "random.hpp"
#include <cassert>
#include <functional>
#include <iostream>
#include <list>
#include <memory>
#include <optional>
#include <set>
#include <variant>
#include <vector>

#define _USE_MATH_DEFINES
#include <cmath>

#ifndef M_PI
#define M_PI (3.14159265358979323846)
#endif

#ifndef M_PIl
#define M_PIl (3.14159265358979323846264338327950288)
#endif

#define NEVOLVER_WIDE8

namespace Nevolver {
#ifdef NEVOLVER_WIDE8

#define NEVOLVER_WIDE

typedef float Float8 __attribute__((vector_size(32)));
using NeuroFloat = Float8;
constexpr int NeuroFloatWidth = 8;

#define NEUROWIDE(_v_, _x_)                                                    \
  NeuroFloat _v_ {                                                             \
    float(_x_), float(_x_), float(_x_), float(_x_), float(_x_), float(_x_),    \
        float(_x_), float(_x_),                                                \
  }
constexpr Float8 NeuroFloatZeros =
    Float8{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
constexpr Float8 NeuroFloatOnes =
    Float8{1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};

inline void print(NeuroFloat f) {
  std::cout << "[";
  for (int i = 0; i < Nevolver::NeuroFloatWidth; i++) {
    std::cout << f[i] << " ";
  }
  std::cout << "]";
}

#else

using NeuroFloat = float;

constexpr NeuroFloat NeuroFloatZeros = 0.0;
constexpr NeuroFloat NeuroFloatOnes = 1.0;
#define NEUROWIDE(_v_, _x_)                                                    \
  NeuroFloat _v_ { float(_x_) }

inline void print(NeuroFloat f) { std::cout << f; }

#endif

class Random {
public:
  static double nextDouble() {
    return double(_gen()) * (1.0 / double(xorshift::max()));
  }

  static NeuroFloat next() {
#ifdef NEVOLVER_WIDE
    NeuroFloat res;
    for (int i = 0; i < NeuroFloatWidth; i++) {
      res[0] = nextDouble();
    }
    return res;
#else
    return nextDouble();
#endif
  }

  static uint32_t nextUInt() { return _gen(); }

  static double normalDouble(double mean, double stdDeviation) {
    double u1 = 0.0;
    while (u1 == 0.0) {
      u1 = nextDouble();
    }

    auto u2 = nextDouble();
    auto rstdNorm = std::sqrt(-2.0 * std::log(u1)) * std::sin(2.0 * M_PI * u2);

    return mean + stdDeviation * rstdNorm;
  }

  static NeuroFloat normal(double mean, double stdDeviation) {
#ifdef NEVOLVER_WIDE
    NeuroFloat res;
    for (int i = 0; i < NeuroFloatWidth; i++) {
      res[0] = normalDouble(mean, stdDeviation);
    }
    return res;
#else
    return normalDouble(mean, stdDeviation);
#endif
  }

private:
#ifdef NDEBUG
  static inline std::random_device _rd{};
  static inline xorshift _gen{_rd};
#else
  static inline xorshift _gen{};
#endif
};

class Node;
class InputNode;
class HiddenNode;
class Weight;

using AnyNode = std::variant<InputNode, HiddenNode>;
using Group = std::vector<std::reference_wrapper<AnyNode>>;
} // namespace Nevolver

// Foundation
#include "connections.hpp"
#include "squash.hpp"

// Nodes
#include "node.hpp"
#include "nodes/annhidden.hpp"

#endif /* NEVOLVER_H */
