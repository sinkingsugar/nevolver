#ifndef NEVOLVER_H
#define NEVOLVER_H

#include "random.hpp"
#include <cassert>
#include <functional>
#include <iostream>
#include <list>
#include <memory>
#include <optional>
#include <ostream>
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

#define NEVOLVER_VERSION 0x1

// #define NEVOLVER_WIDE4

namespace Nevolver {
#ifdef NEVOLVER_WIDE8

#define NEVOLVER_WIDE

typedef float NeuroFloat __attribute__((vector_size(32)));
constexpr int NeuroFloatWidth = 8;

#define NEUROWIDE(_v_, _x_)                                                    \
  NeuroFloat _v_ {                                                             \
    float(_x_), float(_x_), float(_x_), float(_x_), float(_x_), float(_x_),    \
        float(_x_), float(_x_),                                                \
  }
constexpr NeuroFloat NeuroFloatZeros =
    NeuroFloat{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
constexpr NeuroFloat NeuroFloatOnes =
    NeuroFloat{1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};

#elif defined(NEVOLVER_WIDE4)

#define NEVOLVER_WIDE

typedef float NeuroFloat __attribute__((vector_size(16)));
constexpr int NeuroFloatWidth = 4;

#define NEUROWIDE(_v_, _x_)                                                    \
  NeuroFloat _v_ { float(_x_), float(_x_), float(_x_), float(_x_) }

constexpr NeuroFloat NeuroFloatZeros = NeuroFloat{0.0, 0.0, 0.0, 0.0};
constexpr NeuroFloat NeuroFloatOnes = NeuroFloat{1.0, 1.0, 1.0, 1.0};

#else

using NeuroFloat = float;
constexpr int NeuroFloatWidth = 1;

#define NEUROWIDE(_v_, _x_)                                                    \
  NeuroFloat _v_ { float(_x_) }
constexpr NeuroFloat NeuroFloatZeros = 0.0;
constexpr NeuroFloat NeuroFloatOnes = 1.0;

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
      res[i] = nextDouble();
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
      res[i] = normalDouble(mean, stdDeviation);
    }
    return res;
#else
    return normalDouble(mean, stdDeviation);
#endif
  }

private:
#ifdef NDEBUG
  static inline thread_local std::random_device _rd{};
  static inline thread_local xorshift _gen{_rd};
#else
  static inline thread_local xorshift _gen{};
#endif
};

class Node;
class InputNode;
class HiddenNode;
class Weight;

using AnyNode = std::variant<InputNode, HiddenNode>;
using Group = std::vector<std::reference_wrapper<AnyNode>>;
} // namespace Nevolver

#ifdef NEVOLVER_WIDE
inline std::ostream &operator<<(std::ostream &os,
                                const Nevolver::NeuroFloat &f) {
  os << "[";
  for (int i = 0; i < Nevolver::NeuroFloatWidth; i++) {
    os << f[i] << " ";
  }
  os << "]";
  return os;
}
#endif

// Foundation
#include "connections.hpp"
#include "squash.hpp"

// Nodes
#include "node.hpp"
#include "nodes/annhidden.hpp"

#endif /* NEVOLVER_H */
