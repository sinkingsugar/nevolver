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

namespace Nevolver {
using NeuroFloat = float;

class Random {
public:
  static NeuroFloat next() {
    return NeuroFloat(_gen()) * (1.0 / NeuroFloat(xorshift::max()));
  }

  static double nextDouble() {
    return double(_gen()) * (1.0 / double(xorshift::max()));
  }

  static uint32_t nextUInt() { return _gen(); }

  static NeuroFloat normal(NeuroFloat mean, NeuroFloat stdDeviation) {
    NeuroFloat u1 = 0.0;
    while (u1 == 0.0) {
      u1 = next();
    }

    auto u2 = next();
    auto rstdNorm = std::sqrt(-2.0 * std::log(u1)) * std::sin(2.0 * M_PI * u2);

    return mean + stdDeviation * rstdNorm;
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
