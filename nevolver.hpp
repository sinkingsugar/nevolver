#ifndef NEVOLVER_H
#define NEVOLVER_H

// #define NEVOLVER_WIDE

#include "random.hpp"
#include <cassert>
#include <deque>
#include <iostream>
#include <limits>
#include <map>
#include <ostream>
#include <unordered_set>
#include <variant>
#include <vector>

#define _USE_MATH_DEFINES
#include <cmath>

#include <cereal/archives/binary.hpp>
#include <cereal/archives/json.hpp>
#include <cereal/cereal.hpp>
#include <cereal/types/functional.hpp>
#include <cereal/types/variant.hpp>
#include <cereal/types/vector.hpp>
#include <easylogging++.h>

#include "neurofloat.hpp"

#ifndef M_PI
#define M_PI (3.14159265358979323846)
#endif

#ifndef M_PIl
#define M_PIl (3.14159265358979323846264338327950288)
#endif

#define NEVOLVER_VERSION 0x1

namespace Nevolver {
class Random {
public:
  static double nextDouble() {
    return double(_gen()) * (1.0 / double(xorshift::max()));
  }

  static NeuroFloat next() {
#ifdef NEVOLVER_WIDE
    NeuroFloat res;
    for (auto i = 0; i < NeuroFloat::Width; i++) {
      res.vec[i] = nextDouble();
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

    return normalDouble(mean, stdDeviation);
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
struct Connection;

using AnyNode = std::variant<InputNode, HiddenNode>;
using Group = std::vector<std::reference_wrapper<AnyNode>>;
using Weight = std::pair<NeuroFloat, std::unordered_set<const Connection *>>;
} // namespace Nevolver

// Foundation
#include "connections.hpp"
#include "squash.hpp"

// Nodes
#include "node.hpp"
#include "nodes/annhidden.hpp"

#endif /* NEVOLVER_H */
