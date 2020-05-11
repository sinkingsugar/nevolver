#ifndef NEVOLVER_H
#define NEVOLVER_H

// #define NEVOLVER_WIDE 4

#include <cassert>
#include <deque>
#include <iostream>
#include <limits>
#include <map>
#include <ostream>
#include <random>
#include <unordered_set>
#include <variant>
#include <vector>

#define _USE_MATH_DEFINES
#include <cmath>

#include <cereal/archives/binary.hpp>
#include <cereal/archives/json.hpp>
#include <cereal/cereal.hpp>
#include <cereal/types/array.hpp>
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
  static double nextDouble() { return _udis(_gen); }

  static NeuroFloat next() {
#ifdef NEVOLVER_WIDE
    NeuroFloat res;
    for (auto i = 0; i < NeuroFloat::Width; i++) {
      res.vec[i] = _udis(_gen);
    }
    return res;
#else
    return _udis(_gen);
#endif
  }

  static uint32_t nextUInt() { return _uintdis(_gen); }

  // our weight/bias init
  static NeuroFloat init() { return next() * 0.2 - 0.1; }

  // our mutation adjust for weights
  static NeuroFloat adjust() {
#ifdef NEVOLVER_WIDE
    NeuroFloat res;
    for (auto i = 0; i < NeuroFloat::Width; i++) {
      res.vec[i] = _ndis(_gen);
    }
    return res;
#else
    return _ndis(_gen);
#endif
  }

private:
  static inline thread_local std::random_device _rd{};
  static inline thread_local std::mt19937 _gen{_rd()};
  static inline thread_local std::uniform_int_distribution<> _uintdis{};
  static inline thread_local std::uniform_real_distribution<> _udis{0.0, 1.0};
  static inline thread_local std::normal_distribution<> _ndis{0.0, 0.1};
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
