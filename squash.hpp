#ifndef SQUASH_H
#define SQUASH_H

#include "nevolver.hpp"

namespace Nevolver {
struct IdentityS final {
  NeuroFloat operator()(NeuroFloat input) const { return input; }
};

struct IdentityD final {
  NeuroFloat operator()(NeuroFloat state, NeuroFloat fwd) const { return 1; }
};

struct SigmoidS final {
  NeuroFloat operator()(NeuroFloat input) const {
    return 1.0 / (1.0 + std::exp(-input));
  }
};

struct SigmoidD final {
  NeuroFloat operator()(NeuroFloat state, NeuroFloat fwd) const {
    return fwd * (1 - fwd);
  }
};

struct Squash {
  static inline std::array<std::function<NeuroFloat(NeuroFloat)>, 2> All{
      IdentityS(), SigmoidS()};

  static std::function<NeuroFloat(NeuroFloat)> &random() {
    auto idx = size_t(Random::nextDouble() * (1.0 / double(All.size())));
    assert(idx < All.size());
    return All[idx];
  }
};
} // namespace Nevolver

#endif /* SQUASH_H */
