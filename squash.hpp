#ifndef SQUASH_H
#define SQUASH_H

#include "nevolver.hpp"

namespace Nevolver {
struct IdentityS final {
  NeuroFloat operator()(NeuroFloat input) const { return input; }
};

struct IdentityD final {
  NeuroFloat operator()(NeuroFloat state, NeuroFloat fwd) const {
    return NeuroFloatOnes;
  }
};

struct SigmoidS final {
  NeuroFloat operator()(NeuroFloat input) const {
#ifdef NEVOLVER_WIDE
    NeuroFloat res;
    for (int i = 0; i < NeuroFloatWidth; i++) {
      res[i] = 1.0 / (1.0 + __builtin_exp(-input[i]));
    }
    return res;
#else
    return 1.0 / (1.0 + std::exp(-input));
#endif
  }
};

struct SigmoidD final {
  NeuroFloat operator()(NeuroFloat state, NeuroFloat fwd) const {
    return fwd * (1.0 - fwd);
  }
};

struct Squash final {
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
