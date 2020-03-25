#ifndef SQUASH_H
#define SQUASH_H

#include "nevolver.hpp"

namespace Nevolver {
struct IdentityS final {
  NeuroFloat operator()(const NeuroFloat &input) const { return input; }
};

struct IdentityD final {
  NeuroFloat operator()(const NeuroFloat &state, const NeuroFloat &fwd) const {
    return NeuroFloatOnes;
  }
};

struct SigmoidS final {
  NeuroFloat operator()(const NeuroFloat &input) const {
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
  NeuroFloat operator()(const NeuroFloat &state, const NeuroFloat &fwd) const {
    return fwd * (1.0 - fwd);
  }
};

using SquashFunc = std::variant<IdentityS, SigmoidS>;
using DeriveFunc = std::variant<IdentityD, SigmoidD>;

struct Squash final {
  static inline std::array<SquashFunc, 2> funcs{IdentityS(), SigmoidS()};

  static SquashFunc &random() {
    auto idx = size_t(Random::nextDouble() * (1.0 / double(funcs.size())));
    assert(idx < funcs.size());
    return funcs[idx];
  }
};
} // namespace Nevolver

#endif /* SQUASH_H */
