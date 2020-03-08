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
} // namespace Nevolver

#endif /* SQUASH_H */
