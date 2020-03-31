#ifndef SQUASH_H
#define SQUASH_H

#include "nevolver.hpp"

namespace Nevolver {
struct SquashBase {
  template <class Archive>
  void serialize(Archive &ar, std::uint32_t const version) {}
};

struct IdentityS final : public SquashBase {
  NeuroFloat operator()(const NeuroFloat &input) const { return input; }
};

struct IdentityD final : public SquashBase {
  NeuroFloat operator()(const NeuroFloat &state, const NeuroFloat &fwd) const {
    return 1.0;
  }
};

struct SigmoidS final : public SquashBase {
  NeuroFloat operator()(const NeuroFloat &input) const {
    return 1.0 / (1.0 + std::exp(-input));
  }
};

struct SigmoidD final : public SquashBase {
  NeuroFloat operator()(const NeuroFloat &state, const NeuroFloat &fwd) const {
    return fwd * (1.0 - fwd);
  }
};

struct TanhS final : public SquashBase {
  NeuroFloat operator()(const NeuroFloat &input) const {
    return std::tanh(input);
  }
};

struct TanhD final : public SquashBase {
  NeuroFloat operator()(const NeuroFloat &state, const NeuroFloat &fwd) const {
    return 1.0 - std::pow(std::tanh(state), 2);
  }
};

struct ReluS final : public SquashBase {
  NeuroFloat operator()(const NeuroFloat &input) const {
    return either(input > 0.0, input, 0.0);
  }
};

struct ReluD final : public SquashBase {
  NeuroFloat operator()(const NeuroFloat &state, const NeuroFloat &fwd) const {
    return either(state > 0.0, 1.0, 0.0);
  }
};

struct LeakyReluS final : public SquashBase {
  NeuroFloat operator()(const NeuroFloat &input) const {
    return either(input > 0.0, input, input / 20.0);
  }
};

struct LeakyReluD final : public SquashBase {
  NeuroFloat operator()(const NeuroFloat &state, const NeuroFloat &fwd) const {
    return either(state > 0.0, 1.0, 1.0 / 20.0);
  }
};

struct StepS final : public SquashBase {
  NeuroFloat operator()(const NeuroFloat &input) const {
    return either(input > 0.0, input, 0.0);
  }
};

struct StepD final : public SquashBase {
  NeuroFloat operator()(const NeuroFloat &state, const NeuroFloat &fwd) const {
    return 0.0;
  }
};

struct SoftsignS final : public SquashBase {
  NeuroFloat operator()(const NeuroFloat &input) const {
    return input / (1.0 + std::fabs(input));
  }
};

struct SoftsignD final : public SquashBase {
  NeuroFloat operator()(const NeuroFloat &state, const NeuroFloat &fwd) const {
    return 1.0 / std::pow(1.0 + std::fabs(state), 2.0);
  }
};

struct SinS final : public SquashBase {
  NeuroFloat operator()(const NeuroFloat &input) const {
    return std::sin(input);
  }
};

struct SinD final : public SquashBase {
  NeuroFloat operator()(const NeuroFloat &state, const NeuroFloat &fwd) const {
    return std::cos(state);
  }
};

struct GaussianS final : public SquashBase {
  NeuroFloat operator()(const NeuroFloat &input) const {
    return std::exp(-std::pow(input, 2.0));
  }
};

struct GaussianD final : public SquashBase {
  NeuroFloat operator()(const NeuroFloat &state, const NeuroFloat &fwd) const {
    return -2.0 * state * fwd;
  }
};

struct BentIdentityS final : public SquashBase {
  NeuroFloat operator()(const NeuroFloat &input) const {
    return (std::sqrt(std::pow(input, 2.0) + 1.0) - 1.0) / 2.0 + input;
  }
};

struct BentIdentityD final : public SquashBase {
  NeuroFloat operator()(const NeuroFloat &state, const NeuroFloat &fwd) const {
    return state / (2.0 * std::sqrt(std::pow(state, 2.0) + 1.0)) + 1.0;
  }
};

struct BipolarS final : public SquashBase {
  NeuroFloat operator()(const NeuroFloat &input) const {
    return either(input > 0.0, 1.0, -1.0);
  }
};

struct BipolarD final : public SquashBase {
  NeuroFloat operator()(const NeuroFloat &state, const NeuroFloat &fwd) const {
    return 0.0;
  }
};

struct BipolarSigmoidS final : public SquashBase {
  NeuroFloat operator()(const NeuroFloat &input) const {
    return 2.0 / (1.0 + std::exp(-input)) - 1.0;
  }
};

struct BipolarSigmoidD final : public SquashBase {
  NeuroFloat operator()(const NeuroFloat &state, const NeuroFloat &fwd) const {
    return 1.0 / 2.0 * (1.0 + fwd) * (1.0 - fwd);
  }
};

struct HardTanhS final : public SquashBase {
  NeuroFloat operator()(const NeuroFloat &input) const {
    return either(input < -1.0, -1.0, either(input > 1.0, 1.0, input));
  }
};

struct HardTanhD final : public SquashBase {
  NeuroFloat operator()(const NeuroFloat &state, const NeuroFloat &fwd) const {
    return either(state > -1.0 && state < 1.0, 1.0, 0.0);
  }
};

struct AbsoluteS final : public SquashBase {
  NeuroFloat operator()(const NeuroFloat &input) const {
    return std::fabs(input);
  }
};

struct AbsoluteD final : public SquashBase {
  NeuroFloat operator()(const NeuroFloat &state, const NeuroFloat &fwd) const {
    return either(state < 0.0, -1.0, 1.0);
  }
};

struct InverseS final : public SquashBase {
  NeuroFloat operator()(const NeuroFloat &input) const { return 1.0 - input; }
};

struct InverseD final : public SquashBase {
  NeuroFloat operator()(const NeuroFloat &state, const NeuroFloat &fwd) const {
    return -1.0;
  }
};

struct SeluBase : public SquashBase {
  constexpr static auto alpha = 1.6732632423543772848170429916717;
  constexpr static auto scale = 1.0507009873554804934193349852946;
};

struct SeluS final : public SeluBase {
  NeuroFloat operator()(const NeuroFloat &input) const {
    auto fx = either(input > 0.0, input, alpha * (std::exp(input) - 1.0));
    return fx * scale;
  }
};

struct SeluD final : public SeluBase {
  NeuroFloat operator()(const NeuroFloat &state, const NeuroFloat &fwd) const {
    return either(state > 0.0, scale, alpha * std::exp(state) * scale);
  }
};

using SquashFunc =
    std::variant<IdentityS, SigmoidS, TanhS, ReluS, LeakyReluS, StepS,
                 SoftsignS, SinS, GaussianS, BentIdentityS, BipolarS,
                 BipolarSigmoidS, HardTanhS, AbsoluteS, InverseS, SeluS>;
using DeriveFunc =
    std::variant<IdentityD, SigmoidD, TanhD, ReluD, LeakyReluD, StepD,
                 SoftsignD, SinD, GaussianD, BentIdentityD, BipolarD,
                 BipolarSigmoidD, HardTanhD, AbsoluteD, InverseD, SeluD>;

struct Squash final {
  const static inline std::array<SquashFunc, 16> funcs{
      IdentityS(),  SigmoidS(),      TanhS(),     ReluS(),
      LeakyReluS(), StepS(),         SoftsignS(), SinS(),
      GaussianS(),  BentIdentityS(), BipolarS(),  BipolarSigmoidS(),
      HardTanhS(),  AbsoluteS(),     InverseS(),  SeluS()};

  static const SquashFunc &random() {
    auto idx = Random::nextUInt() % funcs.size();
    assert(idx < funcs.size());
    return funcs[idx];
  }
};
} // namespace Nevolver

#endif /* SQUASH_H */
