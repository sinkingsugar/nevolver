#include <cmath>
#include <functional>
#include <iostream>
#include <memory>
#include <vector>

namespace Nevolver {

using NeuroFloat = float;

class Node;
class Weight;

struct Squash {
  virtual NeuroFloat operator()(NeuroFloat input) const { return input; }
};

struct Derivative {
  virtual NeuroFloat operator()(NeuroFloat state, NeuroFloat fwd) const {
    return fwd;
  }
};

struct SigmoidS final : public Squash {
  NeuroFloat operator()(NeuroFloat input) const override {
    return 1.0 / (1.0 + std::exp(-input));
  }
};

struct SigmoidD final : public Derivative {
  NeuroFloat operator()(NeuroFloat state, NeuroFloat fwd) const override {
    return fwd * (1 - fwd);
  }
};

struct Weight final {
  NeuroFloat value;
};

struct Connection final {
  Node *from = nullptr;
  Node *to = nullptr;
  Node *gater = nullptr;

  NeuroFloat gain{1.0};
  NeuroFloat eligibility{0.0};
  NeuroFloat previousDeltaWeight{0.0};
  NeuroFloat totalDeltaWeight{0.0};

  std::shared_ptr<Weight> weight;
};

struct NodeConnections final {
  std::vector<Connection *> inbound;
  std::vector<Connection *> outbound;
  std::vector<Connection *> gate;
  Connection self;
};

class Node {
public:
  Node() {
    connections.self.from = this;
    connections.self.to = this;
    connections.self.weight = std::make_shared<Weight>(Weight{0.0});
  }

  virtual NeuroFloat activate() { return activation; }

  NeuroFloat current() const { return activation; }

protected:
  NeuroFloat activation{0.0};
  NodeConnections connections;
};

class InputNode final : public Node {
public:
  void operator()(NeuroFloat input) { activation = input; }
};

class HiddenNode final : public Node {
public:
  HiddenNode() : squash(SigmoidS()), derive(SigmoidD()) {}

  virtual NeuroFloat activate() {
    old = state;
    state =
        connections.self.gain * connections.self.weight->value * state * bias;

    for (auto &connection : connections.inbound) {
      state += connection->from->current() * connection->weight->value *
               connection->gain;
    }

    auto fwd = squash(state);
    activation = fwd * mask;
    derivative = derive(state, fwd);

    return activation;
  }

private:
  std::function<NeuroFloat(NeuroFloat)> squash;
  std::function<NeuroFloat(NeuroFloat, NeuroFloat)> derive;
  NeuroFloat bias;
  NeuroFloat state;
  NeuroFloat old;
  NeuroFloat mask;
  NeuroFloat derivative;
  NeuroFloat previousDeltaBias;
  NeuroFloat totalDeltaBias;
};
} // namespace Nevolver

int main() {
  std::cout << "Hello!\n";

  Nevolver::HiddenNode node;
  std::cout << node.activate();

  return 0;
}
