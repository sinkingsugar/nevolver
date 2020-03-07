#include <cmath>
#include <functional>
#include <iostream>
#include <memory>
#include <variant>
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
    _connections.self.from = this;
    _connections.self.to = this;
    _connections.self.weight = std::make_shared<Weight>(Weight{0.0});
  }

  virtual NeuroFloat activate() { return _activation; }

  NeuroFloat current() const { return _activation; }

  virtual bool is_output() const { return false; }

protected:
  NeuroFloat _activation{0.0};
  NodeConnections _connections;
};

class InputNode final : public Node {
public:
  void setInput(NeuroFloat input) { _activation = input; }
};

class HiddenNode final : public Node {
public:
  HiddenNode(bool is_output = false) : _is_output(is_output) {}

  NeuroFloat activate() override {
    _old = _state;
    _state = _connections.self.gain * _connections.self.weight->value * _state *
             _bias;

    for (auto &connection : _connections.inbound) {
      _state += connection->from->current() * connection->weight->value *
                connection->gain;
    }

    auto fwd = _squash(_state);
    _activation = fwd * _mask;
    _derivative = _derive(_state, fwd);

    return _activation;
  }

  bool is_output() const override { return _is_output; };

private:
  std::function<NeuroFloat(NeuroFloat)> _squash{SigmoidS()};
  std::function<NeuroFloat(NeuroFloat, NeuroFloat)> _derive{SigmoidD()};
  NeuroFloat _bias{0.0};
  NeuroFloat _state{0.0};
  NeuroFloat _old{0.0};
  NeuroFloat _mask{0.0};
  NeuroFloat _derivative{0.0};
  NeuroFloat _previousDeltaBias{0.0};
  NeuroFloat _totalDeltaBias{0.0};
  bool _is_output;
};

using AnyNode = std::variant<InputNode, HiddenNode>;

using Group = std::vector<std::reference_wrapper<AnyNode>>;

class Network final {
public:
  static Network Perceptron(int inputs, std::vector<int> hidden, int outputs) {
    Network result{};

    Group inputNodes;
    for (int i = 0; i < outputs; i++) {
      auto &node = result._nodes.emplace_back(InputNode());
      result._inputs.emplace_back(std::get<InputNode>(node));
      inputNodes.emplace_back(node);
    }

    auto &previous = inputNodes;

    std::vector<Group> layers;
    for (auto lsize : hidden) {
      auto layer = layers.emplace_back();
      for (int i = 0; i < lsize; i++) {
        auto &node = result._nodes.emplace_back(HiddenNode());
        layer.emplace_back(node);
      }

      result.connect(previous, layer);
      previous = layer;
    }

    Group outputNodes;
    for (int i = 0; i < outputs; i++) {
      auto &node = result._nodes.emplace_back(HiddenNode(true));
      outputNodes.emplace_back(node);
    }

    result.connect(previous, outputNodes);

    return result;
  }

  void connect(AnyNode &from, AnyNode &to) {}

  void connect(const Group &from, const Group &to) {}

  const std::vector<NeuroFloat> &
  activate(const std::vector<NeuroFloat> &input) {
    _outputCache.clear();

    auto isize = input.size();

    if (isize != _inputs.size())
      throw std::runtime_error(
          "Invalid activation input size, differs from actual "
          "network input size.");

    for (size_t i = 0; i < isize; i++) {
      _inputs[i].get().setInput(input[i]);
    }

    for (auto &vnode : _nodes) {
      std::visit(
          [this](auto &&node) {
            auto activation = node.activate();
            if (node.is_output())
              _outputCache.push_back(activation);
          },
          vnode);
    }

    return _outputCache;
  }

private:
  std::vector<NeuroFloat> _outputCache;
  std::vector<std::reference_wrapper<InputNode>> _inputs;
  std::vector<AnyNode> _nodes;
  std::vector<Connection> _connections;
};
} // namespace Nevolver

int main() {
  std::cout << "Hello!\n";

  Nevolver::HiddenNode node;
  std::cout << node.activate();

  auto perceptron = Nevolver::Network::Perceptron(2, {4}, 1);
  auto prediction = perceptron.activate({1.0, 0.0});
  std::cout << prediction[0] << "\n";

  return 0;
}
