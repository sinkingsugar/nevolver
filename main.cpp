#include "random.hpp"
#include <functional>
#include <iostream>
#include <list>
#include <memory>
#include <optional>
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

class Node;
class InputNode;
class HiddenNode;
class Weight;

using AnyNode = std::variant<InputNode, HiddenNode>;
using Group = std::vector<std::reference_wrapper<AnyNode>>;

class Random {
public:
  static NeuroFloat next() {
    return NeuroFloat(_gen()) * (1.0 / NeuroFloat(xorshift::max()));
  }

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
  static inline xorshift _gen{};
};

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

enum ConnectionPattern { AllToAll, AllToElse, OneToOne };

struct Connection final {
  const Node *from;
  const Node *to;
  const Node *gater;

  NeuroFloat gain{1.0};
  NeuroFloat eligibility{0.0};
  NeuroFloat previousDeltaWeight{0.0};
  NeuroFloat totalDeltaWeight{0.0};

  NeuroFloat *weight;
};

struct NodeConnections final {
  std::vector<const Connection *> inbound;
  std::vector<const Connection *> outbound;
  std::vector<const Connection *> gate;
  Connection *self = nullptr;
};

class Node {
public:
  virtual NeuroFloat activate() { return _activation; }

  NeuroFloat current() const { return _activation; }

  virtual bool is_output() const { return false; }

  virtual void addInboundConnection(const Connection &conn) {
    _connections.inbound.push_back(&conn);
  }
  virtual void addOutboundConnection(const Connection &conn) {
    _connections.outbound.push_back(&conn);
  }

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

    if (_connections.self) {
      _state =
          _connections.self->gain * *_connections.self->weight * _state * _bias;
    } else {
      _state = _bias;
    }

    for (auto &connection : _connections.inbound) {
      _state +=
          connection->from->current() * *connection->weight * connection->gain;
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
  NeuroFloat _bias{Random::normal(0.0, 1.0)};
  NeuroFloat _state{0.0};
  NeuroFloat _old{0.0};
  NeuroFloat _mask{1.0};
  NeuroFloat _derivative{0.0};
  NeuroFloat _previousDeltaBias{0.0};
  NeuroFloat _totalDeltaBias{0.0};
  bool _is_output;
};

class Network final {
public:
  static Network Perceptron(int inputs, std::vector<int> hidden, int outputs) {
    Network result{};

    auto total_size = inputs + outputs;
    for (auto lsize : hidden) {
      total_size += lsize;
    }

    result._nodes.reserve(total_size);

    Group inputNodes;
    for (int i = 0; i < inputs; i++) {
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

      result.connect(previous, layer, ConnectionPattern::AllToAll);
      previous = layer;
    }

    Group outputNodes;
    for (int i = 0; i < outputs; i++) {
      auto &node = result._nodes.emplace_back(HiddenNode(true));
      outputNodes.emplace_back(node);
    }

    result.connect(previous, outputNodes, ConnectionPattern::AllToAll);

    // finally setup weights now that we know how many we need
    result._weights.reserve(result._connections.size());
    for (auto &conn : result._connections) {
      auto &w = result._weights.emplace_back(Random::normal(0.0, 1.0));
      conn.weight = &w;
    }

    return result;
  }

  void connect(AnyNode &from, AnyNode &to) {
    Node *fromPtr;
    std::visit([&fromPtr](auto &&node) { fromPtr = &node; }, from);
    Node *toPtr;
    std::visit([&toPtr](auto &&node) { toPtr = &node; }, to);

    auto &conn = _connections.emplace_back(Connection{fromPtr, toPtr, nullptr});

    std::visit([&conn](auto &&node) { node.addOutboundConnection(conn); },
               from);
    std::visit([&conn](auto &&node) { node.addInboundConnection(conn); }, to);
  }

  void connect(const Group &from, const Group &to, ConnectionPattern pattern) {
    switch (pattern) {
    case AllToAll: {
      for (auto &fromNode : from) {
        for (auto &toNode : to) {
          connect(fromNode, toNode);
        }
      }
    } break;
    case AllToElse: {

    } break;
    case OneToOne: {

    } break;
    };
  }

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
  std::list<Connection> _connections;
  std::vector<NeuroFloat> _weights;
};
} // namespace Nevolver

int main() {
  std::cout << "Hello!\n";

  Nevolver::HiddenNode node;
  std::cout << node.activate() << "\n";

  auto perceptron = Nevolver::Network::Perceptron(2, {4}, 1);
  auto prediction = perceptron.activate({1.0, 0.0});
  std::cout << prediction[0] << "\n";

  return 0;
}
