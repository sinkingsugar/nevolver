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

struct ConnectionXTraces {
  std::vector<const Node *> nodes;
  std::vector<NeuroFloat> values;
};

struct Connection final {
  const Node *from;
  const Node *to;
  const Node *gater;

  NeuroFloat gain{1.0};
  NeuroFloat eligibility{0.0};
  NeuroFloat previousDeltaWeight{0.0};

  NeuroFloat *weight;

  ConnectionXTraces xtraces;
};

struct NodeConnections final {
  std::vector<Connection *> inbound;
  std::vector<Connection *> outbound;
  std::vector<Connection *> gate;
  Connection *self = nullptr;
};

class Node {
public:
  NeuroFloat current() const { return _activation; }

  void addInboundConnection(Connection &conn) {
    _connections.inbound.push_back(&conn);
  }
  void addOutboundConnection(Connection &conn) {
    _connections.outbound.push_back(&conn);
  }

  const NodeConnections &connections() const { return _connections; }

  NeuroFloat responsibility() const { return _responsibility; }

protected:
  NeuroFloat _activation{0.0};
  NeuroFloat _responsibility{0.0};
  NodeConnections _connections;
};

template <typename T> class NodeCommon : public Node {
public:
  NeuroFloat activate() { return as_underlying().doActivate(); }

  NeuroFloat activateFast() { return as_underlying().doFastActivate(); }

  void propagate(NeuroFloat rate, NeuroFloat momentum, bool update,
                 NeuroFloat target = 0.0) {
    return as_underlying().doPropagate(rate, momentum, update, target);
  }

  bool is_output() const { return as_underlying().getIsOutput(); }

protected:
  friend T;

private:
  inline T &as_underlying() { return static_cast<T &>(*this); }
  inline T const &as_underlying() const {
    return static_cast<T const &>(*this);
  }
};

class InputNode final : public NodeCommon<InputNode> {
public:
  void setInput(NeuroFloat input) { _activation = input; }

  NeuroFloat doActivate() { return _activation; }

  NeuroFloat doFastActivate() { return _activation; }

  void doPropagate(NeuroFloat rate, NeuroFloat momentum, bool update,
                   NeuroFloat target) {}

  bool getIsOutput() const { return false; }
};

class HiddenNode final : public NodeCommon<HiddenNode> {
public:
  HiddenNode(bool is_output = false, bool is_constant = false)
      : _is_output(is_output), _is_constant(is_constant) {}

  bool getIsOutput() const { return _is_output; }

  NeuroFloat doActivate() {
    _old = _state;

    if (_connections.self) {
      _state =
          _connections.self->gain * *_connections.self->weight * _state * _bias;
    } else {
      _state = _bias;
    }

    for (auto connection : _connections.inbound) {
      _state +=
          connection->from->current() * *connection->weight * connection->gain;
    }

    auto fwd = _squash(_state);
    _activation = fwd * _mask;
    _derivative = _derive(_state, fwd);

    _tmpNodes.clear();
    _tmpInfluence.clear();
    for (auto connection : _connections.gate) {
      auto node = connection->to;
      auto pos = std::find(std::begin(_tmpNodes), std::end(_tmpNodes), node);
      if (pos != std::end(_tmpNodes)) {
        auto idx = std::distance(std::begin(_tmpNodes), pos);
        _tmpInfluence[idx] += *connection->weight * connection->from->current();
      } else {
        _tmpNodes.emplace_back(node);
        auto plus =
            node->connections().self && node->connections().self->gater == this
                ? static_cast<const HiddenNode *>(node)->_old
                : 0.0;
        _tmpInfluence.emplace_back(
            *connection->weight * connection->from->current() + plus);
      }
      connection->gain = _activation;
    }

    for (auto connection : _connections.inbound) {
      if (_connections.self) {
        connection->eligibility =
            _connections.self->gain * *_connections.self->weight *
                connection->eligibility +
            connection->from->current() * connection->gain;
      } else {
        connection->eligibility =
            connection->from->current() * connection->gain;
      }

      auto size = _tmpNodes.size();
      for (size_t i = 0; i < size; i++) {
        auto node = _tmpNodes[i];
        auto influence = _tmpInfluence[i];
        auto pos = std::find(std::begin(connection->xtraces.nodes),
                             std::end(connection->xtraces.nodes), node);
        if (pos != std::end(connection->xtraces.nodes)) {
          auto idx = std::distance(std::begin(connection->xtraces.nodes), pos);
          if (node->connections().self) {
            connection->xtraces.values[idx] =
                node->connections().self->gain *
                    *node->connections().self->weight *
                    connection->xtraces.values[idx] +
                _derivative * connection->eligibility * influence;
          } else {
            connection->xtraces.values[idx] =
                _derivative * connection->eligibility * influence;
          }
        } else {
          connection->xtraces.nodes.emplace_back(node);
          connection->xtraces.values.emplace_back(
              _derivative * connection->eligibility * influence);
        }
      }
    }

    return _activation;
  }

  NeuroFloat doFastActivate() {
    _old = _state;

    if (_connections.self) {
      _state =
          _connections.self->gain * *_connections.self->weight * _state * _bias;
    } else {
      _state = _bias;
    }

    for (auto connection : _connections.inbound) {
      _state +=
          connection->from->current() * *connection->weight * connection->gain;
    }

    auto fwd = _squash(_state);
    _activation = fwd * _mask;

    for (auto connection : _connections.gate) {
      connection->gain = _activation;
    }

    return _activation;
  }

  void doPropagate(NeuroFloat rate, NeuroFloat momentum, bool update,
                   NeuroFloat target) {
    if (_is_output) {
      _responsibility = target - _activation;
      _projected = _responsibility;
    } else {
      NeuroFloat error = 0.0;

      for (auto connection : _connections.outbound) {
        error += connection->to->responsibility() * *connection->weight *
                 connection->gain;
      }
      _projected = _derivative * error;

      error = 0.0;

      for (auto connection : _connections.gate) {
        auto node = connection->to;
        NeuroFloat influence =
            node->connections().self && node->connections().self->gater == this
                ? static_cast<const HiddenNode *>(node)->_old
                : 0.0;
        influence += *connection->weight * connection->from->current();
        error += connection->to->responsibility() * influence;
      }

      _gated = _derivative * error;
      _responsibility = _projected + _gated;
    }

    if (_is_constant)
      return;

    for (auto connection : _connections.inbound) {
      auto gradient = _projected * connection->eligibility;

      // Gated nets only
      size_t size = _tmpNodes.size();
      for (size_t i = 0; i < size; i++) {
        auto node = _tmpNodes[i];
        auto value = _tmpInfluence[i];

        gradient += node->responsibility() * value;
      }

      auto deltaWeight = rate * gradient * _mask;
      if (update) {
        deltaWeight += momentum * connection->previousDeltaWeight;
        *connection->weight += deltaWeight;
        connection->previousDeltaWeight = deltaWeight;
      }
    }

    auto deltaBias = rate * _responsibility;
    deltaBias += momentum * _previousDeltaBias;
    _bias += deltaBias;
    _previousDeltaBias = deltaBias;
  }

private:
  std::function<NeuroFloat(NeuroFloat)> _squash{SigmoidS()};
  std::function<NeuroFloat(NeuroFloat, NeuroFloat)> _derive{SigmoidD()};
  NeuroFloat _bias{Random::normal(0.0, 1.0)};
  NeuroFloat _state{0.0};
  NeuroFloat _old{0.0};
  NeuroFloat _mask{1.0};
  NeuroFloat _derivative{0.0};
  NeuroFloat _previousDeltaBias{0.0};
  bool _is_output;
  bool _is_constant;
  std::vector<const Node *> _tmpNodes;
  std::vector<NeuroFloat> _tmpInfluence;
  NeuroFloat _projected;
  NeuroFloat _gated;
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

  void connect(const Group &from, AnyNode &to) {
    for (auto &fromNode : from) {
      connect(fromNode, to);
    }
  }

  void connect(AnyNode &from, const Group &to) {
    for (auto &toNode : to) {
      connect(from, toNode);
    }
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
      for (auto &fromNode : from) {
        for (auto &toNode : to) {
          if (&from == &to)
            continue;
          connect(fromNode, toNode);
        }
      }
    } break;
    case OneToOne: {
      auto fsize = from.size();
      if (fsize != to.size()) {
        throw std::runtime_error(
            "Connect OneToOne requires 2 node groups with the same size.");
      }
      for (size_t i = 0; i < fsize; i++) {
        connect(from[i], to[i]);
      }
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

  const std::vector<NeuroFloat> &
  activateFast(const std::vector<NeuroFloat> &input) {
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
            auto activation = node.activateFast();
            if (node.is_output())
              _outputCache.push_back(activation);
          },
          vnode);
    }

    return _outputCache;
  }

  NeuroFloat propagate(const std::vector<NeuroFloat> &targets,
                       NeuroFloat rate = 0.3, NeuroFloat momentum = 0.0,
                       bool update = true) {
    size_t outputIdx = targets.size();
    _outputCache.resize(outputIdx); // reuse for MSE
    for (auto it = _nodes.rbegin(); it != _nodes.rend(); ++it) {
      std::visit(
          [&](auto &&node) {
            if (node.is_output()) {
              outputIdx--;
              node.propagate(rate, momentum, update, targets[outputIdx]);
              _outputCache[outputIdx] =
                  std::pow(node.current() - targets[outputIdx], 2);
            } else {
              node.propagate(rate, momentum, update);
            }
          },
          *it);
    }
    NeuroFloat mean = 0.0;
    for (auto err : _outputCache) {
      mean += err;
    }
    mean /= NeuroFloat(_outputCache.size());
    return mean;
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
  for (auto i = 0; i < 250; i++) {
    auto prediction = perceptron.activate({1.0, 0.0});
    std::cout << "Prediction: " << prediction[0] << "\n";
    std::cout << "MSE: " << perceptron.propagate({1.0}) << "\n";
  }

  return 0;
}
