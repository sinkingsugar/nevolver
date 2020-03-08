#include "nevolver.hpp"
#include "squash.hpp"

namespace Nevolver {

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
#ifdef NDEBUG
  static inline std::random_device _rd{};
  static inline xorshift _gen{_rd};
#else
  static inline xorshift _gen{};
#endif
};

enum ConnectionPattern { AllToAll, AllToElse, OneToOne };
enum GatingPattern { Input, Output, Self };

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

  void addInboundConnection(Connection &conn) {
    _connections.inbound.push_back(&conn);
  }
  void addOutboundConnection(Connection &conn) {
    _connections.outbound.push_back(&conn);
  }

  void addGate(Connection &conn) { _connections.gate.push_back(&conn); }

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

  void setSquash(std::function<NeuroFloat(NeuroFloat)> squash,
                 std::function<NeuroFloat(NeuroFloat, NeuroFloat)> derive) {
    _squash = squash;
    _derive = derive;
  }

  void setBias(NeuroFloat bias) { _bias = bias; }

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

class Network {
public:
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

protected:
  Connection &connect(AnyNode &from, AnyNode &to) {
    Node *fromPtr;
    std::visit([&fromPtr](auto &&node) { fromPtr = &node; }, from);
    Node *toPtr;
    std::visit([&toPtr](auto &&node) { toPtr = &node; }, to);

    auto &conn = _connections.emplace_back(Connection{fromPtr, toPtr, nullptr});

    std::visit([&conn](auto &&node) { node.addOutboundConnection(conn); },
               from);
    std::visit([&conn](auto &&node) { node.addInboundConnection(conn); }, to);

    return conn;
  }

  std::vector<std::reference_wrapper<Connection>> connect(const Group &from,
                                                          AnyNode &to) {
    std::vector<std::reference_wrapper<Connection>> conns;
    for (auto &fromNode : from) {
      conns.emplace_back(connect(fromNode, to));
    }
    return conns;
  }

  std::vector<std::reference_wrapper<Connection>> connect(AnyNode &from,
                                                          const Group &to) {
    std::vector<std::reference_wrapper<Connection>> conns;
    for (auto &toNode : to) {
      conns.emplace_back(connect(from, toNode));
    }
    return conns;
  }

  std::vector<std::reference_wrapper<Connection>>
  connect(const Group &from, const Group &to, ConnectionPattern pattern) {
    std::vector<std::reference_wrapper<Connection>> conns;
    switch (pattern) {
    case AllToAll: {
      for (auto &fromNode : from) {
        for (auto &toNode : to) {
          conns.emplace_back(connect(fromNode, toNode));
        }
      }
    } break;
    case AllToElse: {
      for (auto &fromNode : from) {
        for (auto &toNode : to) {
          if (&from == &to)
            continue;
          conns.emplace_back(connect(fromNode, toNode));
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
        conns.emplace_back(connect(from[i], to[i]));
      }
    } break;
    };
    return conns;
  }

  Group addMemoryCell(int size) {
    Group res;
    for (int y = 0; y < size; y++) {
      auto &node = _nodes.emplace_back(HiddenNode(false, true));
      auto &hiddenNode = std::get<HiddenNode>(node);
      hiddenNode.setBias(0.0);
      hiddenNode.setSquash(IdentityS(), IdentityD());
      res.emplace_back(node);
    }
    return res;
  }

  void gate(AnyNode &gater, Connection &conn) {
    Node *gaterPtr;
    std::visit([&gaterPtr](auto &&node) { gaterPtr = &node; }, gater);
    conn.gater = gaterPtr;
    std::visit([&conn](auto &&node) { node.addGate(conn); }, gater);
  }

  void gate(AnyNode &gater,
            std::vector<std::reference_wrapper<Connection>> connections) {
    for (auto &conn : connections) {
      gate(gater, conn);
    }
  }

  void gate(const Group &group,
            std::vector<std::reference_wrapper<Connection>> connections,
            GatingPattern pattern) {
    std::set<const Node *> nodesFrom;
    std::set<const Node *> nodesTo;
    for (auto &conn : connections) {
      nodesFrom.insert(conn.get().from);
      nodesTo.insert(conn.get().to);
    }

    auto gsize = group.size();
    switch (pattern) {
    case GatingPattern::Input: {
      assert(gsize == nodesTo.size());
      size_t idx = 0;
      for (auto node : nodesTo) {
        auto &gater = group[idx++];
        for (auto &conn : node->connections().inbound) {
          gate(gater, *conn);
        }
      }
    } break;
    case GatingPattern::Output: {
      assert(gsize == nodesFrom.size());
      size_t idx = 0;
      for (auto node : nodesFrom) {
        auto &gater = group[idx++];
        for (auto &conn : node->connections().outbound) {
          gate(gater, *conn);
        }
      }
    } break;
    case GatingPattern::Self: {
      size_t idx = 0;
      for (auto node : nodesFrom) {
        auto &gater = group[idx++];
        idx = idx % gsize;
        for (auto &conn : connections) {
          auto connPtr = &conn.get();
          if (connPtr == node->connections().self) {
            gate(gater, conn);
          }
        }
      }
    } break;
    }
  }

  std::vector<std::reference_wrapper<InputNode>> _inputs;
  std::list<Connection> _connections;
  std::vector<AnyNode> _nodes;
  std::vector<NeuroFloat> _weights;

private:
  std::vector<NeuroFloat> _outputCache;
};

class NARX final : public Network {
public:
  NARX(int inputs, std::vector<int> hidden, int outputs, int input_memory,
       int output_memory) {
    // Need to pre allocate
    // to avoid vector reallocations, we need valid ptr/refs
    auto total_size =
        inputs + (inputs * input_memory) + outputs + (outputs * output_memory);
    for (auto lsize : hidden) {
      total_size += lsize;
    }
    _nodes.reserve(total_size);

    // insertion order = activation order = MATTERS

    // keep track of pure ringbuffer memory conns we need to fix in terms of
    // weights
    std::vector<std::reference_wrapper<Connection>> memoryTunnels;

    Group inputNodes;
    for (int i = 0; i < inputs; i++) {
      auto &node = _nodes.emplace_back(InputNode());
      _inputs.emplace_back(std::get<InputNode>(node));
      inputNodes.emplace_back(node);
    }

    // TODO really make creating stuff easier :)
    // but for now... we need to reverse those nodes activations
    // so we store their position in the array
    std::vector<Group> outputMemory;
    outputMemory.resize(output_memory);
    for (int i = output_memory; i > 0; i--) {
      auto memStart = _nodes.end();
      outputMemory[i - 1] = addMemoryCell(outputs);
      auto memEnd = _nodes.end();
      std::reverse(memStart, memEnd);
    }

    Group *previous = &outputMemory[0];
    for (int i = 1; i < output_memory; i++) {
      auto conns =
          connect(*previous, outputMemory[i], ConnectionPattern::OneToOne);
      memoryTunnels.insert(memoryTunnels.end(), conns.begin(), conns.end());
      previous = &outputMemory[i];
    }

    previous = &inputNodes;

    std::vector<Group> layers;
    // must pre alloc as we take ref!
    layers.reserve(hidden.size());
    for (auto lsize : hidden) {
      auto &layer = layers.emplace_back();
      for (int i = 0; i < lsize; i++) {
        auto &node = _nodes.emplace_back(HiddenNode());
        layer.emplace_back(node);
      }

      connect(*previous, layer, ConnectionPattern::AllToAll);
      previous = &layer;
    }

    std::vector<Group> inputMemory;
    inputMemory.resize(input_memory);
    for (int i = input_memory; i > 0; i--) {
      auto memStart = _nodes.end();
      inputMemory[i - 1] = addMemoryCell(inputs);
      auto memEnd = _nodes.end();
      std::reverse(memStart, memEnd);
    }

    previous = &inputMemory[0];
    for (int i = 1; i < input_memory; i++) {
      auto conns =
          connect(*previous, inputMemory[i], ConnectionPattern::OneToOne);
      memoryTunnels.insert(memoryTunnels.end(), conns.begin(), conns.end());
      previous = &inputMemory[i];
    }

    Group outputNodes;
    for (int i = 0; i < outputs; i++) {
      auto &node = _nodes.emplace_back(HiddenNode(true));
      outputNodes.emplace_back(node);
    }

    connect(*previous, outputNodes, ConnectionPattern::AllToAll);

    std::reverse(inputMemory.begin(), inputMemory.end());
    connect(inputNodes, inputMemory.back(), ConnectionPattern::OneToOne);
    for (auto &group : inputMemory) {
      connect(group, layers[0], ConnectionPattern::AllToAll);
    }
    std::reverse(outputMemory.begin(), outputMemory.end());
    connect(outputNodes, outputMemory.back(), ConnectionPattern::OneToOne);
    for (auto &group : outputMemory) {
      connect(group, layers[0], ConnectionPattern::AllToAll);
    }

    // finally setup weights now that we know how many we need
    _weights.reserve(_connections.size());
    for (auto &conn : _connections) {
      auto &w = _weights.emplace_back(Random::normal(0.0, 1.0));
      conn.weight = &w;
    }

    // Fix up memory weights
    for (auto &conn : memoryTunnels) {
      *conn.get().weight = 1.0;
    }
  }
};

class Perceptron final : public Network {
public:
  Perceptron(int inputs, std::vector<int> hidden, int outputs) {
    auto total_size = inputs + outputs;
    for (auto lsize : hidden) {
      total_size += lsize;
    }
    _nodes.reserve(total_size);

    Group inputNodes;
    for (int i = 0; i < inputs; i++) {
      auto &node = _nodes.emplace_back(InputNode());
      _inputs.emplace_back(std::get<InputNode>(node));
      inputNodes.emplace_back(node);
    }

    Group *previous = &inputNodes;

    std::vector<Group> layers;
    // must pre alloc as we take ref!
    layers.reserve(hidden.size());
    for (auto lsize : hidden) {
      auto &layer = layers.emplace_back();
      for (int i = 0; i < lsize; i++) {
        auto &node = _nodes.emplace_back(HiddenNode());
        layer.emplace_back(node);
      }

      connect(*previous, layer, ConnectionPattern::AllToAll);
      previous = &layer;
    }

    Group outputNodes;
    for (int i = 0; i < outputs; i++) {
      auto &node = _nodes.emplace_back(HiddenNode(true));
      outputNodes.emplace_back(node);
    }

    connect(*previous, outputNodes, ConnectionPattern::AllToAll);

    // finally setup weights now that we know how many we need
    _weights.reserve(_connections.size());
    for (auto &conn : _connections) {
      auto &w = _weights.emplace_back(Random::normal(0.0, 1.0));
      conn.weight = &w;
    }
  }
};
} // namespace Nevolver

int main() {
  std::cout << "Hello!\n";

  Nevolver::HiddenNode node;
  std::cout << node.activate() << "\n";

  {
    auto perceptron = Nevolver::Perceptron(2, {4, 4}, 1);
    for (auto i = 0; i < 50000; i++) {
      perceptron.activate({0.0, 0.0});
      perceptron.propagate({1.0});
      perceptron.activate({0.0, 1.0});
      perceptron.propagate({0.0});
      perceptron.activate({1.0, 0.0});
      perceptron.propagate({0.0});
      perceptron.activate({1.0, 1.0});
      auto err = perceptron.propagate({1.0});
      if (!(i % 10000))
        std::cout << "MSE: " << err << "\n";
    }
    std::cout << perceptron.activate({0.0, 0.0})[0] << " (1.0)\n";
    std::cout << perceptron.activate({0.0, 1.0})[0] << " (0.0)\n";
    std::cout << perceptron.activate({1.0, 0.0})[0] << " (0.0)\n";
    std::cout << perceptron.activate({1.0, 1.0})[0] << " (1.0)\n";
  }

  {
    auto perceptron = Nevolver::NARX(2, {4, 2}, 1, 4, 4);
    for (auto i = 0; i < 5000; i++) {
      perceptron.activate({0.0, 0.0});
      perceptron.propagate({1.0});
      perceptron.activate({0.0, 1.0});
      perceptron.propagate({0.0});
      perceptron.activate({1.0, 0.0});
      perceptron.propagate({0.0});
      perceptron.activate({1.0, 1.0});
      auto err = perceptron.propagate({1.0});
      if (!(i % 1000))
        std::cout << "MSE: " << err << "\n";
    }
    std::cout << perceptron.activate({0.0, 0.0})[0] << " (1.0)\n";
    std::cout << perceptron.activate({0.0, 1.0})[0] << " (0.0)\n";
    std::cout << perceptron.activate({1.0, 0.0})[0] << " (0.0)\n";
    std::cout << perceptron.activate({1.0, 1.0})[0] << " (1.0)\n";
  }

  return 0;
}
