#ifndef NODE_H
#define NODE_H

#include "connections.hpp"

namespace Nevolver {
enum class NodeMutations { Squash, Bias, Total };
enum class NodeKind { Normal, Input, Output };

class Node {
public:
  NeuroFloat current() const { return _activation; }

  const NodeConnections &connections() const { return _connections; }

  NeuroFloat responsibility() const { return _responsibility; }

  void addInboundConnection(Connection &conn) const {
    _connections.inbound.push_back(&conn);
  }

  void removeInboundConnection(Connection &conn) const {
    _connections.inbound.erase(
        std::remove_if(_connections.inbound.begin(), _connections.inbound.end(),
                       [&](auto &&c) { return c == &conn; }),
        _connections.inbound.end());
  }

  void addOutboundConnection(Connection &conn) const {
    _connections.outbound.push_back(&conn);
  }

  void removeOutboundConnection(Connection &conn) const {
    _connections.outbound.erase(
        std::remove_if(_connections.outbound.begin(),
                       _connections.outbound.end(),
                       [&](auto &&c) { return c == &conn; }),
        _connections.outbound.end());
  }

  void addSelfConnection(Connection &conn) const {
    if (_connections.self) {
      throw std::runtime_error("Node already has a self connection.");
    }
    _connections.self = &conn;
  }

  void removeSelfConnection(Connection &_conn) const {
    _connections.self = nullptr;
  }

  void addGate(Connection &conn) const { _connections.gate.push_back(&conn); }

  void removeGate(Connection &conn) const {
    _connections.gate.erase(
        std::remove_if(_connections.gate.begin(), _connections.gate.end(),
                       [&](auto &&c) { return c == &conn; }),
        _connections.gate.end());
  }

  bool isOutput() const { return _kind == NodeKind::Output; }
  bool isInput() const { return _kind == NodeKind::Input; }

  void setOutput(bool output) {
    _kind = output ? NodeKind::Output : NodeKind::Normal;
  }

protected:
  NeuroFloat _activation{0};
  NeuroFloat _responsibility{0};
  NodeKind _kind = NodeKind::Normal;
  mutable NodeConnections _connections{};
};

template <typename T> class NodeCommon : public Node {
public:
  NeuroFloat activate() { return as_underlying().doActivate(); }

  NeuroFloat activateFast() { return as_underlying().doFastActivate(); }

  void propagate(double rate, double momentum, bool update,
                 NeuroFloat target = 0) {
    return as_underlying().doPropagate(rate, momentum, update, target);
  }

  void clear() { as_underlying().doClear(); }

  void mutate(NodeMutations mutation) { as_underlying().doMutate(mutation); }

  T clone() {
    // clone this but without any connection and such
    T res = as_underlying();
    res._connections.inbound.clear();
    res._connections.outbound.clear();
    res._connections.gate.clear();
    res._connections.self = nullptr;
    return res;
  }

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
  InputNode() : NodeCommon<InputNode>() { _kind = NodeKind::Input; }

  void setInput(NeuroFloat input) { _activation = input; }

  NeuroFloat doActivate() { return _activation; }

  NeuroFloat doFastActivate() { return _activation; }

  void doPropagate(double rate, double momentum, bool update,
                   NeuroFloat target) {}

  void doClear() {}

  void doMutate(NodeMutations mutation) {
    // Input has none
  }

  template <class Archive>
  void serialize(Archive &ar, std::uint32_t const version) {}
};
} // namespace Nevolver

CEREAL_CLASS_VERSION(Nevolver::InputNode, NEVOLVER_VERSION);

#endif /* NODE_H */
