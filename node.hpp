#ifndef NODE_H
#define NODE_H

#include "nevolver.hpp"

namespace Nevolver {
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
} // namespace Nevolver

#endif /* NODE_H */
