#ifndef NETWORK_H
#define NETWORK_H

#include "nevolver.hpp"
#include <unordered_map>

namespace Nevolver {
enum NetworkMutations {
  AddNode,
  SubNode,
  AddConnection,
  SubConnection,
  ShareWeight,
  SwapNodes,
  AddGate,
  SubGate,
  AddBackConnection,
  SubBackConnection
};

template <typename T> class VectorSet : public std::vector<T> {
public:
  using iterator = typename std::vector<T>::iterator;
  using value_type = typename std::vector<T>::value_type;

  std::pair<iterator, bool> insert(const value_type &val) {
    auto it = std::find(this->begin(), this->end(), val);
    if (it == this->end())
      it = std::vector<T>::insert(this->end(), val);

    return std::pair<iterator, bool>(it, true);
  }
};

class Network {
public:
  virtual const std::vector<NeuroFloat> &
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

    for (auto &rnode : _sortedNodes) {
      auto &vnode = rnode.get();
      std::visit(
          [this](auto &&node) {
            auto activation = node.activate();
            if (node.isOutput())
              _outputCache.push_back(activation);
          },
          vnode);
    }

    return _outputCache;
  }

  virtual const std::vector<NeuroFloat> &
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

    for (auto &rnode : _sortedNodes) {
      auto &vnode = rnode.get();
      std::visit(
          [this](auto &&node) {
            auto activation = node.activateFast();
            if (node.isOutput())
              _outputCache.push_back(activation);
          },
          vnode);
    }

    return _outputCache;
  }

  virtual NeuroFloat propagate(const std::vector<NeuroFloat> &targets,
                               double rate = 0.3, double momentum = 0.0,
                               bool update = true) {
    size_t outputIdx = targets.size();
    _outputCache.resize(outputIdx); // reuse for MSE
    for (auto it = _sortedNodes.rbegin(); it != _sortedNodes.rend(); ++it) {
      auto &vnode = (*it).get();
      std::visit(
          [&](auto &&node) {
            if (node.isOutput()) {
              outputIdx--;
              node.propagate(rate, momentum, update, targets[outputIdx]);
#ifdef NEVOLVER_WIDE
              for (int i = 0; i < NeuroFloatWidth; i++) {
                _outputCache[outputIdx][i] =
                    __builtin_pow(node.current()[i] - targets[outputIdx][i], 2);
              }
#else
              _outputCache[outputIdx] =
                  std::pow(node.current() - targets[outputIdx], 2);
#endif
            } else {
              node.propagate(rate, momentum, update);
            }
          },
          vnode);
    }
    NeuroFloat mean = NeuroFloatZeros;
    for (auto err : _outputCache) {
      mean += err;
    }

    NEUROWIDE(wsize, double(_outputCache.size()));
    mean /= wsize;
    return mean;
  }

  virtual void clear() {
    for (auto &node : _nodes) {
      std::visit([](auto &&node) { node.clear(); }, node);
    }
  }

  virtual void mutate(const std::vector<NetworkMutations> &network_pool,
                      double network_rate,
                      const std::vector<NodeMutations> &node_pool,
                      double node_rate, double weight_rate) {
    for (auto &node : _nodes) {
      for (auto mutation : node_pool) {
        auto chance = Random::nextDouble();
        if (chance < node_rate) {
          std::visit([mutation](auto &&node) { node.mutate(mutation); }, node);
        }
      }
    }

    for (auto &weight : _weights) {
      auto chance = Random::nextDouble();
      if (chance < weight_rate) {
        weight += Random::normal(0.0, 0.1);
      }
    }

    for (auto mutation : network_pool) {
      auto chance = Random::nextDouble();
      if (chance < network_rate) {
        doMutation(mutation);
      }
    }
  }

  static Network crossover(const Network &net1, const Network &net2) {
    Network res{};
    return res;
  }

  template <class Archive>
  void save(Archive &ar, std::uint32_t const version) const {
    std::unordered_map<const Node *, uint64_t> nodeMap;
    std::vector<AnyNode> nodes;
    uint64_t idx = 0;
    for (auto &node : _sortedNodes) {
      nodeMap.emplace((Node *)&node.get(), idx);
      nodes.emplace_back(node.get());
      idx++;
    }

    std::unordered_map<const NeuroFloat *, uint64_t> wMap;
    idx = 0;
    for (auto &w : _weights) {
      wMap.emplace(&w, idx);
      idx++;
    }

    std::vector<ConnectionInfo> conns;
    for (auto &conn : _connections) {
      conns.push_back({nodeMap[conn.from], nodeMap[conn.to],
                       conn.gater != nullptr, nodeMap[conn.gater],
                       wMap[conn.weight]});
    }

    std::vector<uint64_t> inputs;
    for (auto &inp : _inputs) {
      inputs.emplace_back(nodeMap[&inp.get()]);
    }

    ar(nodes, _weights, conns, inputs);
  }

  template <class Archive> void load(Archive &ar, std::uint32_t const version) {
    std::vector<AnyNode> nodes;
    std::vector<ConnectionInfo> conns;
    std::vector<uint64_t> inputs;

    ar(nodes, _weights, conns, inputs);

    for (auto &node : nodes) {
      auto &nref = _nodes.emplace_front(node);
      _sortedNodes.emplace_back(nref);
    }

    for (auto &conn : conns) {
      auto &c = connect(_sortedNodes[conn.fromIdx].get(),
                        _sortedNodes[conn.toIdx].get());
      if (conn.hasGater)
        gate(_sortedNodes[conn.gaterIdx].get(), c);
      c.weight = &_weights[conn.weightIdx];
    }

    for (auto idx : inputs) {
      _inputs.emplace_back(std::get<InputNode>(_sortedNodes[idx].get()));
    }
  }

  struct ConnectionInfo {
    uint64_t fromIdx;
    uint64_t toIdx;
    bool hasGater;
    uint64_t gaterIdx;
    uint64_t weightIdx;

    template <class Archive>
    void serialize(Archive &ar, std::uint32_t const version) {
      ar(fromIdx, toIdx, hasGater, gaterIdx, weightIdx);
    }
  };

  std::vector<NeuroFloat> &weights() { return _weights; }

  const std::list<Connection> &connections() { return _connections; }

  const std::vector<std::reference_wrapper<AnyNode>> &nodes() {
    return _sortedNodes;
  }

protected:
  std::list<AnyNode>::iterator remove(std::list<AnyNode>::iterator &nit) {
    AnyNode &node = *nit;
    if (node.index() == 0)
      return ++nit; // don't remove inputs

    const Node *nptr = (Node *)&node;

    if (nptr->isOutput())
      return ++nit; // don't remove outputs

    // need to remove all connections
    for (auto &conn : nptr->connections().outbound) {
      disconnect(*conn);
    }
    for (auto &conn : nptr->connections().inbound) {
      disconnect(*conn);
    }
    if (nptr->connections().self) {
      disconnect(*nptr->connections().self);
    }

    // gates
    for (auto &conn : nptr->connections().gate) {
      ungate(node, *conn);
    }

    // need to erase from sorted
    // this will shift ptrs around
    auto sit = _sortedNodes.begin();
    while (sit != _sortedNodes.end()) {
      auto &inode = sit->get();
      if (&inode == &node) {
        _sortedNodes.erase(sit);
        break;
      }
      ++sit;
    }

    return _nodes.erase(nit);
  }

  Connection &connect(AnyNode &from, AnyNode &to) {
    auto &conn = _connections.emplace_front(
        Connection{(Node *)&from, (Node *)&to, nullptr});

    if (&from == &to) {
      std::visit([&](auto &&node) { node.addSelfConnection(conn); }, to);
    } else {
      std::visit([&](auto &&node) { node.addOutboundConnection(conn); }, from);
      std::visit([&](auto &&node) { node.addInboundConnection(conn); }, to);
    }

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
    }
    return conns;
  }

  std::list<Connection>::iterator
  disconnect(std::list<Connection>::iterator &cit) {
    Connection &conn = *cit;
    if (conn.from != conn.to) {
      conn.from->removeOutboundConnection(conn);
      conn.to->removeInboundConnection(conn);
    } else {
      conn.to->removeSelfConnection(conn);
    }
    if (conn.gater) {
      conn.gater->removeGate(conn);
    }
    return _connections.erase(cit);
  }

  void disconnect(Connection &conn) {
    auto it = _connections.begin();
    while (it != _connections.end()) {
      auto &c = *it;
      if (&c == &conn) {
        disconnect(it);
        return;
      }
      ++it;
    }
  }

  void disconnect(AnyNode &from, AnyNode &to) {
    auto it = _connections.begin();
    while (it != _connections.end()) {
      auto &conn = *it;
      if (conn.from == (Node *)&from && conn.to == (Node *)&to) {
        it = disconnect(it);
      } else {
        ++it;
      }
    }
  }

  bool isConnected(const Node *from, const Node *to) {
    auto it = _connections.begin();
    while (it != _connections.end()) {
      auto &conn = *it;
      if (conn.from == from && conn.to == to) {
        return true;
      } else {
        ++it;
      }
    }
    return false;
  }

  bool isConnected(AnyNode &from, AnyNode &to) {
    return isConnected((Node *)&from, (Node *)&to);
  }

  void gate(AnyNode &gater, Connection &conn) {
    conn.gater = (Node *)&gater;
    std::visit([&conn](auto &&node) { node.addGate(conn); }, gater);
  }

  void ungate(AnyNode &gater, Connection &conn) {
    conn.gater = nullptr;
    std::visit([&conn](auto &&node) { node.removeGate(conn); }, gater);
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
    VectorSet<const Node *> nodesFrom;
    VectorSet<const Node *> nodesTo;
    std::set<const Connection *> conns;
    for (auto &conn : connections) {
      conns.insert(&conn.get());
      nodesFrom.insert(conn.get().from);
      nodesTo.insert(conn.get().to);
    }

    auto gsize = group.size();
    switch (pattern) {
    case GatingPattern::Input: {
      size_t idx = 0;
      for (auto node : nodesTo) {
        auto &gater = group[idx % gsize];
        for (auto &conn : node->connections().inbound) {
          if (conns.count(conn))
            gate(gater, *conn);
        }
        idx++;
      }
    } break;
    case GatingPattern::Output: {
      size_t idx = 0;
      for (auto node : nodesFrom) {
        auto &gater = group[idx % gsize];
        for (auto &conn : node->connections().outbound) {
          if (conns.count(conn))
            gate(gater, *conn);
        }
        idx++;
      }
    } break;
    case GatingPattern::Self: {
      size_t idx = 0;
      for (auto node : nodesFrom) {
        auto &gater = group[idx % gsize];
        if (conns.count(node->connections().self))
          gate(gater, *node->connections().self);
        idx++;
      }
    } break;
    }
  }

  void doMutation(NetworkMutations mutation) {}

  std::vector<std::reference_wrapper<InputNode>> _inputs;
  std::vector<std::reference_wrapper<AnyNode>> _sortedNodes;

  std::list<AnyNode> _nodes;
  std::list<Connection> _connections;
  std::vector<NeuroFloat> _weights;

private:
#ifdef NEVOLVER_WIDE
  std::vector<NeuroFloat> _wideInputs;
#endif
  std::vector<NeuroFloat> _outputCache;
};
} // namespace Nevolver

CEREAL_CLASS_VERSION(Nevolver::Network, NEVOLVER_VERSION);
CEREAL_CLASS_VERSION(Nevolver::Network::ConnectionInfo, NEVOLVER_VERSION);

#endif /* NETWORK_H */
