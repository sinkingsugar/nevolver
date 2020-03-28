#ifndef NETWORK_H
#define NETWORK_H

#include "nevolver.hpp"

namespace Nevolver {
enum NetworkMutations {
  AddNode,
  SubNode,
  AddFwdConnection,
  AddBwdConnection,
  SubConnection,
  ShareWeight,
  SwapNodes,
  AddGate,
  SubGate,
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
    for (auto &node : _sortedNodes) {
      for (auto mutation : node_pool) {
        auto chance = Random::nextDouble();
        if (chance < node_rate) {
          std::visit([mutation](auto &&node) { node.mutate(mutation); },
                     node.get());
        }
      }
    }

    for (auto &weight : _weights) {
      if (!weight.second.empty()) {
        auto chance = Random::nextDouble();
        if (chance < weight_rate) {
          weight.first += Random::normal(0.0, 0.1);
        }
      }
    }

    for (auto mutation : network_pool) {
      auto chance = Random::nextDouble();
      if (chance < network_rate) {
        doMutation(mutation);
      }
    }
  }

  // https://en.wikipedia.org/wiki/Pairing_function
  static uint64_t pairing(uint64_t a, uint64_t b) {
    return uint64_t(((1.0 / 2.0) * (double(a) + double(b)) *
                     (double(a) + double(b) + 1.0)) +
                    double(b));
  }

  static Network crossover(const Network &net1, const Network &net2) {
    Network res{};
    // try to keep some statistic to see if crossover is worth it
    res._crossoverScore =
        1 + pairing(net1._crossoverScore, net2._crossoverScore);

    auto n1inputs = net1._inputs.size();
    auto n2inputs = net2._inputs.size();
    auto n1outputs = std::count_if(
        net1._sortedNodes.begin(), net1._sortedNodes.end(), [](auto &&n) {
          return std::visit([](auto &&n) { return n.isOutput(); }, n.get());
        });
    auto n2outputs = std::count_if(
        net2._sortedNodes.begin(), net2._sortedNodes.end(), [](auto &&n) {
          return std::visit([](auto &&n) { return n.isOutput(); }, n.get());
        });

    if (n1inputs != n2inputs || n1outputs != n2outputs)
      throw std::runtime_error("Attempted crossover between networks with "
                               "different input/output sizes.");

    // parents are picked randomly already
    // no need to overengineer the `==` case
    auto newLen = net1._fitness > net2._fitness ? net1._sortedNodes.size()
                                                : net2._sortedNodes.size();

    auto &mainnet = net1._fitness > net2._fitness ? net1 : net2;
    for (size_t i = 0; i < newLen; i++) {
      auto &mainnode = mainnet._sortedNodes[i].get();
      auto &newNode = res._nodes.emplace_front();
      res._sortedNodes.emplace_back(newNode);
      if (mainnode.index() == 0 ||
          std::visit([](auto &&n) { return n.isOutput(); }, mainnode)) {
        // if input or output use main as it's the closest architecture
        newNode =
            std::visit([](auto &&n) { return AnyNode(n.clone()); }, mainnode);
        if (mainnode.index() == 0)
          res._inputs.emplace_back(std::get<InputNode>(newNode));
      } else {
        auto &parent = Random::nextDouble() < 0.5 ? net1 : net2;
        auto &node = i < parent._sortedNodes.size()
                         ? parent._sortedNodes[i].get()
                         : mainnet._sortedNodes[i].get();

        // input and outputs are already dealt with
        if (node.index() == 0 ||
            std::visit([](auto &&n) { return n.isOutput(); }, node)) {
          // fall back to mainnode
          newNode =
              std::visit([](auto &&n) { return AnyNode(n.clone()); }, mainnode);
        } else {
          newNode =
              std::visit([](auto &&n) { return AnyNode(n.clone()); }, node);
        }
      }
    }

    // so we do all this to find shared connections
    // in order to give priority
    // shared as in topology

    struct ConnData {
      const Connection *conn;
      size_t fidx;
      size_t tidx;
      size_t gidx;
    };

    std::vector<ConnData> connections;

    {
      std::map<uint64_t, ConnData> conns1;
      std::map<uint64_t, ConnData> conns2;
      {
        std::unordered_map<const Node *, uint64_t> nodeMap1;
        uint64_t idx = 0;
        for (auto &node : net1._sortedNodes) {
          nodeMap1.emplace((Node *)&node.get(), idx);
          idx++;
        }

        std::unordered_map<const Node *, uint64_t> nodeMap2;
        idx = 0;
        for (auto &node : net2._sortedNodes) {
          nodeMap2.emplace((Node *)&node.get(), idx);
          idx++;
        }

        for (auto &conn : net1._activeConns) {
          auto fidx = nodeMap1[conn->from];
          auto tidx = nodeMap1[conn->to];
          auto gidx = conn->gater ? nodeMap1[conn->gater] : 0;
          auto id = pairing(fidx, tidx);
          auto [_, added] =
              conns1.emplace(id, ConnData{conn, fidx, tidx, gidx});
          assert(added); // likely there is a bug somewhere
        }

        for (auto &conn : net2._activeConns) {
          auto fidx = nodeMap2[conn->from];
          auto tidx = nodeMap2[conn->to];
          auto gidx = conn->gater ? nodeMap1[conn->gater] : 0;
          auto id = pairing(fidx, tidx);
          auto [_, added] =
              conns2.emplace(id, ConnData{conn, fidx, tidx, gidx});
          assert(added); // likely there is a bug somewhere
        }
      }

      auto &primary = net1._fitness > net2._fitness ? conns1 : conns2;
      auto &secondary = net1._fitness > net2._fitness ? conns2 : conns1;
      for (auto [k, v] : primary) {
        auto sit = secondary.find(k);
        if (sit != secondary.end()) {
          // this connection is shared
          // so we can pick from the other net
          if (Random::nextDouble() < 0.5) {
            connections.emplace_back(v);
            auto &w = res._weights.emplace_back(*v.conn->weight);
            w.second.clear();
          } else {
            connections.emplace_back(sit->second);
            auto &w = res._weights.emplace_back(*sit->second.conn->weight);
            w.second.clear();
          }
        } else {
          // let the strongest win if not shared
          connections.emplace_back(v);
          auto &w = res._weights.emplace_back(*v.conn->weight);
          w.second.clear();
        }
      }
    }

    auto widx = 0;
    for (auto &conn : connections) {
      auto &nc = res.connect(res._sortedNodes[conn.fidx].get(),
                             res._sortedNodes[conn.tidx].get());
      if (conn.conn->gater) {
        res.gate(res._sortedNodes[conn.gidx].get(), nc);
      }
      nc.weight = &res._weights[widx];
      res._weights[widx].second.insert(&nc);
      widx++;
    }

    return res;
  }

  template <class Archive>
  void save(Archive &ar, std::uint32_t const version) const {
    std::unordered_map<const Node *, uint64_t> nodeMap;
    std::vector<AnyNode> nodes;
    std::vector<NeuroFloat> weights;

    uint64_t idx = 0;
    for (auto &node : _sortedNodes) {
      nodeMap.emplace((Node *)&node.get(), idx);
      nodes.emplace_back(node.get());
      idx++;
    }

    std::unordered_map<const Weight *, uint64_t> wMap;
    idx = 0;
    for (auto &w : _weights) {
      wMap.emplace(&w, idx);
      weights.push_back(w.first);
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

    ar(nodes, weights, conns, inputs);
  }

  template <class Archive> void load(Archive &ar, std::uint32_t const version) {
    std::vector<AnyNode> nodes;
    std::vector<uint64_t> inputs;
    std::vector<ConnectionInfo> conns;
    std::vector<NeuroFloat> weights;

    ar(nodes, weights, conns, inputs);

    for (auto &node : nodes) {
      auto &nref = _nodes.emplace_front(node);
      _sortedNodes.emplace_back(nref);
    }

    for (auto &conn : conns) {
      auto &c = connect(_sortedNodes[conn.fromIdx].get(),
                        _sortedNodes[conn.toIdx].get());
      if (conn.hasGater)
        gate(_sortedNodes[conn.gaterIdx].get(), c);
      auto &w = _weights.emplace_front();
      w.first = weights[conn.weightIdx];
      w.second.insert(&c);
      c.weight = &w;
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

  std::deque<Weight> &weights() { return _weights; }

  const std::deque<Connection> &connections() { return _connections; }

  const std::vector<std::reference_wrapper<AnyNode>> &nodes() {
    return _sortedNodes;
  }

  struct Stats {
    size_t activeNodes;
    size_t activeConnections;
    size_t activeWeights;
    size_t unusedNodes;
    size_t unusedConnections;
    size_t unusedWeights;
    uint64_t crossoverScore;
  };

  Stats getStats() {
    Stats stats;
    stats.activeNodes = _sortedNodes.size();
    stats.activeConnections = _activeConns.size();
    stats.activeWeights = (_weights.size() - _unusedWeights.size());
    stats.unusedNodes = _unusedNodes.size();
    stats.unusedConnections = _unusedConns.size();
    stats.unusedWeights = _unusedWeights.size();
    stats.crossoverScore = _crossoverScore;
    return stats;
  }

  void printStats() {
    std::cout << "Active-Nodes: " << _sortedNodes.size() << "\n";
    std::cout << "Active-Connections: " << _activeConns.size() << "\n";
    std::cout << "Active-Weights: " << (_weights.size() - _unusedWeights.size())
              << "\n";
    std::cout << "Unused-Nodes: " << _unusedNodes.size() << "\n";
    std::cout << "Unused-Connections: " << _unusedConns.size() << "\n";
    std::cout << "Unused-Weights: " << _unusedWeights.size() << "\n";
  }

  template <typename NodesIterator>
  NodesIterator removeNode(NodesIterator &nit) {
    AnyNode &node = (*nit).get();
    if (node.index() == 0)
      return ++nit; // don't remove inputs

    const Node *nptr = (Node *)&node;

    if (nptr->isOutput())
      return ++nit; // don't remove outputs

    cleanupNode(node);

    // need to erase from sorted
    // this will shift ptrs around
    auto sit = _nodes.begin();
    while (sit != _nodes.end()) {
      auto &inode = *sit;
      if (&inode == &node) {
        auto idx = std::distance(_nodes.begin(), sit);
        _unusedNodes.push_back(idx);
        break;
      }
      ++sit;
    }

    return _sortedNodes.erase(nit);
  }

protected:
  void cleanupNode(AnyNode &node) {
    const Node *nptr = (Node *)&node;
    std::vector<Connection *> conns;

    // gates
    // collect first
    for (auto &conn : nptr->connections().gate) {
      conns.push_back(conn);
    }

    for (auto &conn : conns) {
      ungate(node, *conn);
    }

    // need to remove all connections
    // collected them first then disconnect
    conns.clear();
    for (auto &conn : nptr->connections().outbound) {
      conns.push_back(conn);
    }
    for (auto &conn : nptr->connections().inbound) {
      conns.push_back(conn);
    }
    if (nptr->connections().self) {
      conns.push_back(nptr->connections().self);
    }

    for (auto &conn : conns) {
      disconnect(*conn);
    }
  }

  template <typename NodesIterator>
  NodesIterator removeAnyNode(NodesIterator &nit) {
    AnyNode &node = *nit;
    if (node.index() == 0)
      return ++nit; // don't remove inputs

    const Node *nptr = (Node *)&node;

    if (nptr->isOutput())
      return ++nit; // don't remove outputs

    cleanupNode(node);

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

    auto idx = std::distance(_nodes.begin(), nit);
    _unusedNodes.push_back(idx);

    return ++nit;
  }

  Connection &connect(AnyNode &from, AnyNode &to) {
    Connection *conn;

    if (!_unusedConns.empty()) {
      auto cidx = _unusedConns.back();
      _unusedConns.pop_back();
      conn = &_connections[cidx];
    } else {
      conn = &_connections.emplace_front();
    }
    conn->from = (Node *)&from;
    conn->to = (Node *)&to;
    conn->gater = nullptr;

    _activeConns.emplace_back(conn);

    if (&from == &to) {
      std::visit([&](auto &&node) { node.addSelfConnection(*conn); }, to);
    } else {
      std::visit([&](auto &&node) { node.addOutboundConnection(*conn); }, from);
      std::visit([&](auto &&node) { node.addInboundConnection(*conn); }, to);
    }

    return *conn;
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

  void releaseWeight(const Weight *weight) {
    if (weight->second.size() == 0) {
      auto wit = _weights.begin();
      while (wit != _weights.end()) {
        auto &w = *wit;
        if (weight == &w) {
          auto widx = std::distance(_weights.begin(), wit);
          _unusedWeights.push_back(widx);
          break;
        } else {
          ++wit;
        }
      }
    }
  }

  template <typename ConnectionsIterator>
  ConnectionsIterator disconnect(ConnectionsIterator &cit) {
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

    conn.weight->second.erase(&conn);
    releaseWeight(conn.weight);

    // Add storage idx to recycle
    auto idx = std::distance(_connections.begin(), cit);
    _unusedConns.push_back(idx);

    // quick remove from active conns too
    auto pos =
        std::find(std::begin(_activeConns), std::end(_activeConns), &conn);
    if (pos != std::end(_activeConns)) {
      *pos = std::move(_activeConns.back());
      _activeConns.pop_back();
    }

    return ++cit;
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
    std::unordered_set<const Connection *> conns;
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

  void doMutation(NetworkMutations mutation) {
    switch (mutation) {
    case AddNode: {
      if (_activeConns.size() == 0)
        return;

      // Add a node by inserting it in the middle of a connection
      auto ridx = Random::nextUInt() % _activeConns.size();
      auto &conn = *_activeConns[ridx];
      auto gater = conn.gater;

      // store to node position
      auto pos =
          std::find_if(std::begin(_sortedNodes), std::end(_sortedNodes),
                       [&](auto &&n) { return (Node *)&n.get() == conn.to; });

      disconnect(conn);

      // make a new node
      AnyNode *newNode;
      if (!_unusedNodes.empty()) {
        auto nidx = _unusedNodes.back();
        _unusedNodes.pop_back();
        newNode = &_nodes[nidx];
      } else {
        newNode = &_nodes.emplace_front();
      }

      // init and mutate new node
      auto &anyTo = *((AnyNode *)conn.to);
      *newNode = std::visit([](auto &&n) { return AnyNode(n.clone()); }, anyTo);
      std::visit([](auto &&n) { n.mutate(NodeMutations::Squash); }, *newNode);
      std::visit([](auto &&n) { n.setOutput(false); }, *newNode);

      // insert into the network
      if (pos != std::end(_sortedNodes)) {
        _sortedNodes.insert(pos, *newNode);
      }

      // connect it
      auto &anyFrom = *((AnyNode *)conn.from);
      auto &c1 = connect(anyFrom, *newNode);
      auto &c2 = connect(*newNode, anyTo);
      if (gater) {
        if (Random::nextDouble() < 0.5) {
          gate(*newNode, c1);
        } else {
          gate(*newNode, c2);
        }
      }

      // Add new weights
      Weight *w1;
      if (!_unusedWeights.empty()) {
        auto widx = _unusedWeights.back();
        _unusedWeights.pop_back();
        w1 = &_weights[widx];
      } else {
        w1 = &_weights.emplace_front();
      }
      w1->first = Random::normal(0.0, 1.0);
      w1->second.insert(&c1);

      Weight *w2;
      if (!_unusedWeights.empty()) {
        auto widx = _unusedWeights.back();
        _unusedWeights.pop_back();
        w2 = &_weights[widx];
      } else {
        w2 = &_weights.emplace_front();
      }
      w2->first = Random::normal(0.0, 1.0);
      w2->second.insert(&c2);
    } break;
    case SubNode: {
      if (_sortedNodes.size() == 0)
        return;

      auto nidx = Random::nextUInt() % _sortedNodes.size();
      auto &node = _sortedNodes[nidx].get();

      // ignore output and input nodes
      if (node.index() == 0 || ((Node *)&node)->isOutput())
        return;

      auto nit = _sortedNodes.begin() + nidx;
      removeNode(nit);
    } break;
    case AddFwdConnection:
    case AddBwdConnection: {
      // collect possible forward/backward connections
      // includes self connections too!
      std::vector<std::pair<std::reference_wrapper<AnyNode>,
                            std::reference_wrapper<AnyNode>>>
          _availConns;
      if (mutation == AddFwdConnection)
        for (auto fit = _sortedNodes.begin(); fit != _sortedNodes.end();
             ++fit) {
          for (auto tit = fit; tit != _sortedNodes.end(); ++tit) {
            if (!isConnected(*fit, *tit))
              _availConns.emplace_back(*fit, *tit);
          }
        }
      else
        for (auto fit = _sortedNodes.rbegin(); fit != _sortedNodes.rend();
             ++fit) {
          for (auto tit = fit; tit != _sortedNodes.rend(); ++tit) {
            if (!isConnected(*fit, *tit))
              _availConns.emplace_back(*fit, *tit);
          }
        }

      if (_availConns.size() == 0)
        return;

      auto cidx = Random::nextUInt() % _availConns.size();
      auto &pair = _availConns[cidx];
      auto &conn = connect(pair.first.get(), pair.second.get());

      // Add new weights
      Weight *w;
      if (!_unusedWeights.empty()) {
        auto widx = _unusedWeights.back();
        _unusedWeights.pop_back();
        w = &_weights[widx];
      } else {
        w = &_weights.emplace_front();
      }
      w->first = Random::normal(0.0, 1.0);
      w->second.insert(&conn);
    } break;
    case SubConnection: {
      if (_activeConns.size() == 0)
        return;

      auto ridx = Random::nextUInt() % _activeConns.size();
      auto &conn = *_activeConns[ridx];
      disconnect(conn);
    } break;
    case ShareWeight: {
      if (_activeConns.size() < 2)
        return;

      auto c1idx = Random::nextUInt() % _activeConns.size();
      auto c2idx = Random::nextUInt() % _activeConns.size();
      auto &c1 = *_activeConns[c1idx];
      auto &c2 = *_activeConns[c2idx];

      auto w1 = c1.weight;
      auto w2 = c2.weight;

      c1.weight = w2;

      w1->second.erase(&c1);
      releaseWeight(w1);
    } break;
    case SwapNodes: {
      if (_sortedNodes.size() < 2)
        return;

      auto n1idx = Random::nextUInt() % _sortedNodes.size();
      auto n2idx = Random::nextUInt() % _sortedNodes.size();
      auto &n1 = _sortedNodes[n1idx].get();
      // ignore output and input nodes
      if (n1.index() == 0 || ((Node *)&n1)->isOutput())
        return;
      auto &n2 = _sortedNodes[n2idx].get();
      // ignore output and input nodes
      if (n2.index() == 0 || ((Node *)&n2)->isOutput())
        return;

      auto n1copy = n1;
      _sortedNodes[n1idx] = _sortedNodes[n2idx];
      _sortedNodes[n2idx] = n1copy;
    } break;
    case AddGate: {
      std::vector<Connection *> _nonGated;
      for (auto conn : _activeConns) {
        if (!conn->gater)
          _nonGated.emplace_back(conn);
      }

      if (_nonGated.size() == 0)
        return;

      auto nidx = Random::nextUInt() % _sortedNodes.size();
      auto &node = _sortedNodes[nidx].get();
      // ignore input nodes
      if (node.index() == 0)
        return;

      auto ridx = Random::nextUInt() % _nonGated.size();
      auto &conn = *_nonGated[ridx];
      gate(node, conn);
    } break;
    case SubGate: {
      std::vector<Connection *> _gated;
      for (auto conn : _activeConns) {
        if (conn->gater)
          _gated.emplace_back(conn);
      }

      if (_gated.size() == 0)
        return;

      auto ridx = Random::nextUInt() % _gated.size();
      auto &conn = *_gated[ridx];
      ungate(*((AnyNode *)conn.gater), conn);
    } break;
    }
  }

  std::vector<std::reference_wrapper<InputNode>> _inputs;
  std::vector<std::reference_wrapper<AnyNode>> _sortedNodes;
  std::vector<Connection *> _activeConns;

  std::deque<AnyNode> _nodes;
  std::deque<Connection> _connections;
  std::deque<Weight> _weights;

  // the following are useful when mutationg
  // often we remove nodes/conns/weights
  // but we wanna keep them in memory
  // in order to recycle
  std::vector<size_t> _unusedNodes;
  std::vector<size_t> _unusedConns;
  std::vector<size_t> _unusedWeights;

private:
#ifdef NEVOLVER_WIDE
  std::vector<NeuroFloat> _wideInputs;
#endif
  std::vector<NeuroFloat> _outputCache;

  uint64_t _crossoverScore = 0;

  double _fitness = std::numeric_limits<double>::min();
};
} // namespace Nevolver

CEREAL_CLASS_VERSION(Nevolver::Network, NEVOLVER_VERSION);
CEREAL_CLASS_VERSION(Nevolver::Network::ConnectionInfo, NEVOLVER_VERSION);

#endif /* NETWORK_H */
