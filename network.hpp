#ifndef NETWORK_H
#define NETWORK_H

#include "nevolver.hpp"

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

  virtual NeuroFloat propagate(const std::vector<NeuroFloat> &targets,
                               double rate = 0.3, double momentum = 0.0,
                               bool update = true) {
    size_t outputIdx = targets.size();
    _outputCache.resize(outputIdx); // reuse for MSE
    for (auto it = _nodes.rbegin(); it != _nodes.rend(); ++it) {
      std::visit(
          [&](auto &&node) {
            if (node.is_output()) {
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
          *it);
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

  virtual Network crossover(const Network &other) {
    Network offspring = *this;
    // FIXME not so easy, must fixup all references also!
    return offspring;
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

  void doMutation(NetworkMutations mutation) {}

  std::vector<std::reference_wrapper<InputNode>> _inputs;
  std::list<Connection> _connections;
  std::vector<AnyNode> _nodes;
  std::vector<NeuroFloat> _weights;

private:
  std::vector<NeuroFloat> _outputCache;
};
} // namespace Nevolver

#endif /* NETWORK_H */
