#ifndef LIQUID_H
#define LIQUID_H

#include "../nevolver.hpp"

namespace Nevolver {
class Liquid final : public Network {
public:
  Liquid(int inputs, int hidden_start_max, int outputs) {
    Group inputNodes;
    for (int i = 0; i < inputs; i++) {
      auto &node = _nodes.emplace_back(InputNode());
      _sortedNodes.emplace_back(node); // insertion order!
      _inputs.emplace_back(std::get<InputNode>(node));
      inputNodes.emplace_back(node);
    }

    auto nhidden = Random::nextUInt() % uint32_t(hidden_start_max);
    for(uint32_t i = 0; i < nhidden; i++) {
      auto &node = _nodes.emplace_back(HiddenNode());
      _sortedNodes.emplace_back(node); // insertion order!
    }

    Group outputNodes;
    for (int i = 0; i < outputs; i++) {
      auto &node = _nodes.emplace_back(HiddenNode(true));
      _sortedNodes.emplace_back(node); // insertion order!
      _outputs.emplace_back(node);
      outputNodes.emplace_back(node);
    }

    // connect input to outputs directly in the beginning
    connect(inputNodes, outputNodes, ConnectionPattern::AllToAll);

    // finally setup weights now that we know how many we need
    for (auto &conn : _connections) {
      auto &w = _weights.emplace_back();
      w.first = Random::init();
      w.second.insert(&conn);
      conn.weight = &w;
    }

    constexpr uint32_t max_muts = 100;

    auto nmuts = Random::nextUInt() % max_muts;
    std::vector<NetworkMutations> muts;
    for(uint32_t i = 0; i < nmuts; i++) {
      auto mut = Random::nextUInt() % uint32_t(NetworkMutations::Total);
      muts.emplace_back((NetworkMutations)mut);
    }

    nmuts = Random::nextUInt() % max_muts;
    std::vector<NodeMutations> node_muts;
    for(uint32_t i = 0; i < nmuts; i++) {
      auto mut = Random::nextUInt() % uint32_t(NodeMutations::Total);
      node_muts.emplace_back((NodeMutations)mut);
    }

    mutate(muts, 0.9, node_muts, 0.9, 0.5);
  }
};
} // namespace Nevolver

#endif /* PERCEPTRON_H */
