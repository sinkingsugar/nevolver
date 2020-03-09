#ifndef PERCEPTRON_H
#define PERCEPTRON_H

#include "../nevolver.hpp"

namespace Nevolver {
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

#endif /* PERCEPTRON_H */
