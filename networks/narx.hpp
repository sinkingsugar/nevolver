#ifndef NARX_H
#define NARX_H

#include "../nevolver.hpp"

namespace Nevolver {
class NARX final : public Network {
public:
  NARX(int inputs, std::vector<int> hidden, int outputs, int input_memory,
       int output_memory) {
    // keep track of pure ringbuffer memory conns to fix weights
    std::vector<std::reference_wrapper<Connection>> memoryTunnels;

    Group inputNodes;
    for (int i = 0; i < inputs; i++) {
      auto &node = _nodes.emplace_back(InputNode());
      _sortedNodes.emplace_back(node);
      _inputs.emplace_back(std::get<InputNode>(node));
      inputNodes.emplace_back(node);
    }

    std::vector<Group> outputMemory;
    outputMemory.resize(output_memory);
    for (int i = 0; i < output_memory; i++) {
      outputMemory[i] = addMemoryCell(outputs);
      if (i > 0) {
        auto conns = connect(outputMemory[i - 1], outputMemory[i],
                             ConnectionPattern::OneToOne);
        memoryTunnels.insert(memoryTunnels.end(), conns.begin(), conns.end());
      }
    }
    std::reverse(outputMemory.begin(), outputMemory.end());
    for (auto &group : outputMemory) {
      std::reverse(group.begin(), group.end());
    }
    for (auto &group : outputMemory) {
      for (auto &node : group) {
        _sortedNodes.emplace_back(node);
      }
    }

    Group *previous = &inputNodes;
    std::vector<Group> layers;
    // must pre alloc as we take ref!
    layers.reserve(hidden.size());
    for (auto lsize : hidden) {
      auto &layer = layers.emplace_back();
      for (int i = 0; i < lsize; i++) {
        auto &node = _nodes.emplace_back(HiddenNode());
        _sortedNodes.emplace_back(node);
        layer.emplace_back(node);
      }

      connect(*previous, layer, ConnectionPattern::AllToAll);
      previous = &layer;
    }

    std::vector<Group> inputMemory;
    inputMemory.resize(input_memory);
    for (int i = 0; i < input_memory; i++) {
      inputMemory[i] = addMemoryCell(inputs);
      if (i > 0) {
        auto conns = connect(inputMemory[i - 1], inputMemory[i],
                             ConnectionPattern::OneToOne);
        memoryTunnels.insert(memoryTunnels.end(), conns.begin(), conns.end());
      }
    }
    std::reverse(inputMemory.begin(), inputMemory.end());
    for (auto &group : inputMemory) {
      std::reverse(group.begin(), group.end());
    }
    for (auto &group : inputMemory) {
      for (auto &node : group) {
        _sortedNodes.emplace_back(node);
      }
    }

    Group outputNodes;
    for (int i = 0; i < outputs; i++) {
      auto &node = _nodes.emplace_back(HiddenNode(true));
      _outputs.emplace_back(node);
      _sortedNodes.emplace_back(node);
      outputNodes.emplace_back(node);
    }

    connect(*previous, outputNodes, ConnectionPattern::AllToAll);

    auto imemConns =
        connect(inputNodes, inputMemory.back(), ConnectionPattern::OneToOne);
    memoryTunnels.insert(memoryTunnels.end(), imemConns.begin(),
                         imemConns.end());

    for (auto &group : inputMemory) {
      connect(group, layers[0], ConnectionPattern::AllToAll);
    }

    auto omemConns =
        connect(outputNodes, outputMemory.back(), ConnectionPattern::OneToOne);
    memoryTunnels.insert(memoryTunnels.end(), omemConns.begin(),
                         omemConns.end());

    for (auto &group : outputMemory) {
      connect(group, layers[0], ConnectionPattern::AllToAll);
    }

    // finally setup weights now that we know how many we need
    for (auto &conn : _connections) {
      auto &w = _weights.emplace_back();
      w.first = Random::init();
      w.second.insert(&conn);
      conn.weight = &w;
    }

    // Fix up memory weights
    for (auto &conn : memoryTunnels) {
      conn.get().weight->first = 1;
    }
  }

  Group addMemoryCell(int size) {
    Group res;
    for (int y = 0; y < size; y++) {
      auto &node = _nodes.emplace_back(HiddenNode(false, true));
      auto &hiddenNode = std::get<HiddenNode>(node);
      hiddenNode.setBias(0);
      hiddenNode.setSquash(IdentityS(), IdentityD());
      res.emplace_back(node);
    }
    return res;
  }
};
} // namespace Nevolver

#endif /* NARX_H */
