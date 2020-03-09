#ifndef NARX_H
#define NARX_H

#include "../nevolver.hpp"

namespace Nevolver {
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
};
} // namespace Nevolver

#endif /* NARX_H */
