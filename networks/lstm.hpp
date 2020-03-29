#ifndef LSTM_H
#define LSTM_H

#include "../network.hpp"
#include "../nodes/annhidden.hpp"

namespace Nevolver {
class LSTM final : public Network {
public:
  LSTM(int inputs, std::vector<int> hidden, int outputs) {
    Group inputNodes;
    for (int i = 0; i < inputs; i++) {
      auto &node = _nodes.emplace_front(InputNode());
      _sortedNodes.emplace_back(node);
      _inputs.emplace_back(std::get<InputNode>(node));
      inputNodes.emplace_back(node);
    }

    Group outputNodes;
    for (int i = 0; i < outputs; i++) {
      auto &node = _nodes.emplace_front(HiddenNode(true));
      _outputs.emplace_back(node);
      outputNodes.emplace_back(node);
    }

    Group *previous = &inputNodes;
    std::vector<Group> layers;
    // must pre alloc as we take ref!
    auto hsize = hidden.size();
    layers.reserve((hsize * 5) - 1);
    for (size_t i = 0; i < hsize; i++) {
      auto lsize = hidden[i];
      auto &inputGate = layers.emplace_back();
      auto &forgetGate = layers.emplace_back();
      auto &memoryCell = layers.emplace_back();
      auto &outputGate = layers.emplace_back();
      auto &outputBlock = i == hsize - 1 ? outputNodes : layers.emplace_back();

      for (int i = 0; i < lsize; i++) {
        auto &node = _nodes.emplace_front(HiddenNode());
        _sortedNodes.emplace_back(node);
        inputGate.emplace_back(node);
        auto &hidden = std::get<HiddenNode>(node);
        hidden.setBias(NeuroFloatOnes);
      }

      for (int i = 0; i < lsize; i++) {
        auto &node = _nodes.emplace_front(HiddenNode());
        _sortedNodes.emplace_back(node);
        forgetGate.emplace_back(node);
        auto &hidden = std::get<HiddenNode>(node);
        hidden.setBias(NeuroFloatOnes);
      }

      for (int i = 0; i < lsize; i++) {
        auto &node = _nodes.emplace_front(HiddenNode());
        _sortedNodes.emplace_back(node);
        memoryCell.emplace_back(node);
      }

      for (int i = 0; i < lsize; i++) {
        auto &node = _nodes.emplace_front(HiddenNode());
        _sortedNodes.emplace_back(node);
        outputGate.emplace_back(node);
        auto &hidden = std::get<HiddenNode>(node);
        hidden.setBias(NeuroFloatOnes);
      }

      if (i != hsize - 1) {
        for (int i = 0; i < lsize; i++) {
          auto &node = _nodes.emplace_front(HiddenNode());
          _sortedNodes.emplace_back(node);
          outputBlock.emplace_back(node);
        }
      } else {
        for (auto &node : outputNodes) {
          _sortedNodes.emplace_back(node);
        }
      }

      // always connect input to inners
      auto inputConn =
          connect(*previous, memoryCell, ConnectionPattern::AllToAll);
      connect(*previous, inputGate, ConnectionPattern::AllToAll);
      connect(*previous, outputGate, ConnectionPattern::AllToAll);
      connect(*previous, forgetGate, ConnectionPattern::AllToAll);

      connect(memoryCell, inputGate, ConnectionPattern::AllToAll);
      connect(memoryCell, forgetGate, ConnectionPattern::AllToAll);
      connect(memoryCell, outputGate, ConnectionPattern::AllToAll);

      auto forgetConn =
          connect(memoryCell, memoryCell, ConnectionPattern::OneToOne);
      auto outputConn =
          connect(memoryCell, outputBlock, ConnectionPattern::AllToAll);

      gate(inputGate, inputConn, GatingPattern::Input);
      gate(forgetGate, forgetConn, GatingPattern::Self);
      gate(outputGate, outputConn, GatingPattern::Output);

#if 1
      // input to deep
      if (i > 0) {
        auto cellConn =
            connect(inputNodes, memoryCell, ConnectionPattern::AllToAll);
        gate(inputGate, cellConn, GatingPattern::Input);
      }
#endif

      previous = &outputBlock;
    }

#if 1
    // direct input->output connection
    connect(inputNodes, outputNodes, ConnectionPattern::AllToAll);
#endif

    // finally setup weights now that we know how many we need
    for (auto &conn : _connections) {
      auto &w = _weights.emplace_front();
      w.first = Random::normal(0.0, 1.0);
      w.second.insert(&conn);
      conn.weight = &w;
    }
  }
};
} // namespace Nevolver

#endif /* LSTM_H */
