#ifndef LSTM_H
#define LSTM_H

#include "../nevolver.hpp"

namespace Nevolver {
class LSTM final : public Network {
public:
  LSTM(int inputs, std::vector<int> hidden, int outputs) {
    auto total_size = inputs + outputs;
    for (auto lsize : hidden) {
      total_size += lsize * 4;
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
    layers.reserve(hidden.size() * 4);
    for (auto lsize : hidden) {
      auto &inputGate = layers.emplace_back();
      auto &forgetGate = layers.emplace_back();
      auto &memoryCell = layers.emplace_back();
      auto &outputGate = layers.emplace_back();

      for (int i = 0; i < lsize; i++) {
        auto &node = _nodes.emplace_back(HiddenNode());
        inputGate.emplace_back(node);
        auto &hidden = std::get<HiddenNode>(node);
        hidden.setBias(1.0);
      }

      for (int i = 0; i < lsize; i++) {
        auto &node = _nodes.emplace_back(HiddenNode());
        forgetGate.emplace_back(node);
        auto &hidden = std::get<HiddenNode>(node);
        hidden.setBias(1.0);
      }

      for (int i = 0; i < lsize; i++) {
        auto &node = _nodes.emplace_back(HiddenNode());
        memoryCell.emplace_back(node);
      }

      for (int i = 0; i < lsize; i++) {
        auto &node = _nodes.emplace_back(HiddenNode());
        outputGate.emplace_back(node);
        auto &hidden = std::get<HiddenNode>(node);
        hidden.setBias(1.0);
      }
    }

    Group outputNodes;
    for (int i = 0; i < outputs; i++) {
      auto &node = _nodes.emplace_back(HiddenNode(true));
      outputNodes.emplace_back(node);
    }

    // Do connections now!
    for (size_t i = 0; i < hidden.size(); i++) {
      auto &inputGate  = layers[(i * 4) + 0];
      auto &forgetGate = layers[(i * 4) + 1];
      auto &memoryCell = layers[(i * 4) + 2];
      auto &outputGate = layers[(i * 4) + 3];

      // always connect input to inners
      auto inputConn = connect(inputNodes, memoryCell, ConnectionPattern::AllToAll);
      connect(inputNodes, inputGate, ConnectionPattern::AllToAll);
      connect(inputNodes, forgetGate, ConnectionPattern::AllToAll);
      connect(inputNodes, outputGate, ConnectionPattern::AllToAll);

      std::vector<std::reference_wrapper<Connection>> cell;
      if(previous != &inputNodes) {
        cell = connect(*previous, memoryCell, ConnectionPattern::AllToAll);
        connect(*previous, inputGate, ConnectionPattern::AllToAll);
        connect(*previous, forgetGate, ConnectionPattern::AllToAll);
        connect(*previous, outputGate, ConnectionPattern::AllToAll);
      }

      auto outputConn = connect(memoryCell, outputNodes, ConnectionPattern::AllToAll);
      auto forgetConn = connect(memoryCell, memoryCell, ConnectionPattern::OneToOne);

      connect(memoryCell, inputGate, ConnectionPattern::AllToAll);
      connect(memoryCell, forgetGate, ConnectionPattern::AllToAll);
      connect(memoryCell, outputGate, ConnectionPattern::AllToAll);

      gate(inputGate, inputConn, GatingPattern::Input);
      gate(forgetGate, forgetConn, GatingPattern::Self);
      gate(outputGate, outputConn, GatingPattern::Output);
      if(cell.size() > 0) {
        gate(inputGate, cell, GatingPattern::Input);
      }

      previous = &memoryCell;
    }

    // finally setup weights now that we know how many we need
    _weights.reserve(_connections.size());
    for (auto &conn : _connections) {
      auto &w = _weights.emplace_back(Random::normal(0.0, 1.0));
      conn.weight = &w;
    }
  }
};
} // namespace Nevolver

#endif /* LSTM_H */
