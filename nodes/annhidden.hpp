#ifndef ANNHIDDEN_H
#define ANNHIDDEN_H

#include "../nevolver.hpp"

namespace Nevolver {
class HiddenNode final : public NodeCommon<HiddenNode> {
public:
  HiddenNode(bool is_output = false, bool is_constant = false)
      : _is_output(is_output), _is_constant(is_constant) {}

  bool getIsOutput() const { return _is_output; }

  NeuroFloat doActivate() {
    _old = _state;

    if (_connections.self) {
      _state =
          _connections.self->gain * *_connections.self->weight * _state * _bias;
    } else {
      _state = _bias;
    }

    for (auto connection : _connections.inbound) {
      _state +=
          connection->from->current() * *connection->weight * connection->gain;
    }

    auto fwd = _squash(_state);
    _activation = fwd * _mask;
    _derivative = _derive(_state, fwd);

    _tmpNodes.clear();
    _tmpInfluence.clear();
    for (auto connection : _connections.gate) {
      auto node = connection->to;
      auto pos = std::find(std::begin(_tmpNodes), std::end(_tmpNodes), node);
      if (pos != std::end(_tmpNodes)) {
        auto idx = std::distance(std::begin(_tmpNodes), pos);
        _tmpInfluence[idx] += *connection->weight * connection->from->current();
      } else {
        _tmpNodes.emplace_back(node);
        auto plus =
            node->connections().self && node->connections().self->gater == this
                ? static_cast<const HiddenNode *>(node)->_old
                : 0.0;
        _tmpInfluence.emplace_back(
            *connection->weight * connection->from->current() + plus);
      }
      connection->gain = _activation;
    }

    for (auto connection : _connections.inbound) {
      if (_connections.self) {
        connection->eligibility =
            _connections.self->gain * *_connections.self->weight *
                connection->eligibility +
            connection->from->current() * connection->gain;
      } else {
        connection->eligibility =
            connection->from->current() * connection->gain;
      }

      auto size = _tmpNodes.size();
      for (size_t i = 0; i < size; i++) {
        auto node = _tmpNodes[i];
        auto influence = _tmpInfluence[i];
        auto pos = std::find(std::begin(connection->xtraces.nodes),
                             std::end(connection->xtraces.nodes), node);
        if (pos != std::end(connection->xtraces.nodes)) {
          auto idx = std::distance(std::begin(connection->xtraces.nodes), pos);
          if (node->connections().self) {
            connection->xtraces.values[idx] =
                node->connections().self->gain *
                    *node->connections().self->weight *
                    connection->xtraces.values[idx] +
                _derivative * connection->eligibility * influence;
          } else {
            connection->xtraces.values[idx] =
                _derivative * connection->eligibility * influence;
          }
        } else {
          connection->xtraces.nodes.emplace_back(node);
          connection->xtraces.values.emplace_back(
              _derivative * connection->eligibility * influence);
        }
      }
    }

    return _activation;
  }

  NeuroFloat doFastActivate() {
    _old = _state;

    if (_connections.self) {
      _state =
          _connections.self->gain * *_connections.self->weight * _state * _bias;
    } else {
      _state = _bias;
    }

    for (auto connection : _connections.inbound) {
      _state +=
          connection->from->current() * *connection->weight * connection->gain;
    }

    auto fwd = _squash(_state);
    _activation = fwd * _mask;

    for (auto connection : _connections.gate) {
      connection->gain = _activation;
    }

    return _activation;
  }

  void doPropagate(NeuroFloat rate, NeuroFloat momentum, bool update,
                   NeuroFloat target) {
    if (_is_output) {
      _responsibility = target - _activation;
      _projected = _responsibility;
    } else {
      NeuroFloat error = 0.0;

      for (auto connection : _connections.outbound) {
        error += connection->to->responsibility() * *connection->weight *
                 connection->gain;
      }
      _projected = _derivative * error;

      error = 0.0;

      for (auto connection : _connections.gate) {
        auto node = connection->to;
        NeuroFloat influence =
            node->connections().self && node->connections().self->gater == this
                ? static_cast<const HiddenNode *>(node)->_old
                : 0.0;
        influence += *connection->weight * connection->from->current();
        error += connection->to->responsibility() * influence;
      }

      _gated = _derivative * error;
      _responsibility = _projected + _gated;
    }

    if (_is_constant)
      return;

    for (auto connection : _connections.inbound) {
      auto gradient = _projected * connection->eligibility;

      // Gated nets only
      size_t size = _tmpNodes.size();
      for (size_t i = 0; i < size; i++) {
        auto node = _tmpNodes[i];
        auto value = _tmpInfluence[i];

        gradient += node->responsibility() * value;
      }

      auto deltaWeight = rate * gradient * _mask;
      if (update) {
        deltaWeight += momentum * connection->previousDeltaWeight;
        *connection->weight += deltaWeight;
        connection->previousDeltaWeight = deltaWeight;
      }
    }

    auto deltaBias = rate * _responsibility;
    deltaBias += momentum * _previousDeltaBias;
    _bias += deltaBias;
    _previousDeltaBias = deltaBias;
  }

  void setSquash(std::function<NeuroFloat(NeuroFloat)> squash,
                 std::function<NeuroFloat(NeuroFloat, NeuroFloat)> derive) {
    _squash = squash;
    _derive = derive;
  }

  void setBias(NeuroFloat bias) { _bias = bias; }

  void doClear() {
    for (auto &conn : _connections.inbound) {
      conn->eligibility = 0.0;
      conn->xtraces.nodes.clear();
      conn->xtraces.values.clear();
    }
    for (auto &conn : _connections.gate) {
      conn->gain = 0.0;
    }
    _responsibility = 0.0;
    _projected = 0.0;
    _gated = 0.0;
    _old = 0.0;
    _state = 0.0;
    _activation = 0.0;
  }

  void doMutate(NodeMutations mutation) {
    switch (mutation) {
    case Squash: {
      _squash = Squash::random();
    } break;
    case Bias: {
      _bias += Random::normal(0.0, 0.1);
    } break;
    }
  }

private:
  std::function<NeuroFloat(NeuroFloat)> _squash{SigmoidS()};
  std::function<NeuroFloat(NeuroFloat, NeuroFloat)> _derive{SigmoidD()};
  NeuroFloat _bias{Random::normal(0.0, 1.0)};
  NeuroFloat _state{0.0};
  NeuroFloat _old{0.0};
  NeuroFloat _mask{1.0};
  NeuroFloat _derivative{0.0};
  NeuroFloat _previousDeltaBias{0.0};
  bool _is_output;
  bool _is_constant;
  std::vector<const Node *> _tmpNodes;
  std::vector<NeuroFloat> _tmpInfluence;
  NeuroFloat _projected;
  NeuroFloat _gated;
};
} // namespace Nevolver

#endif /* ANNHIDDEN_H */
