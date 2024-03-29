#ifndef CONNECTIONS_H
#define CONNECTIONS_H

#include "nevolver.hpp"

namespace Nevolver {
enum ConnectionPattern { AllToAll, AllToElse, OneToOne };
enum GatingPattern { Input, Output, Self };

struct ConnectionXTraces {
  std::vector<const Node *> nodes;
  std::vector<NeuroFloat> values;
};

struct Connection final {
  const Node *from;
  const Node *to;
  const Node *gater;

  NeuroFloat gain{1};
  NeuroFloat eligibility{0};
  NeuroFloat previousDeltaWeight{0};

  mutable Weight *weight;

  NeuroFloat w() const { return weight->first; }

  ConnectionXTraces xtraces;
};

struct NodeConnections final {
  std::vector<Connection *> inbound;
  std::vector<Connection *> outbound;
  std::vector<Connection *> gate;
  Connection *self = nullptr;
};
} // namespace Nevolver

#endif /* CONNECTIONS_H */
