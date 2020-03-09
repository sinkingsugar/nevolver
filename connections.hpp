#ifndef CONNECTIONS_H
#define CONNECTIONS_H

#include "nevolver.hpp"

namespace Nevolver {
enum ConnectionPattern { AllToAll, AllToElse, OneToOne };
enum GatingPattern { Input, Output, Self };
enum ConnectionMutations { Weight };

struct ConnectionXTraces {
  std::vector<const Node *> nodes;
  std::vector<NeuroFloat> values;
};

struct Connection final {
  const Node *from;
  const Node *to;
  const Node *gater;

  NeuroFloat gain{1.0};
  NeuroFloat eligibility{0.0};
  NeuroFloat previousDeltaWeight{0.0};

  NeuroFloat *weight;

  ConnectionXTraces xtraces;

  void mutate(const std::vector<ConnectionMutations> &allowed) {}
};

struct NodeConnections final {
  std::vector<Connection *> inbound;
  std::vector<Connection *> outbound;
  std::vector<Connection *> gate;
  Connection *self = nullptr;
};
} // namespace Nevolver

#endif /* CONNECTIONS_H */
