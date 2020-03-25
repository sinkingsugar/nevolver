#include "network.hpp"
#include "networks/lstm.hpp"
#include "networks/narx.hpp"
#include "networks/perceptron.hpp"
#include <sstream>

namespace Nevolver {} // namespace Nevolver

int main() {
  std::cout << "Hello!\n";

  Nevolver::SigmoidS func;
  std::cout << func(Nevolver::NeuroFloatOnes) << "\n";

  Nevolver::HiddenNode node;
  std::cout << node.activate() << "\n";

#ifndef NEVOLVER_WIDE
  {
    auto perceptron = Nevolver::Perceptron(2, {4, 4}, 1);
    for (auto i = 0; i < 50000; i++) {
      perceptron.activate({0.0, 0.0});
      perceptron.propagate({1.0});
      perceptron.activate({0.0, 1.0});
      perceptron.propagate({0.0});
      perceptron.activate({1.0, 0.0});
      perceptron.propagate({0.0});
      perceptron.activate({1.0, 1.0});
      auto err = perceptron.propagate({1.0});
      if (!(i % 10000))
        std::cout << "MSE: " << err << "\n";
    }
    std::cout << perceptron.activate({0.0, 0.0})[0] << " (1.0)\n";
    std::cout << perceptron.activate({0.0, 1.0})[0] << " (0.0)\n";
    std::cout << perceptron.activate({1.0, 0.0})[0] << " (0.0)\n";
    std::cout << perceptron.activate({1.0, 1.0})[0] << " (1.0)\n";
  }

  {
    auto perceptron = Nevolver::NARX(2, {4, 2}, 1, 4, 4);
    for (auto i = 0; i < 5000; i++) {
      perceptron.activate({0.0, 0.0});
      perceptron.propagate({1.0});
      perceptron.activate({0.0, 1.0});
      perceptron.propagate({0.0});
      perceptron.activate({1.0, 0.0});
      perceptron.propagate({0.0});
      perceptron.activate({1.0, 1.0});
      auto err = perceptron.propagate({1.0});
      if (!(i % 1000))
        std::cout << "MSE: " << err << "\n";
    }
    std::cout << perceptron.activate({0.0, 0.0})[0] << " (1.0)\n";
    std::cout << perceptron.activate({0.0, 1.0})[0] << " (0.0)\n";
    std::cout << perceptron.activate({1.0, 0.0})[0] << " (0.0)\n";
    std::cout << perceptron.activate({1.0, 1.0})[0] << " (1.0)\n";
  }

  {
    auto lstm = Nevolver::LSTM(1, {6}, 1);
    for (auto i = 0; i < 25000; i++) {
      lstm.activate({0.0});
      lstm.propagate({0.0});
      lstm.activate({0.0});
      lstm.propagate({0.0});
      lstm.activate({0.0});
      lstm.propagate({1.0});
      lstm.activate({1.0});
      lstm.propagate({0.0});
      lstm.activate({0.0});
      lstm.propagate({0.0});
      lstm.activate({0.0});
      auto err = lstm.propagate({1.0});
      if (!(i % 10000))
        std::cout << "MSE: " << err << "\n";
      lstm.clear();
    }

    std::cout << lstm.activate({0.0})[0] << " (0.0)\n";
    std::cout << lstm.activate({0.0})[0] << " (0.0)\n";
    std::cout << lstm.activate({0.0})[0] << " (1.0)\n";
    std::cout << lstm.activate({1.0})[0] << " (0.0)\n";
    std::cout << lstm.activate({0.0})[0] << " (0.0)\n";
    std::cout << lstm.activate({0.0})[0] << " (1.0)\n";

    struct Writer {
      std::stringstream s;
      void operator()(uint8_t *data, size_t size) {
        s.write((const char *)data, size);
      }
    } w;
    lstm.serialize(w);

    struct Reader {
      Writer &w;
      void operator()(uint8_t *data, size_t size) {
        w.s.read((char *)data, size);
      }
    } r{w};
    auto lstm2 = Nevolver::Network::deserialize(r);
  }

  {
    for (auto i = 0; i < 1000; i++) {
      auto r = Nevolver::Random::next();
      if (r < 0.0 || r > 1.0) {
        throw std::runtime_error("Random::next test failed...");
      }
    }
  }
#else
  {
    auto perceptron = Nevolver::Perceptron(2, {4, 4}, 1);
    for (auto i = 0; i < 50000; i++) {
      perceptron.activate(
          {Nevolver::NeuroFloatZeros, Nevolver::NeuroFloatZeros});
      perceptron.propagate({Nevolver::NeuroFloatOnes});
      perceptron.activate(
          {Nevolver::NeuroFloatZeros, Nevolver::NeuroFloatOnes});
      perceptron.propagate({Nevolver::NeuroFloatZeros});
      perceptron.activate(
          {Nevolver::NeuroFloatOnes, Nevolver::NeuroFloatZeros});
      perceptron.propagate({Nevolver::NeuroFloatZeros});
      perceptron.activate({Nevolver::NeuroFloatOnes, Nevolver::NeuroFloatOnes});
      auto err = perceptron.propagate({Nevolver::NeuroFloatOnes});
      if (!(i % 10000))
        std::cout << "MSE: " << err << "\n";
    }
    std::cout << perceptron.activate(
                     {Nevolver::NeuroFloatZeros, Nevolver::NeuroFloatZeros})[0]
              << " (1.0)\n";
    std::cout << perceptron.activate(
                     {Nevolver::NeuroFloatZeros, Nevolver::NeuroFloatOnes})[0]
              << " (0.0)\n";
    std::cout << perceptron.activate(
                     {Nevolver::NeuroFloatOnes, Nevolver::NeuroFloatZeros})[0]
              << " (0.0)\n";
    std::cout << perceptron.activate(
                     {Nevolver::NeuroFloatOnes, Nevolver::NeuroFloatOnes})[0]
              << " (1.0)\n";
  }
#endif

  return 0;
}
