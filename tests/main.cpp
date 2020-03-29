#include "../network.hpp"
#include "../networks/lstm.hpp"
#include "../networks/narx.hpp"
#include "../networks/perceptron.hpp"
#include <fstream>
#include <iostream>

#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

INITIALIZE_EASYLOGGINGPP

TEST_CASE("Perceptron SGD training and serialize", "[perceptron1]") {
  auto perceptron = Nevolver::Perceptron(2, {4, 4}, 1);
  for (auto i = 0; i < 50000; i++) {
    perceptron.activate({0.0, 0.0});
    perceptron.propagate({1.0});
    perceptron.activate({0.0, 1.0});
    perceptron.propagate({0.0});
    perceptron.activate({1.0, 0.0});
    perceptron.propagate({0.0});
    perceptron.activate({1.0, 1.0});
    perceptron.propagate({1.0});
  }

  auto res1 = perceptron.activate({0.0, 0.0})[0];
  auto res2 = perceptron.activate({0.0, 1.0})[0];
  auto res3 = perceptron.activate({1.0, 0.0})[0];
  auto res4 = perceptron.activate({1.0, 1.0})[0];

  {
    std::ofstream os("nn.cereal", std::ios::binary);
    cereal::BinaryOutputArchive oa(os);
    oa(perceptron);
  }

  {
    std::ifstream is("nn.cereal", std::ios::binary);
    cereal::BinaryInputArchive ia(is);
    Nevolver::Network perceptron2;
    ia(perceptron2);

    REQUIRE(perceptron2.activate({0.0, 0.0})[0] == Approx(res1));
    REQUIRE(perceptron2.activate({0.0, 1.0})[0] == Approx(res2));
    REQUIRE(perceptron2.activate({1.0, 0.0})[0] == Approx(res3));
    REQUIRE(perceptron2.activate({1.0, 1.0})[0] == Approx(res4));
  }
}

TEST_CASE("Perceptron test vectors", "[perceptronv]") {
  auto perceptron = Nevolver::Perceptron(2, {4, 3}, 1);

  for (auto &w : perceptron.weights()) {
    w.first = 0.3;
  }
  for (auto &n : perceptron.nodes()) {
    try {
      auto &node = n.get();
      auto &hn = std::get<Nevolver::HiddenNode>(node);
      // hn.setSquash(Nevolver::IdentityS(), Nevolver::IdentityD());
      hn.setBias(0.2);
    } catch (...) {
    }
  }

  REQUIRE(perceptron.activate({1.0, 0.0})[0] == Approx(0.7002422007360097));
  REQUIRE(perceptron.activateFast({1.0, 0.0})[0] == Approx(0.7002422007360097));
  perceptron.propagate({1.0});
  REQUIRE(perceptron.activate({0.0, 0.0})[0] == Approx(0.7431723131333033));
  REQUIRE(perceptron.activateFast({0.0, 0.0})[0] == Approx(0.7431723131333033));
  perceptron.propagate({1.0});
  REQUIRE(perceptron.activate({0.0, 1.0})[0] == Approx(0.7827034965579912));
  REQUIRE(perceptron.activateFast({0.0, 1.0})[0] == Approx(0.7827034965579912));
  perceptron.propagate({1.0});
  REQUIRE(perceptron.activate({1.0, 1.0})[0] == Approx(0.8141762974838022));
  REQUIRE(perceptron.activateFast({1.0, 1.0})[0] == Approx(0.8141762974838022));
}

TEST_CASE("NARX test vectors", "[narxv]") {
  auto narx = Nevolver::NARX(2, {4, 3}, 1, 3, 3);

  for (auto &w : narx.weights()) {
    w.first = 0.3;
  }
  for (auto &n : narx.nodes()) {
    try {
      auto &node = n.get();
      auto &hn = std::get<Nevolver::HiddenNode>(node);
      // hn.setSquash(Nevolver::IdentityS(), Nevolver::IdentityD());
      hn.setBias(0.2);
    } catch (...) {
    }
  }

  REQUIRE(narx.activate({1.0, 0.0})[0] == Approx(0.7021026153497725));
  narx.propagate({1.0});
  REQUIRE(narx.activate({0.0, 0.0})[0] == Approx(0.7516167046460288));
  narx.propagate({1.0});
  REQUIRE(narx.activate({0.0, 1.0})[0] == Approx(0.7906491534319081));
  narx.propagate({1.0});
  REQUIRE(narx.activate({1.0, 1.0})[0] == Approx(0.8212506297527332));
}

TEST_CASE("LSTM test vectors", "[lstmv]") {
  auto lstm = Nevolver::LSTM(2, {4, 2}, 1);

  for (auto &w : lstm.weights()) {
    w.first = 0.3;
  }
  for (auto &n : lstm.nodes()) {
    try {
      auto &node = n.get();
      auto &hn = std::get<Nevolver::HiddenNode>(node);
      hn.setBias(0.2);
    } catch (...) {
    }
  }

  REQUIRE(lstm.activate({1.0, 0.0})[0] == Approx(0.7021348896263643));
  lstm.propagate({1.0});
  REQUIRE(lstm.activate({0.0, 0.0})[0] == Approx(0.672610483084914));
  lstm.propagate({1.0});
  REQUIRE(lstm.activate({0.0, 1.0})[0] == Approx(0.7745991463085304));
  lstm.propagate({1.0});
  REQUIRE(lstm.activate({1.0, 1.0})[0] == Approx(0.8661533781635797));

  auto stats = lstm.getStats();
  REQUIRE(stats.activeNodes == 31);
  REQUIRE(stats.activeConnections == 154);
  REQUIRE(stats.activeWeights == 154);
  REQUIRE(stats.unusedNodes == 0);
  REQUIRE(stats.unusedConnections == 0);
  REQUIRE(stats.unusedWeights == 0);

  auto narx = Nevolver::NARX(2, {4, 3}, 1, 3, 3);
  stats = narx.getStats();
  REQUIRE(stats.activeNodes == 19);
  REQUIRE(stats.activeConnections == 68);
  REQUIRE(stats.activeWeights == 68);
  REQUIRE(stats.unusedNodes == 0);
  REQUIRE(stats.unusedConnections == 0);
  REQUIRE(stats.unusedWeights == 0);

  auto childLstm = Nevolver::Network::crossover(lstm, narx);
  stats = childLstm.getStats();
  REQUIRE(stats.activeNodes == 19);
  REQUIRE(stats.activeConnections == 68);
  REQUIRE(stats.activeWeights == 68);
  REQUIRE(stats.unusedNodes == 0);
  REQUIRE(stats.unusedConnections == 0);
  REQUIRE(stats.unusedWeights == 0);

  auto &nodes = lstm.nodes();
  auto nit = nodes.begin();
  while (nit != nodes.end()) {
    nit = lstm.removeNode(nit);
  }

  stats = lstm.getStats();
  REQUIRE(stats.activeNodes == 3);
  REQUIRE(stats.activeConnections == 2);
  REQUIRE(stats.activeWeights == 2);
  REQUIRE(stats.unusedNodes == 28);
  REQUIRE(stats.unusedConnections == 152);
  REQUIRE(stats.unusedWeights == 152);

  lstm.mutate({Nevolver::NetworkMutations::AddNode}, 1.0, {}, 0.0, 0.0);

  stats = lstm.getStats();
  REQUIRE(stats.activeNodes == 4);
  REQUIRE(stats.activeConnections == 3);
  REQUIRE(stats.activeWeights == 3);
  REQUIRE(stats.unusedNodes == 27);
  REQUIRE(stats.unusedConnections == 151);
  REQUIRE(stats.unusedWeights == 151);

  lstm.mutate({Nevolver::NetworkMutations::AddFwdConnection,
               Nevolver::NetworkMutations::AddBwdConnection},
              1.0, {}, 0.0, 0.0);

  stats = lstm.getStats();
  REQUIRE(stats.activeNodes == 4);
  REQUIRE(stats.activeConnections == 5);
  REQUIRE(stats.activeWeights == 5);
  REQUIRE(stats.unusedNodes == 27);
  REQUIRE(stats.unusedConnections == 149);
  REQUIRE(stats.unusedWeights == 149);

  lstm.mutate({Nevolver::NetworkMutations::SubNode,
               Nevolver::NetworkMutations::AddNode},
              1.0, {}, 0.0, 0.0);

  stats = lstm.getStats();
  REQUIRE(stats.activeNodes == 4);
  REQUIRE(stats.activeConnections == 4);
  REQUIRE(stats.activeWeights == 4);
  REQUIRE(stats.unusedNodes == 27);
  REQUIRE(stats.unusedConnections == 150);
  REQUIRE(stats.unusedWeights == 150);

  lstm.mutate({Nevolver::NetworkMutations::SubConnection,
               Nevolver::NetworkMutations::ShareWeight,
               Nevolver::NetworkMutations::SwapNodes},
              1.0, {}, 0.0, 0.0);

  stats = lstm.getStats();
  REQUIRE(stats.activeNodes == 4);
  REQUIRE(stats.activeConnections == 3);
  REQUIRE(stats.activeWeights == 2);
  REQUIRE(stats.unusedNodes == 27);
  REQUIRE(stats.unusedConnections == 151);
  REQUIRE(stats.unusedWeights == 152);

  childLstm = Nevolver::Network::crossover(lstm, narx);
  stats = childLstm.getStats();
  REQUIRE(stats.activeNodes == 19);
  REQUIRE(stats.activeConnections == 68);
  REQUIRE(stats.activeWeights == 68);
  REQUIRE(stats.unusedNodes == 0);
  REQUIRE(stats.unusedConnections == 0);
  REQUIRE(stats.unusedWeights == 0);
}

TEST_CASE("NARX SGD training", "[narx1]") {
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

TEST_CASE("LSTM SGD training", "[lstm1]") {
  auto lstm = Nevolver::LSTM(1, {4}, 1);
  for (auto i = 0; i < 50000; i++) {
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

  std::cout << "LSTM0: \n";
  std::cout << lstm.activate({0.0})[0] << " (0.0)\n";
  std::cout << lstm.activate({0.0})[0] << " (0.0)\n";
  std::cout << lstm.activate({0.0})[0] << " (1.0)\n";
  std::cout << lstm.activate({1.0})[0] << " (0.0)\n";
  std::cout << lstm.activate({0.0})[0] << " (0.0)\n";
  std::cout << lstm.activate({0.0})[0] << " (1.0)\n";

  {
    std::ofstream os("nn.cereal", std::ios::binary);
    cereal::BinaryOutputArchive oa(os);
    oa(lstm);
  }

  {
    std::ifstream is("nn.cereal", std::ios::binary);
    cereal::BinaryInputArchive ia(is);
    Nevolver::Network lstm2;
    ia(lstm2);

    std::cout << "LSTM1: \n";
    std::cout << lstm2.activate({0.0})[0] << " (0.0)\n";
    std::cout << lstm2.activate({0.0})[0] << " (0.0)\n";
    std::cout << lstm2.activate({0.0})[0] << " (1.0)\n";
    std::cout << lstm2.activate({1.0})[0] << " (0.0)\n";
    std::cout << lstm2.activate({0.0})[0] << " (0.0)\n";
    std::cout << lstm2.activate({0.0})[0] << " (1.0)\n";
  }
}

TEST_CASE("Test rng", "[rng]") {
  for (auto i = 0; i < 1000; i++) {
    auto r = Nevolver::Random::next();
    if (r < 0.0 || r > 1.0) {
      throw std::runtime_error("Random::next test failed...");
    }
  }
}
