/*
For quick-bench.com

Check assembly output, seems that ineheri + variants generate as good as
templated code! In clang generates more junk that taints the bench tho
*/

#include <iostream>
#include <memory>
#include <queue>
#include <variant>

using message_type = int;

namespace using_CRTP_and_variants {
template <typename T> struct actor {
  void update() { as_underlying().update(); }

  void handle_all_messages() { // internal states only
    while (!pending_messages.empty()) {
      auto message = std::move(pending_messages.front());
      pending_messages.pop();
      handle_one_message(std::move(message));
    }
  }

  void receive_message(message_type &&msg) {
    pending_messages.emplace(std::forward<message_type>(msg));
  }

private:
  friend T;
  actor() = default;

  std::queue<message_type> pending_messages;

  inline T &as_underlying() { return static_cast<T &>(*this); }
  inline T const &as_underlying() const {
    return static_cast<T const &>(*this);
  }

  void handle_one_message(message_type &&msg) {
    as_underlying().handle_one_message(std::forward<message_type>(msg));
  }
};

struct A : actor<A> {
  using actor::actor;

  void update() {
    // std::cout << "A : update()\n";
  }

private:
  friend struct actor<A>;

  void handle_one_message(message_type &&msg) {
    // std::cout << "A : handle_one_message : " << msg << '\n';
  }
};
struct B : actor<B> {
  using actor::actor;

  void update() {
    // std::cout << "B : update()\n";
  }

private:
  friend struct actor<B>;

  void handle_one_message(message_type &&msg) {
    // std::cout << "B : handle_one_message : " << msg << '\n';
  }
};
} // namespace using_CRTP_and_variants

namespace using_inheritance {
struct actor {
  virtual ~actor() = default;
  virtual void update() = 0;

  void handle_all_messages() { // internal states only
    while (!pending_messages.empty()) {
      auto message = std::move(pending_messages.front());
      pending_messages.pop();
      handle_one_message(std::move(message));
    }
  }

  void receive_message(message_type &&msg) {
    pending_messages.emplace(std::forward<message_type>(msg));
  }

private:
  std::queue<message_type> pending_messages;

  virtual void handle_one_message(message_type &&msg) = 0;
};

struct A : actor {
  void update() override {
    // std::cout << "A : update()\n";
  }
  void handle_one_message(message_type &&msg) override {
    // std::cout << "A : handle_one_message : " << msg << '\n';
  }
};
struct B : actor {
  void update() override {
    // std::cout << "B : update()\n";
  }
  void handle_one_message(message_type &&msg) override {
    // std::cout << "B : handle_one_message : " << msg << '\n';
  }
};
} // namespace using_inheritance

template <typename... Ts> using poly_T = std::variant<Ts...>;

static void test_CRTP_and_variants(benchmark::State &state) {

  using container_type = std::vector<
      poly_T<using_CRTP_and_variants::A, using_CRTP_and_variants::B>>;

  container_type actors{
      using_CRTP_and_variants::A{}, using_CRTP_and_variants::B{},
      using_CRTP_and_variants::A{}, using_CRTP_and_variants::B{},
      using_CRTP_and_variants::A{}, using_CRTP_and_variants::B{},
      using_CRTP_and_variants::A{}, using_CRTP_and_variants::B{},
      using_CRTP_and_variants::A{}, using_CRTP_and_variants::B{}};

  for (auto _ : state) {
    for (auto &active_actor : actors) { // broadcast messages ...
      std::visit(
          [](auto &act) {
            act.receive_message(41);
            act.receive_message(42);
            act.receive_message(43);
          },
          active_actor);
    }

    for (auto &active_actor : actors) {
      std::visit(
          [](auto &act) {
            act.update();
            act.handle_all_messages();
          },
          active_actor);
    }
    benchmark::DoNotOptimize(actors);
  }
}
// Register the function as a benchmark
BENCHMARK(test_CRTP_and_variants);

static void test_inheritance(benchmark::State &state) {

  using container_type = std::vector<std::unique_ptr<using_inheritance::actor>>;

  container_type actors;
  {
    actors.emplace_back(std::make_unique<using_inheritance::A>());
    actors.emplace_back(std::make_unique<using_inheritance::B>());
    actors.emplace_back(std::make_unique<using_inheritance::A>());
    actors.emplace_back(std::make_unique<using_inheritance::B>());
    actors.emplace_back(std::make_unique<using_inheritance::A>());
    actors.emplace_back(std::make_unique<using_inheritance::B>());
    actors.emplace_back(std::make_unique<using_inheritance::A>());
    actors.emplace_back(std::make_unique<using_inheritance::B>());
    actors.emplace_back(std::make_unique<using_inheritance::A>());
    actors.emplace_back(std::make_unique<using_inheritance::B>());
  }

  for (auto _ : state) {
    for (auto &active_actor : actors) { // broadcast messages ...
      active_actor->receive_message(41);
      active_actor->receive_message(42);
      active_actor->receive_message(43);
    }

    for (auto &active_actor : actors) {
      active_actor->update();
      active_actor->handle_all_messages();
    }
  }
}
BENCHMARK(test_inheritance);

static void test_inheri_and_variants(benchmark::State &state) {

  using container_type =
      std::vector<poly_T<using_inheritance::A, using_inheritance::B>>;

  container_type actors{using_inheritance::A{}, using_inheritance::B{},
                        using_inheritance::A{}, using_inheritance::B{},
                        using_inheritance::A{}, using_inheritance::B{},
                        using_inheritance::A{}, using_inheritance::B{},
                        using_inheritance::A{}, using_inheritance::B{}};

  for (auto _ : state) {
    for (auto &active_actor : actors) { // broadcast messages ...
      std::visit(
          [](auto &act) {
            act.receive_message(41);
            act.receive_message(42);
            act.receive_message(43);
          },
          active_actor);
    }

    for (auto &active_actor : actors) {
      std::visit(
          [](auto &act) {
            act.update();
            act.handle_all_messages();
          },
          active_actor);
    }
    benchmark::DoNotOptimize(actors);
  }
}
// Register the function as a benchmark
BENCHMARK(test_inheri_and_variants);
