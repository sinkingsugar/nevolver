#include "../network.hpp"
#include "../networks/mlp.hpp"

// order matters!
INITIALIZE_EASYLOGGINGPP

#include "chainblocks.hpp"
#include "dllblock.hpp"

#include <sstream>

using namespace chainblocks;

namespace Nevolver {
struct SharedNetwork final {
  constexpr static auto Vendor = 'sink';
  constexpr static auto Type = 'nnet';

  static CBBool serialize(CBPointer pnet, uint8_t **outData, size_t *outLen,
                          CBPointer *handle) {
    LOG(DEBUG) << "SharedNetwork serialize";
    auto p = reinterpret_cast<SharedNetwork *>(pnet);
    auto buffer = new std::string();

    {
      std::stringstream ss;
      cereal::BinaryOutputArchive oa(ss);
      oa(*p->_holder.get());
      *buffer = ss.str();
    }

    *outData = (uint8_t *)buffer->data();
    *outLen = buffer->size();
    *handle = buffer;
    return true;
  }

  static void freeMem(CBPointer handle) {
    auto buffer = reinterpret_cast<std::string *>(handle);
    delete buffer;
  }

  static CBPointer deserialize(uint8_t *data, size_t len) {
    LOG(DEBUG) << "SharedNetwork deserialize";
    std::stringstream ss;
    ss.write((const char *)data, len);
    cereal::BinaryInputArchive ia(ss);
    auto net = new Network();
    ia(*net);
    auto sn = new SharedNetwork(net);
    return sn;
  }

  // Used when cloneVar is called around the chain
  static void addRef(CBPointer pnet) {
    auto p = reinterpret_cast<SharedNetwork *>(pnet);
    p->_refcount++;
    LOG(TRACE) << "Network refcount add: " << p->_refcount;
  }

  // Used when cloneVar is called around the chain
  static void decRef(CBPointer pnet) {
    auto p = reinterpret_cast<SharedNetwork *>(pnet);
    p->_refcount--;
    LOG(TRACE) << "Network refcount dec: " << p->_refcount;
    if (p->_refcount == 0) {
      LOG(TRACE)
          << "Releasing a Network reference; remaining use_count on holder: "
          << (p->_holder.use_count() - 1);
      delete p;
    }
  }

  static inline CBObjectInfo ObjInfo{"Nevolver.Network", &serialize, &freeMem,
                                     &deserialize,       &addRef,    &decRef};

  SharedNetwork(const std::shared_ptr<Network> &net)
      : _holder(net), _refcount(1) {}

  SharedNetwork(Network *net) : _holder(net), _refcount(1) {}

  ~SharedNetwork() { assert(_refcount == 0); }

  SharedNetwork(const SharedNetwork &other) = delete;
  SharedNetwork(const SharedNetwork &&other) = delete;
  SharedNetwork &operator=(const SharedNetwork &other) = delete;
  SharedNetwork &operator=(const SharedNetwork &&other) = delete;

  std::shared_ptr<Network> get() { return _holder; }

private:
  // Might be confusing!
  // but we actually have 2 RC mechanism
  // the reason is that we want to keep our networks detached from the chain
  // in order to do mutation/crossover etc
  std::shared_ptr<Network> _holder;
  uint32_t _refcount;
};

struct NetVar final : public CBVar {
  NetVar(const NetVar &other) = delete;
  NetVar(const NetVar &&other) = delete;
  NetVar &operator=(const NetVar &other) = delete;
  NetVar &operator=(const NetVar &&other) = delete;

  NetVar &operator=(const std::shared_ptr<Network> &other) {
    // this is from producer warmup
    // we want to cleanup older refs if any

    if (valueType == CBType::Object && payload.objectValue)
      SharedNetwork::decRef(payload.objectValue);

    valueType = CBType::Object;
    // notice that new SharedNetwork sets refcount to 1
    payload.objectValue = new SharedNetwork(other);
    payload.objectVendorId = SharedNetwork::Vendor;
    payload.objectTypeId = SharedNetwork::Type;
    objectInfo = &SharedNetwork::ObjInfo;
    flags |= CBVAR_FLAGS_USES_OBJINFO | CBVAR_FLAGS_REF_COUNTED;

    return *this;
  }

  NetVar &operator=(const CBVar &other) {
    // this is from setState
    // so we actually want to addRef

    if (valueType == CBType::Object && payload.objectValue)
      SharedNetwork::decRef(payload.objectValue);

    assert(other.valueType == Object);
    assert(other.payload.objectValue);
    assert(other.payload.objectVendorId == SharedNetwork::Vendor);
    assert(other.payload.objectTypeId == SharedNetwork::Type);
    valueType = Object;
    payload.objectValue = other.payload.objectValue;
    payload.objectVendorId = other.payload.objectVendorId;
    payload.objectTypeId = other.payload.objectTypeId;
    objectInfo = &SharedNetwork::ObjInfo;
    flags |= CBVAR_FLAGS_USES_OBJINFO | CBVAR_FLAGS_REF_COUNTED;

    SharedNetwork::addRef(payload.objectValue);

    return *this;
  }

  NetVar() = default;

  // explicit NetVar(const std::shared_ptr<Network> &net) : CBVar() {
  //   valueType = CBType::Object;
  //   // notice that new SharedNetwork sets refcount to 1
  //   payload.objectValue = new SharedNetwork(net);
  //   payload.objectVendorId = SharedNetwork::Vendor;
  //   payload.objectTypeId = SharedNetwork::Type;
  //   objectInfo = &SharedNetwork::ObjInfo;
  //   flags |= CBVAR_FLAGS_USES_OBJINFO | CBVAR_FLAGS_REF_COUNTED;
  // }

  explicit NetVar(const CBVar &other) : CBVar() {
    // Notice that we don't addRef here
    // as we mostly use the shared pointer behind it
    // we use this just as utility basically
    // and assertion
    assert(other.valueType == Object);
    assert(other.payload.objectValue);
    assert(other.payload.objectVendorId == SharedNetwork::Vendor);
    assert(other.payload.objectTypeId == SharedNetwork::Type);
    valueType = Object;
    payload.objectValue = other.payload.objectValue;
    payload.objectVendorId = other.payload.objectVendorId;
    payload.objectTypeId = other.payload.objectTypeId;
    objectInfo = &SharedNetwork::ObjInfo;
    flags |= CBVAR_FLAGS_USES_OBJINFO | CBVAR_FLAGS_REF_COUNTED;
  }

  std::shared_ptr<Network> get() {
    return reinterpret_cast<SharedNetwork *>(payload.objectValue)->get();
  }
};

struct NeuroVar final : public CBVar {
  // Converters

  NeuroVar(const NeuroFloat &f) : CBVar() {
    valueType = Float;
    // TODO other cases were we want actual vectors
    // when compiled as WIDE
    payload.floatValue = mean(f);
  }

  operator NeuroFloat() const {
    assert(valueType == Float);
    return payload.floatValue;
  }
};

using NeuroVars =
    IterableArray<CBSeq, NeuroVar, &Core::seqResize, &Core::seqFree>;

struct NeuroSeq : public CBVar {
  NeuroSeq(std::vector<NeuroVar> &vec) : CBVar() {
    valueType = Seq;
    payload.seqValue.elements = &vec[0];
    payload.seqValue.len = uint32_t(vec.size());
    payload.seqValue.cap = 0;
  }
};

static Type IntType{{CBType::Int}};
static Type IntSeq{{CBType::Seq, .seqTypes = IntType}};
static Type AnyType{{CBType::Any}};
static Type StringType{{CBType::String}};
static Type FloatType{{CBType::Float}};
static Type FloatSeq{{CBType::Seq, .seqTypes = FloatType}};
static Type NetType{
    {CBType::Object, .object = {SharedNetwork::Vendor, SharedNetwork::Type}}};
static Type NetVarType{{CBType::ContextVar, .contextVarTypes = NetType}};

static Parameters CommonParams{
    {"Name", "The network model variable.", {NetVarType, StringType}}};
static Parameters PropParams{
    CommonParams,
    {{"Rate", "The number of input nodes.", {FloatType}},
     {"Momentum",
      "The number of hidden nodes, can be a sequence for multiple layers.",
      {FloatType}}}};
static Parameters MlpParams{
    CommonParams,
    {{"Inputs", "The number of input nodes.", {IntType}},
     {"Hidden",
      "The number of hidden nodes, can be a sequence for multiple layers.",
      {{IntType, IntSeq}}},
     {"Outputs", "The number of output nodes.", {IntType}}}};

// Problems
// We want to train this with genetic
// so if we release networks on cleanup it's bad

// TODO Add exposed/required variables

struct NetworkUser {
  ~NetworkUser() { LOG(TRACE) << "NetworkUser destroy."; }

  static CBTypesInfo inputTypes() { return AnyType; }
  static CBTypesInfo outputTypes() { return AnyType; }
  static CBParametersInfo parameters() { return CommonParams; }

  void setParam(int index, CBVar value) { _netParam = value; }

  CBVar getParam(int index) { return _netParam; }

  ParamVar _netParam{};
  // we keep a ref here, in order to keep alive even
  // after cleanups
  std::shared_ptr<Network> _netRef;

  void warmup(CBContext *context) { _netParam.warmup(context); }

  void cleanup() { _netParam.cleanup(); }
};

struct NetworkConsumer : public NetworkUser {
  static CBTypesInfo inputTypes() { return FloatSeq; }

  void warmup(CBContext *context) {
    NetworkUser::warmup(context);
    // Consumers basically don't care about
    // SharedNetwork refcount
    // here we use a temporary to write directly
    // our shared_ptr
    _netRef = NetVar(_netParam.get()).get();
  }
};

struct NetworkProducer : public NetworkUser {
  NetVar _state{};

  ~NetworkProducer() {
    if (_state.valueType == Object)
      SharedNetwork::decRef(_state.payload.objectValue);
  }

  CBVar activate(CBContext *context, const CBVar &input) { return input; }

  CBVar getState() { return _state; }

  void setState(CBVar state) {
    if (state.valueType == None)
      return;
    _state = state;
    _netRef = _state.get();
  }

  void mutate(CBTable options) {
    const std::vector<NetworkMutations> muts{
        NetworkMutations::AddNode,          NetworkMutations::SubNode,
        NetworkMutations::AddFwdConnection, NetworkMutations::AddBwdConnection,
        NetworkMutations::SubConnection,    NetworkMutations::ShareWeight,
        NetworkMutations::SwapNodes,        NetworkMutations::AddGate,
        NetworkMutations::SubGate};

    const std::vector<NodeMutations> nmuts{NodeMutations::Squash,
                                           NodeMutations::Bias};

    _netRef->mutate(muts, 0.2, nmuts, 0.2, 0.2);
  }

  void crossover(CBVar parent1, CBVar parent2) {
    // crossover happens before warmup
    // so all we gotta do is assign to the shared ptr (deref)
    // this will propagate to other activate etc blocks
    // once warmup is called
    auto p1 = NetVar(parent1).get();
    auto p2 = NetVar(parent2).get();
    *_netRef.get() = Network::crossover(*p1.get(), *p2.get());
    // Also update the state cos state is used in getState
    // by other crossovers
    _state = _netRef;
  }

  void cleanup() {
    NetworkUser::cleanup();

    // Do an implicit clear on the network
    if (_netRef) {
      _netRef->clear();
    }
  }

  // actual netref construction here
  virtual void resetState() = 0;

  void warmup(CBContext *context) {
    NetworkUser::warmup(context);

    // assert we are the creators of this network
    assert(_netParam.get().valueType == CBType::None);

    if (!_netRef) {
      // create the network if it never existed yet
      resetState();
    }

    // ensure addref
    // and that _state is up to date
    _state = _netRef;
    // one ref will be removed on cleanup
    // so we need to add one
    // that will be removed if reassigned or destroied
    SharedNetwork::addRef(_state.payload.objectValue);

    // it will be decRef-ed by destroyVar
    _netParam.get() = _state;
    _netParam.get().refcount++;
  }
};

struct Activate : public NetworkConsumer {
  static CBTypesInfo outputTypes() { return FloatSeq; }

  std::vector<NeuroVar> _outputCache;

  CBVar activate(CBContext *context, const CBVar &input) {
    // NO-Copy activate
    NeuroVars in(input);
    _netRef->activate(in, _outputCache);
    return NeuroSeq(_outputCache);
  }
};

struct Predict final : public Activate {
  CBVar activate(CBContext *context, const CBVar &input) {
    // NO-Copy activate
    NeuroVars in(input);
    _netRef->activateFast(in, _outputCache);
    return NeuroSeq(_outputCache);
  }
};

struct Propagate : public NetworkConsumer {
  CBParametersInfo parameters() { return PropParams; }

  void setParam(int index, CBVar value) {
    switch (index) {
    case 0:
      NetworkUser::setParam(index, value);
      break;
    case 1:
      _rate = value.payload.floatValue;
      break;
    case 2:
      _momentum = value.payload.floatValue;
      break;
    default:
      break;
    }
  }

  CBVar getParam(int index) {
    switch (index) {
    case 0:
      return NetworkUser::getParam(index);
    case 1:
      return Var(_rate);
    case 2:
      return Var(_momentum);
      break;
    default:
      return CBVar();
    }
  }

  CBTypesInfo outputTypes() { return FloatType; }

  double _rate = 0.3;
  double _momentum = 0.0;

  CBVar activate(CBContext *context, const CBVar &input) {
    // NO-Copy propagate
    NeuroVars in(input);
    return _netRef->propagate<NeuroVar>(in, _rate, _momentum, true);
  }
};

struct MLPBlock final : public NetworkProducer {
  CBParametersInfo parameters() { return MlpParams; }

  void setParam(int index, CBVar value) {
    switch (index) {
    case 0:
      NetworkUser::setParam(index, value);
      break;
    case 1:
      _inputs = int(value.payload.intValue);
      break;
    case 2:
      _hidden = value;
      break;
    case 3:
      _outputs = int(value.payload.intValue);
      break;
    default:
      break;
    }
  }

  CBVar getParam(int index) {
    switch (index) {
    case 0:
      return NetworkUser::getParam(index);
    case 1:
      return Var(_inputs);
    case 2:
      return Var(_hidden);
      break;
    case 3:
      return Var(_outputs);
    default:
      return CBVar();
    }
  }

  int _inputs = 2;
  OwnedVar _hidden{Var(4)};
  int _outputs = 1;

  void resetState() override {
    std::vector<int> hiddens;
    if (_hidden.valueType == Int) {
      hiddens.push_back(int(_hidden.payload.intValue));
    } else if (_hidden.valueType == Seq) {
      IterableSeq s(_hidden);
      for (auto &n : s) {
        hiddens.push_back(int(n.payload.intValue));
      }
    }
    _netRef = std::make_shared<MLP>(_inputs, hiddens, _outputs);
  }
};
}; // namespace Nevolver

namespace chainblocks {
void registerBlocks() {
  REGISTER_CBLOCK("Nevolver.Activate", Nevolver::Activate);
  REGISTER_CBLOCK("Nevolver.Predict", Nevolver::Predict);
  REGISTER_CBLOCK("Nevolver.Propagate", Nevolver::Propagate);
  REGISTER_CBLOCK("Nevolver.MLP", Nevolver::MLPBlock);

  // TODO
  // Add .Clear block
}
} // namespace chainblocks

// Compiling with mingw compilers we need this trick to make sure
// no symbols are exported
#ifdef _WIN32
extern "C" {
__declspec(dllexport) void hiddenSymbols() {}
};
#endif
