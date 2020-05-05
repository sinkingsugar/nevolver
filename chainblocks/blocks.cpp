// order matters!
#include <easylogging++.h>
INITIALIZE_EASYLOGGINGPP

#include "../network.hpp"
#include "../networks/liquid.hpp"
#include "../networks/lstm.hpp"
#include "../networks/mlp.hpp"
#include "../networks/narx.hpp"

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
    LOG(TRACE) << "Network refcount add: " << p->_refcount
               << " SharedNetwork: " << p;
  }

  // Used when cloneVar is called around the chain
  static void decRef(CBPointer pnet) {
    auto p = reinterpret_cast<SharedNetwork *>(pnet);
    p->_refcount--;
    LOG(TRACE) << "Network refcount dec: " << p->_refcount
               << " SharedNetwork: " << p;
    if (p->_refcount == 0) {
      LOG(DEBUG) << "Releasing a Network reference: " << p->_holder.get()
                 << " use_count: " << (p->_holder.use_count() - 1)
                 << " SharedNetwork: " << p;
      delete p;
    }
  }

  static inline CBObjectInfo ObjInfo{"Nevolver.Network", &serialize, &freeMem,
                                     &deserialize,       &addRef,    &decRef};

  SharedNetwork(const std::shared_ptr<Network> &net)
      : _holder(net), _refcount(1) {
    LOG(TRACE) << "SharedNetwork _holder: " << _holder.get()
               << " use_count: " << _holder.use_count()
               << " SharedNetwork: " << this;
  }

  SharedNetwork(Network *net) : _holder(net), _refcount(0) {
    LOG(TRACE) << "SharedNetwork _holder: " << _holder.get()
               << " use_count: " << _holder.use_count()
               << " SharedNetwork: " << this;
  }

  ~SharedNetwork() {
    assert(_refcount == 0);
    LOG(TRACE) << "~SharedNetwork _holder: " << _holder.get()
               << " use_count: " << (_holder.use_count() - 1)
               << " SharedNetwork: " << this;
  }

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
    flags |= CBVAR_FLAGS_USES_OBJINFO;

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
    flags |= CBVAR_FLAGS_USES_OBJINFO;

    SharedNetwork::addRef(payload.objectValue);

    return *this;
  }

  NetVar() = default;

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
    flags |= CBVAR_FLAGS_USES_OBJINFO;
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
    // Also accept Ints, for example a CB Int8 is 16 bytes
    // perfectly useful if assumed to be a image input
    // we can directly on implicit conversion normalize /255
    // and use Wide 16!
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

static Type NoneType{{CBType::None}};
static Type IntType{{CBType::Int}};
static Type BytesType{{CBType::Bytes}};
static Type IntSeq{{CBType::Seq, {.seqTypes = IntType}}};
static Type AnyType{{CBType::Any}};
static Type StringType{{CBType::String}};
static Type FloatType{{CBType::Float}};
static Type FloatSeq{{CBType::Seq, {.seqTypes = FloatType}}};
static Type NetType{
    {CBType::Object, {.object = {SharedNetwork::Vendor, SharedNetwork::Type}}}};
static Type NetVarType{{CBType::ContextVar, {.contextVarTypes = NetType}}};

static Parameters CommonParams{
    {"Name", "The network model variable.", {NetVarType}}};
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
static Parameters LiquidParams{
    CommonParams,
    {{"Inputs", "The number of input nodes.", {IntType}},
     {"Hidden", "The number of max starting hidden nodes.", {{IntType}}},
     {"Outputs", "The number of output nodes.", {IntType}}}};
static Parameters NarxParams{
    CommonParams,
    {{"Inputs", "The number of input nodes.", {IntType}},
     {"Hidden",
      "The number of hidden nodes, can be a sequence for multiple layers.",
      {{IntType, IntSeq}}},
     {"Outputs", "The number of output nodes.", {IntType}},
     {"InputMemory", "The number of inputs to memorize.", {IntType}},
     {"OutputMemory", "The number of outputs to memorize.", {IntType}}}};

// Problems
// We want to train this with genetic
// so if we release networks on cleanup it's bad

struct NetworkUser {
  ~NetworkUser() {
    if (_netRef) {
      LOG(DEBUG) << "NetworkUser Network destroy, network: " << _netRef.get()
                 << " use_count: " << (_netRef.use_count() - 1);
    }
  }

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

protected:
  CBExposedTypeInfo _expInfo{};
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

  CBExposedTypesInfo requiredVariables() {
    if (_netParam.isVariable()) {
      _expInfo = CBExposedTypeInfo{_netParam.variableName(),
                                   "The required neural network.", NetType};
    } else {
      throw ComposeError("No network name specified.");
    }
    return CBExposedTypesInfo{&_expInfo, 1, 0};
  }
};

struct NetworkProducer : NetworkUser {
  NetVar _state{};

  ~NetworkProducer() {
    if (_state.valueType == Object)
      SharedNetwork::decRef(_state.payload.objectValue);
  }

  CBExposedTypesInfo exposedVariables() {
    if (_netParam.isVariable()) {
      _expInfo = CBExposedTypeInfo{_netParam.variableName(),
                                   "The exposed neural network.", NetType};
    } else {
      throw ComposeError("No network name specified.");
    }
    return CBExposedTypesInfo{&_expInfo, 1, 0};
  }

  CBTypeInfo compose(const CBInstanceData &data) {
    data.block->inlineBlockId = NoopBlock;
    return data.inputType;
  }

  CBVar activate(CBContext *context, const CBVar &input) {
    throw ActivationError("NoopBlock activation called.");
  }

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

struct Clear final : public NetworkConsumer {
  static CBTypesInfo inputTypes() { return AnyType; }
  static CBTypesInfo outputTypes() { return AnyType; }

  CBVar activate(CBContext *context, const CBVar &input) {
    _netRef->clear();
    return input;
  }
};

struct Propagate final : public NetworkConsumer {
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
      return Var::Empty;
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

struct SaveModel final : public NetworkConsumer {
  static CBTypesInfo inputTypes() { return NoneType; }
  static CBTypesInfo outputTypes() { return BytesType; }

  CBVar activate(CBContext *context, const CBVar &input) {
    _buffer.clear();
    std::stringstream ss;
    cereal::BinaryOutputArchive oa(ss);
    oa(*_netRef);
    _buffer = ss.str();
    return Var((uint8_t *)_buffer.data(), _buffer.length());
  }

private:
  std::string _buffer;
};

struct LoadModel final : public NetworkProducer {
  static CBTypesInfo inputTypes() { return BytesType; }
  static CBTypesInfo outputTypes() { return NoneType; }

  void resetState() override {
    LOG(TRACE) << "Loading a model, creating foo network!";
    // we need a stub network here anyway
    _netRef = std::make_shared<Network>();
  }

  CBTypeInfo compose(const CBInstanceData &data) {
    // override noop behavior
    return data.inputType;
  }

  CBVar activate(CBContext *context, const CBVar &input) {
    LOG(TRACE) << "Loading a model! activate";
    std::stringstream ss;
    ss.write((const char *)input.payload.bytesValue, input.payload.bytesSize);
    cereal::BinaryInputArchive ia(ss);
    ia(*_netRef);
    return Var::Empty;
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
      return Var::Empty;
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

struct LiquidBlock final : public NetworkProducer {
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
      _hidden = int(value.payload.intValue);
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
      return Var::Empty;
    }
  }

  int _inputs = 2;
  int _hidden = 4;
  int _outputs = 1;

  void resetState() override {
    _netRef = std::make_shared<Liquid>(_inputs, _hidden, _outputs);
  }
};

struct NARXBlock final : public NetworkProducer {
  CBParametersInfo parameters() { return NarxParams; }

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
    case 4:
      _inputMem = int(value.payload.intValue);
      break;
    case 5:
      _outputMem = int(value.payload.intValue);
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
    case 4:
      return Var(_inputMem);
    case 5:
      return Var(_outputMem);
    default:
      return Var::Empty;
    }
  }

  int _inputs = 2;
  OwnedVar _hidden{Var(4)};
  int _outputs = 1;
  int _inputMem = 2;
  int _outputMem = 2;

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
    _netRef = std::make_shared<NARX>(_inputs, hiddens, _outputs, _inputMem,
                                     _outputMem);
  }
};

struct LSTMBlock final : public NetworkProducer {
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
      return Var::Empty;
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
    _netRef = std::make_shared<LSTM>(_inputs, hiddens, _outputs);
  }
};
}; // namespace Nevolver

namespace chainblocks {
void registerBlocks() {
  Core::registerObjectType(Nevolver::SharedNetwork::Vendor,
                           Nevolver::SharedNetwork::Type,
                           Nevolver::SharedNetwork::ObjInfo);
  REGISTER_CBLOCK("Nevolver.Activate", Nevolver::Activate);
  REGISTER_CBLOCK("Nevolver.Predict", Nevolver::Predict);
  REGISTER_CBLOCK("Nevolver.Propagate", Nevolver::Propagate);
  REGISTER_CBLOCK("Nevolver.Clear", Nevolver::Clear);
  REGISTER_CBLOCK("Nevolver.MLP", Nevolver::MLPBlock);
  REGISTER_CBLOCK("Nevolver.NARX", Nevolver::NARXBlock);
  REGISTER_CBLOCK("Nevolver.LSTM", Nevolver::LSTMBlock);
  REGISTER_CBLOCK("Nevolver.Liquid", Nevolver::LiquidBlock);
  REGISTER_CBLOCK("Nevolver.SaveModel", Nevolver::SaveModel);
  REGISTER_CBLOCK("Nevolver.LoadModel", Nevolver::LoadModel);
}
} // namespace chainblocks

// Compiling with mingw compilers we need this trick to make sure
// no symbols are exported
#ifdef _WIN32
extern "C" {
__declspec(dllexport) void hiddenSymbols() {}
};
#endif
