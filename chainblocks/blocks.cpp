#include "../network.hpp"
#include "../networks/mlp.hpp"
// order matters!
INITIALIZE_EASYLOGGINGPP
#include "chainblocks.hpp"
#include "dllblock.hpp"

using namespace chainblocks;

namespace Nevolver {
struct SharedNetwork final {
  constexpr static auto Vendor = 'sink';
  constexpr static auto Type = 'nnet';

  static CBBool serialize(CBPointer pnet, uint8_t **outData, size_t *outLen,
                          CBPointer *handle) {}

  static void freeMem(CBPointer handle) {}

  static CBPointer deserialize(uint8_t *data, size_t len) {}

  static void addRef(CBPointer pnet) {
    auto p = reinterpret_cast<SharedNetwork *>(pnet);
    p->refcount++;
  }

  static void decRef(CBPointer pnet) {
    auto p = reinterpret_cast<SharedNetwork *>(pnet);
    p->refcount--;
    if (p->refcount == 0)
      delete p;
  }

  static inline CBObjectInfo ObjInfo{"Nevolver.Network", &serialize, &freeMem,
                                     &deserialize,       &addRef,    &decRef};

  SharedNetwork(const std::shared_ptr<Network> &net) {
    _holder = net;
    refcount = 1;
  }

  ~SharedNetwork() { assert(refcount == 0); }

  SharedNetwork(const SharedNetwork &other) = delete;
  SharedNetwork &operator=(const SharedNetwork &other) = delete;

  std::shared_ptr<Network> get() { return _holder; }

private:
  std::shared_ptr<Network> _holder;
  uint32_t refcount;
};

struct NetVar final : public CBVar {
  NetVar(const NetVar &other) = delete;
  NetVar &operator=(const NetVar &other) = delete;

  NetVar(const std::shared_ptr<Network> &net) : CBVar() {
    valueType = CBType::Object;
    payload.objectValue = new SharedNetwork(net);
    payload.objectVendorId = SharedNetwork::Vendor;
    payload.objectTypeId = SharedNetwork::Type;
    objectInfo = &SharedNetwork::ObjInfo;
    flags |= CBVAR_FLAGS_USES_OBJINFO | CBVAR_FLAGS_REF_COUNTED;
  }

  NetVar(const CBVar &other) : CBVar() {
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

  operator std::shared_ptr<Network>() {
    return reinterpret_cast<SharedNetwork *>(payload.objectValue)->get();
  }
};

struct NeuroVar final : public CBVar {
  NeuroVar(const NeuroFloat &f) : CBVar() {
    valueType = Float;
    payload.floatValue = f;
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

// Problems
// We want to train this with genetic
// so if we release networks on cleanup it's bad

struct NetworkUser {
  static inline Type IntType{{CBType::Int}};
  static inline Type IntSeq{{CBType::Seq, .seqTypes = IntType}};
  static inline Type AnyType{{CBType::Any}};
  static inline Type NetType{
      {CBType::Object, .object = {SharedNetwork::Vendor, SharedNetwork::Type}}};
  static inline Type NetVarType{
      {CBType::ContextVar, .contextVarTypes = NetType}};

  static inline Parameters _userParam{
      {"Name", "The name of the network model variable.", {NetVarType}}};

  static CBTypesInfo inputTypes() { return AnyType; }
  static CBTypesInfo outputTypes() { return AnyType; }
  static CBParametersInfo parameters() { return _userParam; }

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
  static inline Type FloatType{{CBType::Float}};
  static inline Type FloatSeq{{CBType::Seq, .seqTypes = FloatType}};

  static CBTypesInfo inputTypes() { return FloatSeq; }

  void warmup(CBContext *context) {
    NetworkUser::warmup(context);
    _netRef = NetVar(_netParam.get());
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

struct Propagate : public NetworkConsumer {
  CBTypesInfo outputTypes() { return FloatType; }

  double _rate = 0.3;
  double _momentum = 0.0;

  CBVar activate(CBContext *context, const CBVar &input) {
    // NO-Copy propagate
    NeuroVars in(input);
    return _netRef->propagate<NeuroVar>(in, _rate, _momentum, true);
  }
};

struct MLPBlock : public NetworkUser {
  static inline Parameters _mlpParams{
      _userParam,
      {{"Inputs", "The number of input nodes.", {IntType}},
       {"Hidden",
        "The number of hidden nodes, can be a sequence for multiple layers.",
        {{IntType, IntSeq}}},
       {"Outputs", "The number of output nodes.", {IntType}}}};

  CBParametersInfo parameters() { return _mlpParams; }

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

  void warmup(CBContext *context) {
    NetworkUser::warmup(context);

    // assert we are the creators of this network
    assert(_netParam.get().valueType == CBType::None);

    // create the network if it never existed yet
    if (!_netRef) {
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

    _netParam.get() = NetVar(_netRef);
    _netParam.get().refcount++;
  }

  CBVar activate(CBContext *context, const CBVar &input) { return input; }
};
}; // namespace Nevolver

namespace chainblocks {
void registerBlocks() {
  LOG(DEBUG) << "Loading Nevolver blocks...";
  REGISTER_CBLOCK("Nevolver.MLP", Nevolver::MLPBlock);
  REGISTER_CBLOCK("Nevolver.Activate", Nevolver::Activate);
  REGISTER_CBLOCK("Nevolver.Propagate", Nevolver::Propagate);
}
} // namespace chainblocks
