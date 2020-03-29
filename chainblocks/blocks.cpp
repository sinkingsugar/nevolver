#include "../network.hpp"
// order matters!
INITIALIZE_EASYLOGGINGPP
#include "dllblock.hpp"

namespace Nevolver {}

namespace chainblocks {
void registerBlocks() { LOG(DEBUG) << "Loading Nevolver blocks..."; }
} // namespace chainblocks
