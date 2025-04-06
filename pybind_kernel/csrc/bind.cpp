#include "prtc/wmma/bind.h"

#include "torch/extension.h"

namespace prtc {
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  auto prtc = m.def_submodule("prtc");
  bind(prtc);
}
} // namespace prtc
