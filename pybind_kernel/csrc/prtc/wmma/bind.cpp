#include "bind.h"
#include "wmma.h"

namespace prtc {
namespace py = pybind11;
void bind(pybind11::module &m) { m.def("my_wmma", &my_wmma, "prtc::my_wmma"); }
} // namespace prtc
