#pragma once
#include <torch/extension.h>

namespace prtc {

extern void bind(pybind11::module &m);

}
