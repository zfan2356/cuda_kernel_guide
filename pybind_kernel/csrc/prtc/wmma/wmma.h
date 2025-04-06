#pragma once
#include <torch/extension.h>
#include <tuple>

namespace prtc {
at::Tensor my_wmma(at::Tensor a, at::Tensor b, at::Tensor c);
}
