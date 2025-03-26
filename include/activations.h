#pragma once
#include "tensor.h"
#include <functional>

namespace lamp {

typedef std::function<void(Tensor&)> activ_fn;

void identity(Tensor& x);

void relu(Tensor& x);

}
