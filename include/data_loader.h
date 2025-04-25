#pragma once
#include "tensor.h"

namespace lamp {

class DataLoader {
    public:

        Tensor& next_batch();
};

}

