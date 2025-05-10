#pragma once
#include "sequential.h"
#include "conv2d.h"
#include "linear.h"
#include "maxpool.h"
#include "batch_norm2d.h"

namespace lamp {

static Sequential* lenet() {
    Conv2d* conv1 = new Conv2d(1, 6, 6, 1);
    return nullptr;
}

}
