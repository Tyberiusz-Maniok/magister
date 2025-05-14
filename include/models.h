#pragma once
#include "sequential.h"
#include "conv2d.h"
#include "linear.h"
#include "maxpool.h"
#include "batch_norm2d.h"
#include "model.h"
#include "flatten.h"
#include "stats.h"
#include <sequential.h>

namespace lamp {

namespace models {

static Model* lenet() {
    Conv2d* conv1 = new Conv2d(1, 6, 6, 1);
    conv1->name = "conv1";
    MaxPool* maxpool1 = new MaxPool(2);
    maxpool1-> name = "maxpool1";
    Conv2d* conv2 = new Conv2d(6, 16, 6, 1);
    conv2->name = "conv2";
    MaxPool* maxpool2 = new MaxPool(2);
    maxpool2-> name = "maxpool2";
    Flatten* flatten = new Flatten();
    flatten->name = "flatten";
    // Linear lin1 = new Linear(120);
    std::vector<Layer*> layers = {conv1, maxpool1, conv2, maxpool2, flatten};
    Sequential* lenet = new Sequential(layers, 5);
    lenet->name = "lenet";
    StatTracker* stat_tracker;
    return new Model(lenet, 0.1, stat_tracker);
}

}

}
