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
    Conv2dP conv1 = Conv2dP(new Conv2d(1, 6, 6, 1));
    conv1->name = "conv1";
    MaxPoolP maxpool1 = MaxPoolP(new MaxPool(2));
    maxpool1->name = "maxpool1";
    Conv2dP conv2 = Conv2dP(new Conv2d(6, 16, 6, 1));
    conv2->name = "conv2";
    MaxPoolP maxpool2 = MaxPoolP(new MaxPool(2));
    maxpool2->name = "maxpool2";
    FlattenP flatten = FlattenP(new Flatten());
    flatten->name = "flatten";
    // Linear lin1 = new Linear(120);
    std::vector<LayerP> layers = {conv1, maxpool1, conv2, maxpool2, flatten};
    Sequential* lenet = new Sequential(layers, 5);
    lenet->name = "lenet";
    StatTrackerP stat_tracker = StatTrackerP(new StatTracker());
    return new Model(lenet, 0.1, stat_tracker);
}

}

}
