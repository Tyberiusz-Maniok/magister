#pragma once
#include "sequential.h"
#include "conv2d.h"
#include "linear.h"
#include "maxpool.h"
#include "batch_norm2d.h"
#include "model.h"
#include "flatten.h"
#include "stats.h"
#include "residual.h"
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

static Model* vgg16() {
    Conv2dP conv1 = Conv2dP(new Conv2d(1, 64, 3, 1, relu));
    conv1->name = "conv1";
    Conv2dP conv2 = Conv2dP(new Conv2d(64, 64, 3, 1, relu));
    conv2->name = "conv2";
    MaxPoolP maxpool1 = MaxPoolP(new MaxPool(2));
    maxpool1->name = "maxpool1";

    Conv2dP conv3 = Conv2dP(new Conv2d(64, 128, 3, 1, relu));
    conv3->name = "conv3";
    Conv2dP conv4 = Conv2dP(new Conv2d(128, 128, 3, 1, relu));
    conv4->name = "conv4";
    MaxPoolP maxpool2 = MaxPoolP(new MaxPool(2));
    maxpool2->name = "maxpool2";

    Conv2dP conv5 = Conv2dP(new Conv2d(128, 256, 3, 1, relu));
    conv5->name = "conv5";
    Conv2dP conv6 = Conv2dP(new Conv2d(256, 256, 3, 1, relu));
    conv6->name = "conv6";
    Conv2dP conv7 = Conv2dP(new Conv2d(256, 256, 3, 1, relu));
    conv7->name = "conv7";
    MaxPoolP maxpool3 = MaxPoolP(new MaxPool(2));
    maxpool3->name = "maxpool3";

    Conv2dP conv8 = Conv2dP(new Conv2d(256, 512, 3, 1, relu));
    conv8->name = "conv8";
    Conv2dP conv9 = Conv2dP(new Conv2d(512, 512, 3, 1, relu));
    conv9->name = "conv9";
    Conv2dP conv10 = Conv2dP(new Conv2d(512, 512, 3, 1, relu));
    conv10->name = "conv10";
    MaxPoolP maxpool4 = MaxPoolP(new MaxPool(2));
    maxpool4->name = "maxpool4";

    Conv2dP conv11 = Conv2dP(new Conv2d(512, 512, 3, 1, relu));
    conv11->name = "conv11";
    Conv2dP conv12 = Conv2dP(new Conv2d(512, 512, 3, 1, relu));
    conv12->name = "conv12";
    Conv2dP conv13 = Conv2dP(new Conv2d(512, 512, 3, 1, relu));
    conv13->name = "conv13";
    MaxPoolP maxpool5 = MaxPoolP(new MaxPool(2));
    maxpool5->name = "maxpool5";

    FlattenP flatten = FlattenP(new Flatten);
    flatten->name = "flatten";

    LinearP lin1 = LinearP(new Linear(512, 4096, relu));
    lin1->name = "dense1";
    LinearP lin2 = LinearP(new Linear(4096, 1000, relu));
    lin2->name = "dense2";
    LinearP lin3 = LinearP(new Linear(1000, 10, relu));
    lin3->name = "dense3";


    std::vector<LayerP> layers = {conv1, conv2, maxpool1, conv3, conv4, maxpool2, conv5, conv6, conv7, maxpool3,
        conv8, conv9, conv10, maxpool4, conv11, conv12, conv13, maxpool5, flatten, lin1, lin2, lin3};
    Sequential* vgg = new Sequential(layers, 22);
    vgg->name = "vgg16";
    StatTrackerP stat_tracker = StatTrackerP(new StatTracker());
    return new Model(vgg, 0.1, stat_tracker);
}

static Model* resnet18() {
    Conv2dP conv1 = Conv2dP(new Conv2d(1, 64, 3, 1, relu));
    conv1->name = "conv1";
    
    MaxPoolP maxpool1 = MaxPoolP(new MaxPool(3, 1));
    maxpool1->name = "maxpool1";

    Conv2dP conv2_1 = Conv2dP(new Conv2d(64, 64, 3, 1, relu));
    Conv2dP conv2_2 = Conv2dP(new Conv2d(64, 64, 3, 1, relu));
    ResidualP res1 = ResidualP(new Residual(SequentialP(new Sequential({conv2_1, conv2_2}, 2))));
    res1->name = "res1";

    Conv2dP conv3_1 = Conv2dP(new Conv2d(64, 64, 3, 1, relu));
    Conv2dP conv3_2 = Conv2dP(new Conv2d(64, 64, 3, 1, relu));
    ResidualP res2 = ResidualP(new Residual(SequentialP(new Sequential({conv3_1, conv3_2}, 2))));
    res2->name = "res2";

    Conv2dP conv4_1 = Conv2dP(new Conv2d(64, 128, 3, 1, relu));
    Conv2dP conv4_2 = Conv2dP(new Conv2d(128, 128, 3, 1, relu));
    ResidualP res3 = ResidualP(new Residual(SequentialP(new Sequential({conv4_1, conv4_2}, 2))));
    res3->name = "res3";

    Conv2dP conv5_1 = Conv2dP(new Conv2d(128, 128, 3, 1, relu));
    Conv2dP conv5_2 = Conv2dP(new Conv2d(128, 128, 3, 1, relu));
    ResidualP res4 = ResidualP(new Residual(SequentialP(new Sequential({conv5_1, conv5_2}, 2))));
    res4->name = "res4";

    Conv2dP conv6_1 = Conv2dP(new Conv2d(128, 256, 3, 1, relu));
    Conv2dP conv6_2 = Conv2dP(new Conv2d(256, 256, 3, 1, relu));
    ResidualP res5 = ResidualP(new Residual(SequentialP(new Sequential({conv6_1, conv6_2}, 2))));
    res5->name = "res5";

    Conv2dP conv7_1 = Conv2dP(new Conv2d(256, 256, 3, 1, relu));
    Conv2dP conv7_2 = Conv2dP(new Conv2d(256, 256, 3, 1, relu));
    ResidualP res6 = ResidualP(new Residual(SequentialP(new Sequential({conv7_1, conv7_2}, 2))));
    res6->name = "res6";

    Conv2dP conv8_1 = Conv2dP(new Conv2d(256, 512, 3, 1, relu));
    Conv2dP conv8_2 = Conv2dP(new Conv2d(512, 512, 3, 1, relu));
    ResidualP res7 = ResidualP(new Residual(SequentialP(new Sequential({conv8_1, conv8_2}, 2))));
    res7->name = "res7";

    Conv2dP conv9_1 = Conv2dP(new Conv2d(512, 512, 3, 1, relu));
    Conv2dP conv9_2 = Conv2dP(new Conv2d(512, 512, 3, 1, relu));
    ResidualP res8 = ResidualP(new Residual(SequentialP(new Sequential({conv9_1, conv9_2}, 2))));
    res8->name = "res8";

    MaxPoolP maxpool2 = MaxPoolP(new MaxPool(2));
    maxpool2->name = "maxpool2";

    FlattenP flatten = FlattenP(new Flatten());

    LinearP lin1 = LinearP(new Linear(512, 10, relu));
    lin1->name = "dense1";

    std::vector<LayerP> layers = {conv1, maxpool1, res1, res2, res3, res4, res5, res6, res7, res8,
        maxpool2, flatten, lin1};
    Sequential* resnet = new Sequential(layers, 13);
    resnet->name = "resnet18";
    StatTrackerP stat_tracker = StatTrackerP(new StatTracker());
    return new Model(resnet, 0.1, stat_tracker);
}

}

}
