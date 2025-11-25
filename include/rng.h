#pragma once
#include <random>
#include "consts.h"

namespace lamp {

class RandomGen {
    private:
        std::mt19937 generator;
        std::uniform_real_distribution<float> distribution;

    public:
        RandomGen(int seed);
        ~RandomGen();

        void populate(int size, float* data, float low = -1, float high = 1);
};

static RandomGen* global_rand = new RandomGen(SEED);

}
