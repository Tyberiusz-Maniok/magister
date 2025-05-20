#pragma once
#include <mkl_vsl.h>
#include "consts.h"

namespace lamp {

class RandomGen {
    private:
        VSLStreamStatePtr stream;

    public:
        RandomGen(int seed);
        ~RandomGen();

        void populate(int size, float* data, float low = -1, float high = 1);
};

static RandomGen* global_rand = new RandomGen(SEED);

}
