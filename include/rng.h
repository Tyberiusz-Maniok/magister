#pragma once
#include <mkl_vsl.h>

class RandomGen {
    private:
        VSLStreamStatePtr stream;

    public:
        RandomGen(int seed);
        ~RandomGen();

        void populate(int size, float* data, float low, float high);
};