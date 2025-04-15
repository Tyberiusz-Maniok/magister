#pragma once
#include <mkl_vsl.h>

namespace lamp {

class RandomGen {
    private:
        VSLStreamStatePtr stream;

    public:
        RandomGen(int seed);
        ~RandomGen();

        void populate(int size, float* data, float low = -1, float high = 1);
};

// class RngContainer {
//     private:
//         static RandomGen& rng;

//     public:
//         static RandomGen& init(int seed);
//         static void populate(int size, float* data, float low = -1, float high = 1);
// };

}
