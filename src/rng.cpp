#include "rng.h"

using namespace lamp;

RandomGen::RandomGen(int seed) : generator(seed), distribution(-1.0f, 1.0f) {
    // Generator and distribution initialized in initializer list
}

RandomGen::~RandomGen() {
    // No cleanup needed for std::mt19937
}

void RandomGen::populate(int size, float* data, float low, float high) {
    // Update distribution range if different from default
    if (low != -1.0f || high != 1.0f) {
        distribution = std::uniform_real_distribution<float>(low, high);
    }
    
    for (int i = 0; i < size; ++i) {
        data[i] = distribution(generator);
    }
}
