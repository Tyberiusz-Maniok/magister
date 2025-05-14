#include "rng.h"
#include "mkl.h"

using namespace lamp;

RandomGen::RandomGen(int seed) {
    vslNewStream(&stream, VSL_BRNG_MT19937, seed);
}

RandomGen::~RandomGen() {
    vslDeleteStream(&stream);
}

void RandomGen::populate(int size, float* data, float low, float high) {
    vsRngUniform(VSL_RNG_METHOD_UNIFORM_STD, stream, size, data, low, high);
}
