#include "data_loader.h"
#include <iostream>
#include <fstream>

using namespace lamp;

DataLoader::DataLoader(int batch_size) : batch_size(batch_size) {
    this->total_size = 60000; //TODO move to const
}

Tesnor& DataLoader::next_batch() {
    return 
}
