#include "stats.h"
#include <iostream>
#include <fstream>

using namespace lamp;

void StatTracker::add(Stats* stats) {
    this->stats.push_back(stats);
}

void StatTracker::to_csv(std::string filename) {
    std::ofstream file;
    file.open(filename);
    for (Stats* s: stats) {
        file << s->to_csv_row();
    }
    file.close();
}
