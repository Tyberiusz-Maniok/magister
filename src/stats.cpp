#include "stats.h"
#include <iostream>
#include <fstream>

using namespace lamp;

void StatTracker::add(Stats stat) {
    std::cout << stat.to_csv_row();
    stats.push_back(stat);
}

void StatTracker::to_csv(std::string filename) {
    std::ofstream file;
    file.open(filename);
    // for (Stats s : this->stats) {
    for (int i = 0; i < stats.size(); i++) {
        file << stats.at(i).to_csv_row();
    }
    file.close();
}
