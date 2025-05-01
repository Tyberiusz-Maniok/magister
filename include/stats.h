#pragma once
#include <vector>
#include <string>

namespace lamp {

struct Stats {
    std::string name;
    double time;

    Stats(std::string name, double time) : name(name), time(time) {}

    std::string to_csv_row() {
        return name + ";" + std::to_string(time);
    }
};

class StatTracker {
    public:
        std::vector<Stats*> stats = std::vector<Stats*>();

        void add(Stats* stats);

        void to_csv(std::string filename);
};


}