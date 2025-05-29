#pragma once
#include <vector>
#include <string>
#include <memory>

namespace lamp {

struct Stats {
    std::string name;
    double time;

    Stats(std::string name, double time) : name(name), time(time) {}

    std::string to_csv_row() {
        return name + ";" + std::to_string(time) + "\n";
    }
};

class StatTracker {
    public:
        std::vector<Stats> stats;

        void add(Stats stat);

        void to_csv(std::string filename);
};

typedef std::shared_ptr<StatTracker> StatTrackerP;

}