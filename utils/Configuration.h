/**
 *@file Configuration.h
 *@date 19.12.2025
 *@author Luis Gall
 */

#pragma once

#include <string>

class Configuration {

public:

    void parseConfig(int argc, char** argv) {

        std::map<std::string, std::string> options;

        // Creating a map of [--option : value] entries
        for (int i = 1; i < argc; ++i) {
            std::string arg = argv[i];
            if (!arg.empty() && arg[0] == '-') {
                std::string key = arg;
                std::string value;

                if (i+1 < argc) {
                    std::string potentialValue = argv[i+1];
                    if (!potentialValue.empty() && potentialValue[0] != '-') {
                        value = potentialValue;
                        ++i;
                    }
                }
                options[key] = value;
            }
        }

        // Extracting the values of the map
        for (auto& pair : options) {
            if (pair.first == "--cutoff") {
                _cutoff = std::stod(pair.second);
            } else if (pair.first == "--iterations") {
                _numIterations = std::stoi(pair.second);
            } else if (pair.first == "--deltaT") {
                _deltaT = std::stod(pair.second);
            } else if (pair.first == "--boxMin") {
                _boxMin = std::stod(pair.second);
            } else if (pair.first == "--boxMax") {
                _boxMax = std::stod(pair.second);
            } else if (pair.first == "--numParticles") {
                _numParticles = std::stoi(pair.second);
            } else if (pair.first == "--numHalos") {
                _numHalos = std::stoi(pair.second);
            }
        }
    }

    auto getCutoff() const {
        return _cutoff;
    }

    auto getDeltaT() const {
        return _deltaT;
    }

    auto getBoxMin() const {
        return _boxMin;
    }

    auto getBoxMax() const {
        return _boxMax;
    }

    auto getNumHalos() const {
        return _numHalos;
    }

    auto getNumParticles() const {
        return _numParticles;
    }

    auto getNumIterations() const {
        return _numIterations;
    }

private:
    double _cutoff {0.1};

    double _boxMin {0};

    double _boxMax {0};

    size_t _numIterations {0};

    size_t _numParticles {0};

    size_t _numHalos {0};

    double _deltaT {0};
};