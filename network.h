#ifndef NETWORK_H
#define NETWORK_H

#include <vector>

#include "layer.h"


class Network {
public:
    Network(std::vector<int>);
    void addLayer(int);
    std::vector<float> feedForward(std::vector<float>);
    void resetDeriv();
    void backPropagate(std::vector<float>);
    void backPropagate();   // Same as other backProp, but uses calcErr() to 
                            // create a rolling error deriv.
    std::vector<float> output();
    void updateWeights();
    //void train(std::vector<float>);
    void printOutput();
    void printError();
    void calculateError(std::vector<float>);
    void resetError() { this->error.clear(); };
    void updateStepsize();
    float getStepsize();

private:
    std::vector<Layer> layers;
    std::vector<float> error;
    float getGradientMagnitude();
    float stepSize = 0.02;  // adaptive
};

#endif
