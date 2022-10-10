#ifndef layer_h
#define layer_h

#include <vector>
#include "neuron.h"

class Layer {
public:
    Layer(int);
    void setInputs(std::vector<float>);
    void setOutputs(std::vector<float>);
    std::vector<float> feedForward();
    void gatherInputs();
    //float error(std::vector<float>);
    void connectLayer(Layer&);
    void addInputConnections();
    void resetDeriv();
    void backPropagate();
    void updateWeights();
    float getGradientMagnitude();
    void updateStepsize(float stepSize);
    std::vector<float> activation();
    std::vector<Neuron> neurons;
    void printOutput();
private:
    //std::vector<float> inputs;
};

#endif
