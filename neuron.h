#ifndef NEURON_H
#define NEURON_H

#include <vector>

struct Connection;

class Neuron {
public:
    void addConnection(Neuron&);
    void addInput();
    void setInput(float);
    void setOutput(float);
    void gatherInputs();
    float feedForward();
    float activation(); 
    void backPropagate();
    void resetDeriv();
    void calcDeriv(float inputDeriv);
    void updateWeights();
    float lRate = 0.2;
private: 
    float val=0;
    float bias = 0.0;
    float a;
    float deriv;   // ∂a/∂z
    //float derivZA; // ∂z/∂a^(L-1)
    //float delta;
    std::vector<Connection> connections;
};

class Connection {
public:
    Connection(Neuron*, float);
    Connection(float);
    float feedForwardInput(float);
    float feedForward();

    Neuron* neuron;
    float weight;
};

#endif
