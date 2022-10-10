#include <vector>
#include <iostream>
#include <cmath>
#include "neuron.h"
#include "sigmoid.h"

void Neuron::addConnection(Neuron& neuronConn) {
    float randWeight = ((rand() % 120)-60)/100.0;
    Connection conn(&neuronConn, randWeight);
    this->connections.push_back(conn);
}

void Neuron::addInput() {
    this->connections.push_back(Connection(((rand() % 120)-60)/100.0));
}

void Neuron::setInput(float newVal) {
    this->val = this->connections[0].feedForwardInput(newVal) + this->bias;
}
void Neuron::setOutput(float newVal) {
    this->a = newVal;
}

void Neuron::gatherInputs() {
    // add up weights and bias
    this->val = this->bias;
    for (int i=0;i<this->connections.size();i++) {
        Connection& cConn = this->connections[i];
        this->val += cConn.feedForward();
    }
}

float Neuron::feedForward() {
    float z = this->val;
    this->a = sigmoid(z);
    return this->a;
}

float Neuron::activation() { 
    return this->a; 
}

void Neuron::calcDeriv(float inputDeriv) {
    // Takes ∂E/∂a as param where a is this->a 
    this->deriv += inputDeriv * sigmoidDeriv(this->activation());
}

void Neuron::backPropagate() {
    // Calls 'calcDeriv' on connected neurons.
    int tConn = this->connections.size();
    for (int i=0;i<tConn;i++) {
        Connection& cConn = this->connections[i];
        Neuron* cNeuron = cConn.neuron;
        cNeuron->calcDeriv(this->deriv * cConn.weight);
    }
}

float Neuron::getGradientMagnitude() {
    float magSum = 0;  // Magnitude sum for neuron
    int tConn = this->connections.size();
    for (int i=0;i<tConn;i++) {
        Connection& cConn = this->connections[i];
        //magSum += abs(this->deriv * cConn.neuron->activation());
        magSum += abs(this->deriv);
    }
    return magSum;
}

void Neuron::updateStepsize(float stepSize) {
    this->lRate = stepSize;
}

void Neuron::updateWeights() {
    int tConn = this->connections.size();
    for (int i=0;i<tConn;i++) {
        Connection& cConn = this->connections[i];
        cConn.weight -= this->deriv * this->lRate * cConn.neuron->activation();
    }
    this->bias -= this->lRate * this->deriv;
    //this->bias -= 0.1 * this->deriv;
}

void Neuron::resetDeriv() {
    this->deriv = 0.0;
}





Connection::Connection(Neuron* n, float w) : neuron(n) {
    this->weight = w;
}
Connection::Connection(float weight) {
    this->weight = weight;
}

float Connection::feedForwardInput(float input) {
    return this->weight * input;
}

float Connection::feedForward() {
    return this->weight * this->neuron->activation();
}
