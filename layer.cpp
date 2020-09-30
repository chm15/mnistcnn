#include <vector>
#include <iostream>
#include "layer.h"
#include "neuron.h"


Layer::Layer(int totalNeurons) {
    for(int i=0;i<totalNeurons;i++) {
        this->neurons.push_back(Neuron());
    }
}

void Layer::connectLayer(Layer& layerConn) {
    int TNeuronsL1 = layerConn.neurons.size();
    int TNeuronsThis = this->neurons.size();

    for (int i=0;i<TNeuronsThis;i++) {
        for (int j=0;j<TNeuronsL1;j++) {
            this->neurons[i].addConnection(layerConn.neurons[j]);
        }
    }
}

void Layer::addInputConnections() {
    int tNeurons = this->neurons.size();
    for (int i=0;i<tNeurons;i++) {
        this->neurons[i].addInput();
    }
}

void Layer::setInputs(std::vector<float> inputs) {
    int inputSize = inputs.size();
    int layerSize = this->neurons.size();
    for (int i=0;i<layerSize;i++) {
        Neuron& cNeuron = this->neurons[i];
        cNeuron.setInput(inputs[i]);
    }
    return;
}
void Layer::setOutputs(std::vector<float> inputs) {
    int inputSize = inputs.size();
    int layerSize = this->neurons.size();
    for (int i=0;i<layerSize;i++) {
        Neuron& cNeuron = this->neurons[i];
        cNeuron.setOutput(inputs[i]);
    }
    return;
}

void Layer::resetDeriv() {
    for (int i=0;i<this->neurons.size();i++) {
        this->neurons[i].resetDeriv();
    }
}

std::vector<float> Layer::feedForward() {
    std::vector<float> output;
    for (int i=0;i< this->neurons.size();i++) {
        Neuron& cNeuron = this->neurons[i];
        cNeuron.feedForward();
        output.push_back(cNeuron.activation());
    }
    return output;
}

void Layer::gatherInputs() {
    int tNeurons = this->neurons.size();
    for (int i=0;i<tNeurons;i++) {
        this->neurons[i].gatherInputs();
    }
}

std::vector<float> Layer::activation() {
    std::vector<float> output;
    for (int i=0;i<this->neurons.size();i++) {
        output.push_back(this->neurons[i].activation());
    }
    return output;
}

void Layer::backPropagate() {
    int tNeurons = this->neurons.size();
    for (int i=0;i<tNeurons;i++) {
        Neuron& cNeuron = this->neurons[i];
        cNeuron.backPropagate();
    }
}

void Layer::updateWeights() {
    for (int i=0;i<this->neurons.size();i++) {
        this->neurons[i].updateWeights();
    }
}

void Layer::printOutput() {
    std::cout << "Layer output: " << std::endl;
    for (int i=0;i<this->neurons.size();i++) {
        std::cout << this->neurons[i].activation() << ' ';
    }
    std::cout << '\n';
}
