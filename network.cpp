#include <vector>
#include <iostream>
#include "network.h"
#include "layer.h"

Network::Network(std::vector<int> layers) {
    for (int i=0;i<layers.size();i++) {
        this->addLayer(layers[i]);
    }
}
void Network::addLayer(int totalNeurons) {
    this->layers.push_back(Layer(totalNeurons));
    Layer& newLayer = this->layers.back();

    // Connect all neurons in this layer to the neurons of the previous layer.
    int totalLayers = this->layers.size();
    if (totalLayers == 1) {
        newLayer.addInputConnections();
    } else if (totalLayers > 1){
        Layer& previousLayer = this->layers[totalLayers-2];
        newLayer.connectLayer(previousLayer);
    }
    return;
}

std::vector<float> Network::feedForward(std::vector<float> input) {

    Layer& inputLayer = this->layers[0];
    std::vector<float> output;

    if (input.size() != inputLayer.neurons.size()) { 
        std::cout << "Network.h error: input != size(inputLayer)"<<std::endl;
        return output; 
    }
    
    // Set the z in each neuron of the input layer to the corresponding input value.
    // Must call feedForward afterwards.
    //inputLayer.setInputs(input);
    //inputLayer.feedForward();
    //inputLayer.printOutput();
    inputLayer.setOutputs(input);

    for (int i=1;i<this->layers.size();i++) {
        Layer& cLayer = this->layers[i];
        cLayer.gatherInputs();
        cLayer.feedForward();
        //cLayer.printOutput();
    }

    output = this->output();

    return output;
}

void Network::resetDeriv() {
    for (int i=0;i<this->layers.size();i++) {
        this->layers[i].resetDeriv();
    }
}

void Network::backPropagate(std::vector<float> yValues) {
    Layer& outputLayer = this->layers.back();
    int tNeurons = outputLayer.neurons.size();

    this->resetDeriv();

    for (int i=0;i<tNeurons;i++) {
        Neuron& cNeuron = outputLayer.neurons[i];
        float errorDeriv = cNeuron.activation() - yValues[i];
        cNeuron.calcDeriv(errorDeriv);
    }

    int tLayers = this->layers.size();
    for (int i=tLayers-1; i>0;i--) {
        this->layers[i].backPropagate();
    }

    // Update the network's weights.
    this->updateWeights();
}

void Network::updateWeights() {
    for (int i=0;i<this->layers.size();i++) {
        this->layers[i].updateWeights();
    }
}

float Network::getGradientMagnitude() {
    float magSum = 0;
    for (int i=0;i<this->layers.size();i++) {
        magSum += this->layers[i].getGradientMagnitude();
    }
    return magSum;
}

void Network::updateStepsize() {
    float magSum = this->getGradientMagnitude();
    float stepSize = 1/(0.5+magSum);
    this->stepSize = stepSize;

    for (int i=0;i<this->layers.size();i++) {
        this->layers[i].updateStepsize(stepSize);
    }
}

float Network::getStepsize() {
    return this->stepSize;
}

void Network::backPropagate() {
    Layer& outputLayer = this->layers.back();
    int tNeurons = outputLayer.neurons.size();

    this->resetDeriv();

    for (int i=0;i<tNeurons;i++) {
        Neuron& cNeuron = outputLayer.neurons[i];
        cNeuron.calcDeriv(this->error[i]);
    }

    int tLayers = this->layers.size();
    for (int i=tLayers-1; i>0;i--) {
        this->layers[i].backPropagate();
    }

    // Update the network's weights.
    this->updateStepsize();
    this->updateWeights();
    this->resetError();
}

//void Network::train(std::vector<float> answer) {
    //this->backPropagate

std::vector<float> Network::output() {
    return this->layers.back().activation();
}

void Network::printError() {
    std::cout << "Error: ";
    for (float f : this->error) {
        std::cout << f << ' ';
    }
    std::cout << std::endl;
}


void Network::printOutput() {
    std::vector<float> output = this->output();

    std::cout << "Network output: " << std::endl;
    for (int i=0;i<output.size();i++) {
        std::cout << output[i] << ' ';
    }
    std::cout << '\n';
}


void Network::calculateError(std::vector<float> err) {
    if (this->error.size() != this->output().size()) { this->error = err; }
    for (int i=0;i<this->output().size();i++) {
        this->error[i]= (this->error[i]+this->output()[i]-err[i])/2.0;
    }
}
