#include <vector>
#include <iostream>
#include "network.h"


int main() {
    
    Network network({3,6,2});

    //network.addLayer(3);
    //network.addLayer(16);
    //network.addLayer(6);
    //network.addLayer(2);

    // Feed forward with vector of values.
    std::vector< std::vector<float> > inputs;
    std::vector< std::vector<float> > answers;
    
    // Initializing inputs and answers.
    inputs.push_back({1,0,0});
    answers.push_back({0,0});
    inputs.push_back({0,1,1});
    answers.push_back({0,1});

    // Training network.
    int it=0;
    while (it<60000) {
        //std::cout << "Iteration " << it << std::endl;
        for(int i=0;i<inputs.size();i++) {
            network.feedForward(inputs[i]);
            //network.printOutput();
            network.backPropagate(answers[i]);
        }
        
        //std::cout << "\n\n";
        it++;
    }

    // Testing network after training.
    network.feedForward({1,0,0});
    network.printOutput();

    network.feedForward({0,1,1});
    network.printOutput();
    
    // Calculate the error based on the expected values.
    //std::vector<float> answers;
    //network.train(answers);

    // Train the network based on the average error after a few iterations of feed forward processing.


    return 0;
}
