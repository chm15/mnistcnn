#include <vector>
//#include <xmmintrin.h>
#include <iostream>
#include "network.h"
#include "mnist/mnist.h"



int prediction(std::vector<float> a) {
    float highest = 0.0;
    int number = 0;
    for (int i=0; i<a.size();i++) {
        if (a[i] > highest) { 
            number = i; 
            highest = a[i];
        }
    }
    return number;
}
        

int main() {
    Network network({28*28,60,10});
    MNISTSet trainingData("mnist/train-images", "mnist/train-labels");
    
    int epochs = 25;

    for (int j=0;j<epochs;j++) {
        std::cout << "Epoch: "<<j<<std::endl;
        for (int i=0;i<trainingData.totalSets();i++) {

            std::vector<int> TEMPimage;
            int label;
            trainingData.getSet(TEMPimage, label);
            std::vector<float> image;


            for( int i : TEMPimage) {
                image.push_back(((float)i)/255.0);
            }

            network.feedForward(image);

            std::vector<float> answer(10, 0.0);
            answer[label]=1.00;
            network.calculateError(answer);
            //network.backPropagate(answer);
            //if (i%1000==0) { network.printError(); }
            if (i%60==0) { network.backPropagate(); }

            if (i%5000==0) {
                std::cout<<"Number: "<<label<<" ";
                std::vector<float> output = network.output();
                std::cout << " Prediction: " << prediction(output) << '\n';

                for (int t=1;t<29;t++) {
                    std::cout<<'\n';
                    for (int u=1;u<29;u++) {
                        if(image[u+28*t]>0.4) { std::cout << "▓"; }
                        else { std::cout << "░"; }
                    }
                }

            }
        }

        trainingData.reset();
    }

    return 0;
}
