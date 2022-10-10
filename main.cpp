#include <vector>
//#include <xmmintrin.h>
#include <iostream>
#include <random>
#include "network.h"
#include "mnist/mnist.h"


void printOut(int label, int prediction, std::vector<float> image) {
    std::cout<<"\nNumber: "<<label<<" ";
    std::cout << " Prediction: " << prediction << '\n';

    for (int t=1;t<29;t++) {
        std::cout<<'\n';
        for (int u=1;u<29;u++) {
            if(image[u+28*t]>0.4) { std::cout << "▓"; }
            else { std::cout << "░"; }
        }
    }
}

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

int random(int min, int max) //range : [min, max]
{
    static bool first = true;
    if (first)
    {
        srand( time(NULL) ); //seeding for the first time only!
        first = false;
    }
    return min + rand() % (( max + 1 ) - min);
}



int main() {
    Network network({28*28,60,10});
    MNISTSet trainingData("mnist/train-images", "mnist/train-labels");

    int epochs = 25;
    int randomBatch = random(0,60);


    for (int j=0;j<epochs;j++) {
        int n_correct = 0;
        std::cout << "\n\nEpoch: "<<j<<std::endl;
        std::cout << "Stepsize:\n";
        for (int i=0;i<trainingData.totalSets();i++) {

            std::vector<int> TEMPimage;
            int label;
            trainingData.getSet(TEMPimage, label);
            std::vector<float> image;


            for( int i : TEMPimage) {
                image.push_back(((float)i)/255.0);
            }

            network.feedForward(image);
            // :)

            std::vector<float> answer(10, 0.0);
            answer[label]=1.00;
            network.calculateError(answer);
            //network.backPropagate(answer);
            //if (i%1000==0) { network.printError(); }
            if (i%20 == 0) { 
                network.backPropagate(); 
                if (i<1000) {
                    std::cout << network.getStepsize() << ", ";
                }
                randomBatch += random(0,60);
            }


            //            if (i%5000==0) {
            //                std::vector<float> output = network.output();
            //                printOut(label, prediction(output) ) {
            //            }
            std::vector<float> output = network.output();
            if (prediction(output)==label) { n_correct++; }
        }


        std::cout << "\n  correct/total: " << n_correct << "/" << trainingData.totalSets() << "\n\n";

        trainingData.reset();
        }

        return 0;
    }
