#include <iostream>
#include "mnist.h"

int main() {
    
    //MNISTImage testImages("train-images");
    //MNISTLabel testLabels("train-labels");


    MNISTSet testSet("train-images", "train-labels");

    std::vector<int> image;
    int label;

    for (int i=0;i<100&&i<testSet.totalSets();i++) {
        testSet.getSet(image, label);
        std::cout <<"Label: "<< label<< std::endl;
    }




    return 0;
}
