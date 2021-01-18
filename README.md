## MNIST Handwritten Solution (from scratch)
The goal of this project was to implement a neural network in an object oriented manner. It is not efficient.  


This is a CNN I implemented without the use of any libraries. It uses sigmoid as the activation function throughout. The network learns through backpropagation.

## Building
``` bash
# Don't laugh. Best build script ever.

g++ *.cpp mnist/*.cpp -o HANDWRITTEN
./HANDWRITTEN
```
