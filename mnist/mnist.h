#ifndef MNIST_H
#define MNIST_H
#include <vector>
#include <string>
#include <fstream>

class MNISTImage {
public:
    MNISTImage(std::string);
    std::vector<int> getImage();
    int totalImages() { return this->tImages; }
    void reset();

private:
    std::ifstream file;
    int magicNumber;
    int tImages;
    int tRows;
    int tCols;
    std::streampos startPos;

    static int reverseInt (int i) {
        unsigned char ch1, ch2, ch3, ch4;
        ch1=i&255;
        ch2=(i>>8)&255;
        ch3=(i>>16)&255;
        ch4=(i>>24)&255;
        return((int)ch1<<24)+((int)ch2<<16)+((int)ch3<<8)+ch4;
    }

};

class MNISTLabel {
public:
    MNISTLabel(std::string);
    int getLabel();
    int totalLabels() { return this->tLabels; }
    void reset();

private:
    static int reverseInt (int i) {
        unsigned char ch1, ch2, ch3, ch4;
        ch1=i&255;
        ch2=(i>>8)&255;
        ch3=(i>>16)&255;
        ch4=(i>>24)&255;
        return((int)ch1<<24)+((int)ch2<<16)+((int)ch3<<8)+ch4;
    }

    std::ifstream file;
    int magicNumber;
    int tLabels;
    std::streampos startPos;
};


class MNISTSet {
public:
    MNISTSet(std::string imageFile, std::string labelFile);
    int getSet(std::vector<int>&, int&);
    int totalSets();
    void reset();
private:
    MNISTImage imageFile;
    MNISTLabel labelFile;
};

#endif
