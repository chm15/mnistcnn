#include <iostream>
#include <fstream>
#include <string>
#include "mnist.h"

MNISTImage::MNISTImage(std::string fileName) : file(fileName,
        std::ios::binary) {
    if (file.is_open()) {
        file.read((char*)&this->magicNumber,sizeof(magicNumber));
        this->magicNumber = reverseInt(this->magicNumber);

        if (this->magicNumber != 2051) {
            if(this->magicNumber != 2051) throw std::runtime_error("Invalid\
                                MNIST label file!");
        }

        file.read((char*)&this->tImages,sizeof(tImages));
        this->tImages= reverseInt(this->tImages);

        file.read((char*)&this->tRows,sizeof(tRows));
        this->tRows= reverseInt(this->tRows);

        file.read((char*)&this->tCols,sizeof(tCols));
        this->tCols= reverseInt(this->tCols);

        this->startPos = file.tellg();

    } else {
        std::cout << "Could not open file." << std::endl;
    }
}

void MNISTImage::reset() { this->file.seekg(this->startPos); }

std::vector<int> MNISTImage::getImage() {
    std::vector<int> image;

    if (!this->file.is_open()) { return image; };

    for (int r=0;r<this->tRows;++r) {
        for (int c=0;c<this->tCols;++c){
            uint8_t tmp=0;
            this->file.read((char*)&tmp, sizeof(tmp));
            image.push_back(tmp);
        }
    }
    return image;
}


MNISTLabel::MNISTLabel(std::string fileName) : file(fileName,
        std::ios::binary) {
    if (file.is_open()) {
        file.read((char*)&this->magicNumber,sizeof(magicNumber));
        this->magicNumber = reverseInt(this->magicNumber);

        if (this->magicNumber != 2049) {
            if(this->magicNumber != 2049) throw std::runtime_error(\
                    "Invalid MNIST label file!");
        }

        file.read((char*)&this->tLabels,sizeof(tLabels));
        this->tLabels= reverseInt(this->tLabels);

        this->startPos = file.tellg();

    } else {
        std::cout << "Could not open file." << std::endl;
    }
}

int MNISTLabel::getLabel() {
    unsigned char tmp;
    if (!this->file.is_open()) { return -1; }
    this->file.read((char*)&tmp, sizeof(tmp));
    return (int)tmp;
}

void MNISTLabel::reset() { this->file.seekg(this->startPos); }


MNISTSet::MNISTSet(std::string imageStr, std::string labelStr) : imageFile(imageStr), labelFile(labelStr) {}

int MNISTSet::getSet(std::vector<int> &image, int &label) {
    image = this->imageFile.getImage();
    label = this->labelFile.getLabel();
    return 1;
}

int MNISTSet::totalSets() {
    int tImages = imageFile.totalImages();
    int tLabels= labelFile.totalLabels();
    if (tImages == tLabels) { return tImages; }
    else { return 0; }
}

void MNISTSet::reset() {
    this->imageFile.reset();
    this->labelFile.reset();
}

