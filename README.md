# ActiveStereoNet
This repository builds upon the open source [Active stereo net implementation](https://github.com/linjc16/ActiveStereoNet) for active tartanair and D435i datsets.

## Requirments

###### CUDA = v11.1
###### CuDNN >= v8.2.1
###### Python > 3.8
###### Pytorch
###### Torchvision

## Dataset

Datasets used:

1. [D435i dataset (real data)](https://drive.google.com/file/d/10RpBacPfDK3jwqf0yYSm1ovHMN05OHWm/view?usp=sharing)
2. [Active TartanAir dataset (virtual data)](https://drive.google.com/file/d/1hyYzBhzsl9uK8bfmIufC0EVpfjZ7GC_H/view?usp=sharing)

Please, use the links provided to download the datasets and update the ```data_root``` field in the ```Options/*.json``` files.

## Usage
To train on D435i sequences:

```
./d435i.sh
```

To train on Active Tartanair sequences:

```
./tartanair.sh
```
