#!/bin/bash 

python3 tauclassifier.py train 

python3 tauclassifier.py rank -weights=network_weights/weights-13.h5

python3 tauclassifier.py plot -weights=network_weights/weights-13.h5