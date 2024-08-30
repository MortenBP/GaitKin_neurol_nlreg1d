# GaitKin_neurol_nlreg1d

This repository contains the code for generating a graphical representation of data for the study on gait pathology, kinematics and self-reported comfort in body weight supported gait of young adults with neurological disorders.

<br>
This repository contains code and example data associated with the paper:

Pedersen MB, Simonsen MB, Aagaard P, Rasmussen G, Stengaard A, Holsgaard-Larsen A (in review). Quality, kinematics, and self-reported comfort of gait during body weight support in young adults with gait impairments – a cross-sectional study. 

<img src="https://github.com/user-attachments/assets/a138a053-76d7-454c-a2f1-e7a6dd639441" width="600" height="600" />

<img src="https://github.com/user-attachments/assets/420cbaef-986d-4bd6-bbb0-262ec73a0574" width="600" height="600" />

<img src="https://github.com/user-attachments/assets/e60a8c56-3ac8-4c87-b97a-30b57375ad2e" width="600" height="600" />

<br>

This repository is for use in Matlab. However, since the [nlreg1d](https://github.com/0todd0000/nlreg1d) repository is currently only available in python, the non-linear registration is performed by running a python environment and executing python code through Matlab.
The nlreg1d repository contains primarily wrapper functions to key functionality in [fdasrsf](https://github.com/jdtuck/fdasrsf_python) and [scikit-fda](https://fda.readthedocs.io/en/latest/) and [spm1d](https://spm1d.org).

<br>

## Dependencies
Matlab dependencies include:
- [spm1d](https://github.com/0todd0000/spm1dmatlab) (to install, download from repository and add to Matlab path)
- daviolinplot
- fdasrvf
- Curve fitting

Python Environment dependencies include:

- [nlreg1d](https://github.com/0todd0000/nlreg1d) (to install, follow instructions on repository site)
- python 3.9
- numpy 1.22
- scipy 1.8
- spm1d 0.4
- fdasrsf 2.3
- scikit-fda 0.7

<br>

## Installation

To install all dependencies for this repository, we suggest creating an Anaconda environment using Python 3.9. This repository has not been tested with other python versions.

<br>

To link your python environment to Matlab, follow the instructions in the main_analysis.m script in Matlab.

<br>

## Get started

Open main_analysis.m in Matlab and follow the instructions in the script.
