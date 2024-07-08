# GaitKin_neurol_nlreg1d

Graphical representation of data for the study on gait pathology, kinematics and self-reported comfort in body weight supported gait of young adults with neurological disorders.

<br>

This repository contains code and example data associated with the paper:

Pedersen MB, Simonsen MB, Holsgaard-Larsen A (in review). Pathology, kinematics, and self-reported comfort of gait during body weight support in young adults with neurological disorders. 

<br>

This repository is for use in Matlab. However, since the [nlreg1d](https://github.com/0todd0000/nlreg1d) repository is currently only available in python, the non-linear registration is performed by running a python environment and executing python code through Matlab.
The nlreg1d repository contains primarily wrapper functions to key functionality in [fdasrsf](https://github.com/jdtuck/fdasrsf_python) and [scikit-fda](https://fda.readthedocs.io/en/latest/) and [spm1d](https://spm1d.org).

<br>

## Dependencies
Matlab dependencies include:
- spm1d
- daviolinplot
- fdasrvf
- Curve fitting

Python Environment dependencies include:

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

Open main_analysis.m in Matlab and follow the instructions in the live script.