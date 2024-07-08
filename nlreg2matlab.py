# nlreg2matlab.py
"""performs non-linear registration of kinematic trajectories and returns data to Matlab format.
Takes Matlab double with rows = participants, columns = observations.
Input data should be linearly registered"""
import numpy as np
import nlreg1d.nlreg1d as nl


def get_nlreg_data(y):
    # set parameters:
    np.random.seed(123456789)
    niter = 5  # max iterations for SRSF registration

    # load and register data:
    yr, wf = nl.register_srsf(y, MaxItr=niter)
    yr_mat = np.mat(yr)
    return yr_mat


def get_nlreg_displacement(y):
    # set parameters:
    np.random.seed(123456789)
    niter = 5  # max iterations for SRSF registration

    # load and register data:
    yr, wf = nl.register_srsf(y, MaxItr=niter)
    wlist = nl.Warp1DList(wf)
    d = wlist.get_displacement_field()
    d_mat = np.mat(d)
    return d_mat
