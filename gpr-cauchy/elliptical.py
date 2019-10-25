#!/usr/local/bin/python
#
#  Elliptical slice sampling.
#  $Id: elliptical.py,v 1.1 2018/02/27 03:23:39 daichi Exp $
#  based on the code of Iain Murray
#  http://homepages.inf.ed.ac.uk/imurray2/pub/10ess/elliptical_slice.m
#
#  xx     : Dx1 initial vector
#  prior  : DxD matrix from chol(S)
#  likfun : function of likelihood evaluation
#  params : parameters passed to likfun (optional)
#  curlik : current likelihood (possibly from previous iteration)
#  angle  : default 0
#
#
import numpy as np
from pylab import *


def elliptical(xx, prior, likfun, params=(), curlik=None, angle=0):
    # initialize
    D = len(xx)
    if curlik is None:
        curlik = likfun(xx, params)
    # set up the ellipse
    nu = np.dot(prior, randn(D))
    hh = log(rand()) + curlik
    # set up the bracket
    if angle <= 0:
        phi = rand() * 2 * pi
        min_phi = phi - 2 * pi
        max_phi = phi
    else:
        min_phi = - angle * rand()
        max_phi = min_phi + angle
        phi = min_phi + rand() * (max_phi - min_phi)

    # slice sampling loop
    while True:
        prop = xx * cos(phi) + nu * sin(phi)
        curlik = likfun(prop, params)
        if curlik > hh:
            break
        if phi > 0:
            max_phi = phi
        elif phi < 0:
            min_phi = phi
        else:
            raise IOError(
                'BUG: slice sampling shrunk to the current position.')
        phi = min_phi + rand() * (max_phi - min_phi)

    return (prop, curlik)
