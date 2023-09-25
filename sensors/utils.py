# -*- coding: utf-8 -*-
from scipy.stats import multivariate_normal,chi2
from numpy import zeros,sqrt
from .sensors_ranging import *

def observe(mu,cov): 
    n_targets = mu.shape[0]
    obs = zeros(mu.shape)
    for i in range(n_targets):
        obs[i] = multivariate_normal.rvs(mean=mu[i], cov=cov[i])
    return obs


def q_alpha_d(dim,alpha=0.05):
    return sqrt(chi2.ppf(1-alpha,dim))


def is_ranging(sensor):
    return isinstance(sensor,RangingSensor)|isinstance(sensor,RangingSensorT)
            
__all__=['observe','q_alpha_d','is_ranging']