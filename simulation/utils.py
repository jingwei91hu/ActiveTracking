# -*- coding: utf-8 -*-
from scipy.stats import multivariate_normal,chi2
from numpy import zeros,sqrt

def observe(mu,cov): 
    n_targets = mu.shape[0]
    obs = zeros(mu.shape)
    for i in range(n_targets):
        obs[i] = multivariate_normal.rvs(mean=mu[i], cov=cov[i])
    return obs


def q_alpha_d(dim,alpha=0.05):
    return sqrt(chi2.ppf(1-alpha,dim))

__all__=['observe']