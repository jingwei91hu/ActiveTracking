# -*- coding: utf-8 -*-
from scipy.optimize import minimize

import numpy as np
from numpy.linalg import norm,inv,pinv,slogdet,matrix_rank,cholesky
from numpy import array,trace,diag,zeros,ones,full,einsum,sqrt,eye,log


def neglog_gaussian(obs,mu,cov):
    muo = obs-mu
    n_targets = mu.shape[0]
    ll = 0
    for i in range(n_targets):
        if cov[i].max()==np.inf:
            icov = zeros(cov[i].shape)
        else:
            icov = inv(cov[i])  
        ll += muo[i,:].T@icov@muo[i,:] + slogdet(cov[i])[1]
        
    return ll

def weighted_l2(obs,mu,cov):
    muo = obs-mu
    n_targets = mu.shape[0]
    wt = zeros(n_targets)
    for i in range(n_targets):
        if cov[i].max()==np.inf:
            icov = zeros(cov[i].shape)
        else:
            icov = inv(cov[i])
        wt[i] = muo[i,:].T@icov@muo[i,:] 
    return wt

def fisher_information(mu,cov,d_mu,d_cov):
    n_targets,dim,n_sensors = d_mu.shape
    I = zeros((n_targets,dim,dim))
    for k in range(n_targets):
        inv_cov = inv(cov[k])
        for i in range(dim):
            for j in range(dim):
                I[k,i,j] = d_mu[k,i,:]@inv_cov@d_mu[k,j,:].T + 0.5*trace(inv_cov@d_cov[k,i,:,:]@inv_cov@d_cov[k,j,:,:])
    return I
