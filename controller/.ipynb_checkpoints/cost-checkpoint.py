# -*- coding: utf-8 -*-
import torch
from torch.linalg import inv
from torch import trace,zeros,eye

def fisher_information_torch(mu,cov,d_mu,d_cov):
    n_targets,dim,n_sensors = d_mu.shape
    I = zeros((n_targets,dim,dim))
    for k in range(n_targets):
        inv_cov = inv(cov[k])
        for i in range(dim):
            for j in range(dim):
                I[k,i,j] = d_mu[k,[i],:]@inv_cov@d_mu[k,[j],:].T + 0.5*trace(inv_cov@d_cov[k,i,:,:]@inv_cov@d_cov[k,j,:,:])
    return I

def crlb_torch(mu,cov,d_mu,d_cov,kappa = 1):
    f = fisher_information_torch(mu,cov,d_mu,d_cov)
    n_targets,dim,dim = f.shape
    
    W = eye(dim)
    if kappa!=1:
        W[dim//2:,dim//2:] = eye(dim//2)*kappa
    
    J = 0
    for i in range(n_targets):
        lb = inv(f[i])
        J += trace(W@lb)
        
    return J


__all__=['crlb_torch']