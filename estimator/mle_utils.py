# -*- coding: utf-8 -*-
from scipy.stats import multivariate_normal
from scipy.optimize import minimize

import numpy as np
from numpy.linalg import norm,inv,pinv,slogdet,matrix_rank,cholesky
from numpy import array,trace,diag,zeros,ones,full,einsum,sqrt,eye



def neglog_gaussian(obs,mu,cov):
    muo = obs-mu
    n_targets = mu.shape[0]
    ll = 0
    for i in range(n_targets):
        inv_cov = inv(cov[i])
        ll += muo[i,:].T@inv_cov@muo[i,:] + slogdet(cov[i])[1]
    return ll

def fun_mle_gaussian_1(*params):
    x,sensors,xs,obs = params
    n_targets = obs.shape[0]
    dim = int(len(x)/n_targets)
    x = array(x).reshape((dim,n_targets))
    mu = sensors.hx(x,xs)
    cov = sensors.sx(x,xs)
    ll = neglog_gaussian(obs,mu,cov)
    return ll

def fisher_information(mu,cov,d_mu,d_cov):
    n_targets,dim,n_sensors = d_mu.shape
    I = zeros((n_targets,dim,dim))
    for k in range(n_targets):
        inv_cov = inv(cov[k])
        for i in range(dim):
            for j in range(dim):
                I[k,i,j] = d_mu[k,i,:]@inv_cov@d_mu[k,j,:].T + 0.5*trace(inv_cov@d_cov[k,i,:,:]@inv_cov@d_cov[k,j,:,:])
    return I

def crlb(*args):
    f = fisher_information(*args)
    n_targets,dim,dim = f.shape
    J = 0
    for i in range(n_targets):
        fr = matrix_rank(f[i])
        if fr == dim: 
            lb = inv(f[i])
            J += trace(lb)
        else:
            J += np.inf
        
    return J

def crlb_rmse(*args):
    crb_v,mu,cov,d_mu,d_cov = args
    f = fisher_information(mu,cov,d_mu,d_cov)
    n_targets,dim,dim = f.shape
    rmse_p = zeros(n_targets)
    rmse_v = zeros(n_targets)
    for i in range(n_targets):
        fr = matrix_rank(f[i])
        if fr == dim:
            lb = inv(f[i])
            if crb_v:
                rmse_p[i] = sqrt(trace(lb[:dim//2,:dim//2]))
                rmse_v[i] = sqrt(trace(lb[dim//2:,dim//2:]))
            else:
                rmse_p[i] = sqrt(trace(lb))
        else:
            rmse_p[i] = np.inf
            rmse_v[i] = np.inf
    return rmse_p,rmse_v
    
def error_cov(*args):
    f = fisher_information(*args)
    n_targets,dim,dim = f.shape
    C_sqt = zeros((n_targets,dim,dim))
    for i in range(n_targets):
        fr = matrix_rank(f[i])
        if fr == dim: 
            try:
                C_sqt[i,:,:] = cholesky(inv(f[i])).T
            except:
                C_sqt[i,:,:] = 1000*eye(dim)  
        else:
            C_sqt[i,:,:] = 1000*eye(dim)   
    return C_sqt


def estimate_mle(sensors,xs,obs,initial_guess):
    x_shape = initial_guess.shape
    res = minimize(fun_mle_gaussian_1,x0=initial_guess.flatten(),args=(sensors,xs,obs),method='L-BFGS-B',options={'gtol':1e-32,'ftol':1e-32,'maxiter':1000})
    #print(res)
    est = res.x.reshape(x_shape)
    
    #variance bound of estimator
    C_sqt = error_cov(sensors.hx(est,xs),sensors.sx(est,xs),sensors.H(est,xs),sensors.S(est,xs))
    
    dim,n_target = x_shape
    
    if C_sqt.shape[1]<dim:
        C_ = zeros((n_target,dim,dim))
        for i in range(n_target):
            C_[i,:,:] = eye(dim)
            C_[i,:C_sqt.shape[1],:C_sqt.shape[1]] = C_sqt[i,:,:]
    
        C_sqt = C_
  
    return est,C_sqt,res.success