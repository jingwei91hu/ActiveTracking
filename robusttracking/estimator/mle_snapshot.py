# -*- coding: utf-8 -*-
from scipy.optimize import minimize

import numpy as np
from numpy.linalg import norm,inv,pinv,slogdet,matrix_rank,cholesky
from numpy import array,trace,diag,zeros,ones,full,sqrt,eye

from ..sensors.utils import is_ranging
from .utils import neglog_gaussian,fisher_information

def fun_likelihood(*params):
    x,sensors,xs,obs = params
    n_targets = obs.shape[0]
    dim = int(len(x)/n_targets)
    x = array(x).reshape((dim,n_targets))
    mu = sensors.hx(x,xs)
    cov = sensors.sx(x,xs)
    ll = neglog_gaussian(obs,mu,cov)
    return ll
    
def error_snapshot(f,isranging):
    n_targets,fd,_ = f.shape
    C_sqt = zeros((n_targets,fd,fd))
    crb_p = 0
    crb_v = 0
    for i in range(n_targets):
        try:
            if isranging: 
                C_sqt[i] = cholesky(inv(f[i])).T
                
                crb_p+=trace(C_sqt[i])
            else:
                C_sqt[i] = cholesky(inv(f[i])).T
                crb_p+=trace(C_sqt[i,:fd//2,:fd//2])
                crb_v+=trace(C_sqt[i,fd//2:,fd//2:])
        except:
            C_sqt[i,:,:] = 1e5*eye(fd)
    return C_sqt,crb_p,crb_v

def mle_snapshot(sensors,xs,obs,initial_guess):
    x_shape = initial_guess.shape
    res = minimize(fun_likelihood,x0=initial_guess.flatten(),args=(sensors,xs,obs),method='L-BFGS-B',options={'gtol':1e-32,'ftol':1e-32,'maxiter':1000})
    #print(res)
    est = res.x.reshape(x_shape)
    
    #variance bound of estimator
    
    f = fisher_information(sensors.hx(est,xs),sensors.sx(est,xs),sensors.H(est,xs),sensors.S(est,xs))
    C_sqt,_,_ = error_snapshot(f,is_ranging(sensors))
    return est,C_sqt,res.success
