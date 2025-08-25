# -*- coding: utf-8 -*-
from scipy.optimize import minimize,basinhopping


import numpy as np
from numpy.linalg import norm,inv,pinv,slogdet,matrix_rank
from scipy.linalg import cholesky
from numpy import array,trace,diag,zeros,ones,full,sqrt,eye,log
from multiprocess import Pool
from copy import deepcopy,copy

from robusttracking.estimator.utils import neglog_gaussian,fisher_information,weighted_l2
from robusttracking.estimator.sigma_points import get_sigmoid_points

def cost(*params):
    x,sensors,x_pred,inv_cov_pred,xs,obs = params
    #observation
    n_targets = obs.shape[0]
    dim = int(len(x)/n_targets)
    x = array(x).reshape((dim,n_targets))
    mu = sensors.hx(x,xs)
    cov = sensors.sx(x,xs)
    dmu = sensors.H(x,xs)
    dcov = sensors.S(x,xs)
    
    mux = x.T-x_pred.T
    muo = obs-mu
    n_targets = mu.shape[0]
    ll1 = 0
    ll2 = 0
    for i in range(n_targets):
        if cov[i].max()==np.inf:
            icov = zeros(cov[i].shape)
        else:
            icov = inv(cov[i])  
            
        icov_muo = icov@muo[i,:]
        muo_icov = muo[i,:].T@icov
            
        ll1 += muo_icov@muo[i,:] + slogdet(cov[i])[1]
    
        igamma_mux = inv_cov_pred[i]@mux[i,:]
        ll2 += mux[i,:].T@igamma_mux

                
    return ll1,ll2

def f_and_df(*params):#validated with autodiff jacobian
    x,sensors,x_pred,inv_cov_pred,xs,obs = params
    #observation
    n_targets = obs.shape[0]
    dim = int(len(x)/n_targets)
    x = array(x).reshape((dim,n_targets))
    mu = sensors.hx(x,xs)
    cov = sensors.sx(x,xs)
    dmu = sensors.H(x,xs)
    dcov = sensors.S(x,xs)
    
    mux = x.T-x_pred.T
    muo = obs-mu
    n_targets = mu.shape[0]
    ll = 0
    d_ll = zeros((dim,n_targets))
    d_info = dmu.shape[1]
    
    for i in range(n_targets):
        if cov[i].max()==np.inf:
            icov = zeros(cov[i].shape)
        else:
            icov = inv(cov[i])  
            
        icov_muo = icov@muo[i,:]
        muo_icov = muo[i,:].T@icov
            
        ll += muo_icov@muo[i,:] + slogdet(cov[i])[1]
    
        igamma_mux = inv_cov_pred[i]@mux[i,:]
        ll += mux[i,:].T@igamma_mux
        
        
        for k in range(dim):
            if k<d_info:
                d_ll[k,i] = trace(icov@dcov[i,k,:])-2*dmu[i,k,:]@icov_muo-muo_icov@dcov[i,k]@icov_muo+2*igamma_mux[k]
            else:
                d_ll[k,i] = 2*igamma_mux[k]
                
    return ll,d_ll.flatten()

def df2(*params):
    x,sensors,x_pred,inv_cov_pred,xs,obs = params
    #observation
    n_targets = obs.shape[0]
    dim = int(len(x)/n_targets)
    x = array(x).reshape((dim,n_targets))
    mu = sensors.hx(x,xs)
    cov = sensors.sx(x,xs)
    dmu = sensors.H(x,xs)
    dcov = sensors.S(x,xs)
    d_info = dmu.shape[1]
    Fim = fisher_information(sensors.hx(x,xs),sensors.sx(x,xs),sensors.H(x,xs),sensors.S(x,xs))
    H = zeros((n_targets*dim,n_targets*dim))
    for i in range(n_targets):
        H[i*dim:i*dim+d_info,i*dim:i*dim+d_info] = Fim[i]
        H[i*dim:i*dim+dim,i*dim:i*dim+dim] = inv_cov_pred[i]
    return -2*H
    
def error_fuse(f,inv_cov_pred):
    n_targets,fd,_ = f.shape
    C_sqt = zeros(inv_cov_pred.shape)
    C_ = zeros(inv_cov_pred.shape)
    d = C_sqt.shape[-1]//2
    crb_p = 0
    crb_v = 0
    for i in range(n_targets):
        try:
            fi = inv_cov_pred[i]
                
            fi[:fd,:fd] += f[i]
            C_[i] = inv(fi)
            try:
                C_sqt[i] = cholesky(C_[i]).T
            except:
                C_sqt[i] = np.diag(np.sqrt(np.diag(C_[i])))
            crb_p+=np.sqrt(trace(C_[i,:d,:d]))
            crb_v+=np.sqrt(trace(C_[i,d:,d:]))
                
        except:
            C_sqt[i,:,:] = 1e6*eye(2*d)
            C_[i,:,:] = 1e12*eye(2*d)
    return C_sqt,crb_p,crb_v,C_

def find_symmetric(obj,a,b):
    dim = obj.shape[0]//2
    m = (b[:dim]+a[:dim])/2
    obj_sym = np.zeros(obj.shape)
    obj_sym[:dim] = 2*m-obj[:dim]
    return obj_sym

def multiple_start_points(x_pred,xs):
    M = xs.shape[1]
    N = x_pred.shape[1]
    n_inits = M*(M-1)/2
    x_inits = []
    x_inits.append(x_pred)
    for i in range(M):
        for j in range(i+1,M):
            if np.random.random()<0.2:
                x_ = zeros(x_pred.shape)
                for k in range(N):
                    x_[:,k] = find_symmetric(x_pred[:,k],xs[:,i],xs[:,j])
                x_inits.append(x_)
    return x_inits


def task(pargs):
    x_init,args = pargs
    (sensors,x_pred,inv_cov_pred,xs,obs) = args
    #return minimize(f_and_df,x0=x_init.flatten(),args=(deepcopy(sensors),copy(x_pred),copy(inv_cov_pred),copy(xs),copy(obs)),jac=True,method='L-BFGS-B',options={'gtol':1e-8,'maxiter':1000,'ftol':1e-8})
    
    minimizer_kwargs = {"method": "L-BFGS-B","args":(deepcopy(sensors),copy(x_pred),copy(inv_cov_pred),copy(xs),copy(obs)),"jac":True,"options":{'gtol':1e-8,'ftol':1e-8}}
    return basinhopping(f_and_df, x_init.flatten(), minimizer_kwargs=minimizer_kwargs,niter=200)

def mle_fuse(sensors,x_pred,cov_pred,xs,obs,multistarts=False):
    x_shape = x_pred.shape
    _,n_targets = x_pred.shape
    inv_cov_pred = inv(cov_pred)
    
    if multistarts:
        x_inits = multiple_start_points(x_pred,xs)
        
        #x_inits = get_sigmoid_points(x_pred,cov_pred)
        print('multistarts',len(x_inits),x_inits)
        min_loss = 1e24
        best_res = None
        with Pool() as pool:
            for res in pool.imap(task,[(x_inits[i],(sensors,x_pred,inv_cov_pred,xs,obs)) for i in range(len(x_inits))]):
                if res.fun<min_loss:
                    min_loss = res.fun
                    best_res = res
        print(best_res)
        est = best_res.x.reshape(x_shape)
    else:
        res = task([x_pred,(sensors,x_pred,inv_cov_pred,xs,obs)])
        print(res)
        est = res.x.reshape(x_shape)
    
    #variance bound of estimator
    f = fisher_information(sensors.hx(est,xs),sensors.sx(est,xs),sensors.H(est,xs),sensors.S(est,xs))
     
    C_sqt,_,_,C_ = error_fuse(f,inv_cov_pred)
    
    return est,C_sqt,res.success,C_