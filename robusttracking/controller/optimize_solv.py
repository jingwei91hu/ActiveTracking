# -*- coding: utf-8 -*-
import numpy as np
import scipy
from cyipopt import minimize_ipopt

from numpy.linalg import inv,matrix_power
from numpy import trace,zeros,eye,dot,ones

import jax.numpy as jnp
from jax import jit, grad, jacfwd, jacrev
import jax
from scipy.optimize import LinearConstraint,minimize,basinhopping,approx_fprime
import time
from .cost_np import crlb_np
from .cost_jnp import crlb_jnp

from .sigmapoints import get_sigmoid_points
from ..sensors import is_ranging


jax.config.update('jax_enable_x64', True)

def create_sigmapoints(K,target_np,x_t,Q,C_prev,alpha=0.95):
    dimx,n_target = x_t.shape
    dim = dimx//2
    
    # eta_0 = [x_t_0,a_1,a_2,...a_{k-1}]
    eta_0 = np.pad(x_t,[(0, dim*K), (0, 0)])
  
    x_t_set = []
    c_inv_set = []
    for it in range(n_target):
        #1. generate sigma points
        
        C0 = scipy.linalg.block_diag(*([C_prev[it]]+K*[Q[dim:,dim:]]))
        
        eta_sets,wmx0,wcx0,wcxi = get_sigmoid_points(eta_0[:,it],C0,dimx, alpha)
     
        #2. predict
        _,N = eta_sets.shape
        x_pred = zeros((dimx,N))
        weights_m = zeros((N,1))
        weights_c = zeros((N,1))
        for i in range(N):
            x_t_ = eta_sets[:dimx,[i]]
            for k in range(K):
                a_k = zeros(x_t_.shape) 
                a_k[dim:] = eta_sets[dimx+dim*k:dimx+dim*(k+1),[i]]
                x_t_ = target_np.transition(x_t_,a_k)
        
                
            x_pred[:,[i]] = x_t_.copy()
            
            if i==0:
                weights_m[i] = wmx0
                weights_c[i] = wcx0
            else:
                weights_m[i] = wcxi
                weights_c[i] = wcxi
            
            x_t_set.append(x_t_[:,0].copy())
            
        x_pred0 = x_pred@weights_m
        C = (x_pred-x_pred0)@np.diag(weights_c[:,0])@(x_pred-x_pred0).T
        
        
        c_inv_set.append(np.tile(np.expand_dims(inv(C),0),(N,1,1)))
    return np.vstack(x_t_set).T,np.vstack(c_inv_set)

def construct_c_inv_hat(K,target_np,Q,C_prev):
    n_target = C_prev.shape[0]
    C_hat = zeros(C_prev.shape)
    
    for it in range(n_target):
        C_ = C_prev[it].copy()
        for k in range(K):
            C_ = target_np.predictC(C_,Q)
        C_hat[it] = C_.copy()
    return inv(C_hat)

def state_constraints(sensor_np,x_s,K,S_min,S_max,n_dynamic_sensors):
    dim,_ = x_s.shape
    F,B = sensor_np.getFB()
    As = []
    ls = []
    us = []
    for k in range(K):
        lb = (S_min-matrix_power(F,k+1)@x_s).flatten('F')
        ub = (S_max-matrix_power(F,k+1)@x_s).flatten('F')
        
        A = np.zeros((dim*n_dynamic_sensors,K*dim*n_dynamic_sensors))
        for j in range(k+1):
            A[:,j*dim*n_dynamic_sensors:(j+1)*dim*n_dynamic_sensors] = scipy.linalg.block_diag(*n_dynamic_sensors*[(matrix_power(F,k-j)@B)])
        As.append(A)
        ls.append(lb)
        us.append(ub)
     
    A_ = np.vstack(As)
    l_ = np.hstack(ls)
    u_ = np.hstack(us)
    
    
    A_e = np.vstack([A_,-A_])
    b_e = np.hstack([u_,-l_])
    return A_e,b_e

def flatten(u_s):
    K = u_s.shape[0]
    return u_s.reshape((K,-1,),order='F').reshape((-1,1),order='C')


def de_flatten(u_s,shape):
    K,dim,M = shape
    return u_s.reshape((K,-1),order='C').reshape((K,dim,M),order='F')

from functools import partial

def no_fly_cost(x_s,nfc,S,r):
    return (jnp.maximum(1-jnp.linalg.norm(S@(x_s-nfc),axis=1)/r,0)).sum()

# def objective(u_s,x_s,x_t,c_inv_hat,sensor_np,K,dim_x_s,Fk,Bk,rho,nfc,S,r):
    
#     _,M = x_s.shape
    
#     u_s_ = jnp.concatenate(de_flatten(u_s,(K,dim_x_s,M)),axis=0)
#     x_s_k = Fk@x_s + Bk@u_s_
   
#     mu = sensor_np.hx(x_t,x_s_k[-dim_x_s:,:])
#     cov = sensor_np.sx(x_t,x_s_k[-dim_x_s:,:])
#     d_mu = sensor_np.H(x_t,x_s_k[-dim_x_s:,:])
#     d_cov = sensor_np.S(x_t,x_s_k[-dim_x_s:,:])
#     return crlb_jnp(mu,cov,d_mu,d_cov,None,c_inv_hat)+rho*no_fly_cost(x_s_k.reshape(K,dim_x_s,M),nfc,S,r)/K/M

def objective(u_s,x_s,x_t,c_inv_hat,sensor_np,K,Fk,Bk):
    
    dim_x_s,M = x_s.shape
    
    u_s_ = jnp.concatenate(de_flatten(u_s,(K,dim_x_s,M)),axis=0)
    x_s_k = Fk@x_s + Bk@u_s_
   
    mu = sensor_np.hx(x_t,x_s_k)
    cov = sensor_np.sx(x_t,x_s_k)
    d_mu = sensor_np.H(x_t,x_s_k)
    d_cov = sensor_np.S(x_t,x_s_k)
    return crlb_jnp(mu,cov,d_mu,d_cov,None,c_inv_hat)

        
def move_scipy(K,sensor_np,target_np,x_s,x_t,rs,n_dynamic_sensors,C_prev,S_min,S_max,rho = 0,nfc = None,S = None,r = None,take_sigmapoints=True, n_iter = 1000, Q = None, init_u_s = None,W = None):
    s_shape = x_s.shape
    t_shape = x_t.shape
    d = s_shape[0]//2

   
    if take_sigmapoints:
        x_t_,c_inv_hat = create_sigmapoints(K,target_np,x_t,Q,C_prev)
    else:
        x_t_ = x_t.copy()
        c_inv_hat =  construct_c_inv_hat(K,target_np,Q,C_prev)
   
    

    
    
    A_,b_ = state_constraints(sensor_np,x_s,K,S_min,S_max,n_dynamic_sensors)
    
    cons = [LinearConstraint(A_,ub=b_)]
        
    Fk,Bk = sensor_np.getFBk(K)
    
#     dim_x_s = d*2
#     Fk = np.zeros((K*dim_x_s,K*dim_x_s))
#     Bk = np.zeros((K*dim_x_s,K*dim_x_s))
#     for i in range(K):
#         Fi,Bi = sensor_np.getFBk(i+1)
#         Fk[i*dim_x_s:i*dim_x_s+dim_x_s,i*dim_x_s:i*dim_x_s+dim_x_s] = Fi
#         Bk[i*dim_x_s:i*dim_x_s+dim_x_s,:i*dim_x_s+dim_x_s] = Bi
    
#     x_s_ = np.tile(x_s,(K,1))
    print(x_s.shape,Fk.shape,Bk.shape)


    print('obj',objective(flatten(init_u_s),x_s,x_t_,c_inv_hat,sensor_np,K,Fk,Bk))
    
    t0 = time.time()
    obj_jit = jit(partial(objective,x_s=jnp.array(x_s),x_t=jnp.array(x_t_),c_inv_hat=jnp.array(c_inv_hat),sensor_np=sensor_np,K=K,Fk=jnp.array(Fk),Bk=jnp.array(Bk)))
    obj_grad = jit(grad(obj_jit))  # objective gradient
    t1 = time.time()
    print('jit time',t1-t0)
    
    minimizer_kwargs = {'bounds':(tuple(rs),),'constraints':cons,'jac':obj_grad,'options':{'maxiter':30}}
    res = minimize_ipopt(obj_jit,flatten(init_u_s),**minimizer_kwargs)
    t2 = time.time()
    print('ipopt time',t2-t1)
    
    u_s = de_flatten(res.x,(K,s_shape[0],n_dynamic_sensors))
    
    return u_s

