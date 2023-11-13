# -*- coding: utf-8 -*-
import numpy as np

import torch
from torch.linalg import inv
from torch import trace,zeros,eye,sigmoid,dot
from torch.optim import Adam


from .cost import crlb_torch
from ..sensors import is_ranging

def reparam(u,lower,upper): #reparameterize
    return sigmoid(u).mul(upper-lower)+lower

def forward(sensor_torch,target_torch,x_s,x_t,u_s,u_t,v,rs,rt,rv,C_sqt,w,W,regularize = False,Q = None, c_hat = None):
    """ forward propagation of cost
    sensor_torch: simulation of sensors 
    target_torch: simulation of target
    x_s: d x M, state of sensors
    x_t: d x N, state of targets
    u_s: K x d x M, sensor control
    u_t: K x d x N, target control
    v: d x N, target uncertainty
    rs: bound of u_s 
    rt: bound of u_t 
    rv: bound of v 
    C_sqt: ellipsoid transition of v
    w: weight of cost at each time step, w_k*g(x_s[k],x_t[k]) 
    gamma: weight of velocity cost
    """
    
    #time horizon K
    K = u_t.shape[0]
    d = u_t.shape[1]//2
    n_target = u_t.shape[-1]
    
    #reparameterize x_0
    if v.requires_grad:
        v_ = zeros(v.shape)
        for it in range(n_target):
            v_[:,it] = reparam(C_sqt[it,:,:]@v[:,it],*rv)
            
        v = v_
    
    #reparameterize
    if u_s.requires_grad:
        u_s = reparam(u_s,*rs)
        
    if u_t.requires_grad:
        u_t = reparam(u_t,*rt)
    
    #initialize variables
    L = zeros(K)
    x_s_ = zeros((K,*x_s.shape))
    x_t_ = zeros((K,*x_t.shape))
    
    
    
   
    for k in range(K):
        if k == 0:
            x_s_[k] = sensor_torch.transition(x_s,u_s[k])
            x_t_[k] = target_torch.transition(x_t+v,u_t[k])
        else:
            x_s_[k] = sensor_torch.transition(x_s_[k-1],u_s[k])
            x_t_[k] = target_torch.transition(x_t_[k-1],u_t[k])
        
        #step cost
        mu = sensor_torch.hx(x_t_[k],x_s_[k])
        cov = sensor_torch.sx(x_t_[k],x_s_[k])
        d_mu = sensor_torch.H(x_t_[k],x_s_[k])
        d_cov = sensor_torch.S(x_t_[k],x_s_[k])
        
        if regularize:
            L[k] = crlb_torch(mu,cov,d_mu,d_cov,W,c_hat[k])
        else:
            L[k] = crlb_torch(mu,cov,d_mu,d_cov,W)
    
    return dot(L,w)

# def ekf(P,FPF_Q,F,d_mu,cov):
#     n_targets,dim,n_sensors = d_mu.shape
#     s_dim = F.shape[0]
    
#     P_ = zeros(P.shape)
#     J = 0
    
#     for it in range(n_targets): #for each target
#         H = zeros((n_sensors,s_dim))
#         H[:,:dim] = d_mu[it].T
#         R = cov[it]
        
#         FPH_T = F@P[it]@H.T 
#         inv_S = inv(H@P[it]@H.T + R)
#         P_[it] = FPF_Q[it] - FPH_T@inv_S@FPH_T.T
#         J += trace(P_[it])
#     return J,P_

#def forward_ekf(sensor_torch,target_torch,x_s,x_t,u_s,u_t,v,rs,rt,rv,C_sqt,w):
#    return None
#     u_s_shape = u_s.shape
#     u_t_shape = u_t.shape
    
#     #time horizon K
#     K = u_t_shape[0]
    
#     #deal with x_0
#     if v.requires_grad:
#         v_ = torch.zeros(v.shape)
#         for n_target in range(v.shape[1]):
#             v_[:,n_target] = reparam(C_sqt[n_target,:,:]@v[:,n_target],*rv)
            
#         v = v_
    
#     #reparameterize
#     if u_s.requires_grad:
#         u_s = reparam(u_s,*rs)
    
#     if u_t.requires_grad:
#         u_t = reparam(u_t,*rt)
    
#     #initialize variables
#     L = torch.zeros(K)
#     x_s_ = torch.zeros(u_s_shape)
#     x_t_ = torch.zeros(u_t_shape)
    
#     #P0
#     n_targets,dim,dim = C_sqt.shape
#     P = torch.zeros(C_sqt.shape)
#     for it in range(n_targets):
#         P[it] = torch.linalg.inv(C_sqt[it,:,:].T@C_sqt[it,:,:])
    
#     for k in range(K):
#         if k == 0:
#             x_s_[k] = sensor_torch.transition(x_s,u_s[k])
#             x_t_[k] = target_torch.transition(x_t+v,u_t[k])
#         else:
#             x_s_[k] = sensor_torch.transition(x_s_[k-1],u_s[k])
#             x_t_[k] = target_torch.transition(x_t_[k-1],u_t[k])
            
#         d_mu = sensor_torch.H(x_t_[k],x_s_[k])
#         d_cov = sensor_torch.S(x_t_[k],x_s_[k])
#         cov = sensor_torch.sx(x_t_[k],x_s_[k])
#         F = sensor_torch.getF()
        
#         FPF_Q = torch.zeros(P.shape)
     
#         for it in range(n_targets):
#             if isinstance(sensor_torch,DopplerSensor3DTorchUtils) or isinstance(sensor_helper,DopplerSensor3DTorchUtils_2):
#             ##TODO!!!!!
#                 FPF_Q[it] =F@P[it]@F.T+B@torch.diag((h-l)**2)@B.T/12
#             else:
#                 FPF_Q[it] =sensor_helper.transition_matrix(P[it],torch.ones(dim)*range_u_s[1],torch.ones(dim)*rs[0])
            
#         l,P = ekf(P,FPF_Q,F,d_mu,cov)
#         L[k] = l
        
#     return torch.dot(L,w)


def step(sensor_torch,target_torch,opts,x_s,x_t,u_s,u_t,v,rs,rt,rv,C_sqt,w,W, regularize = False,Q = None, c_hat = None):
    """ One step of gradient method
    x_s: d x M, state of sensors
    x_t: d x N, state of targets
    u_s: K x d x M, sensor control
    u_t: K x d x N, target control
    v: d x N, target uncertainty
    rs: bound of u_s 
    rt: bound of u_t 
    rv: bound of v 
    C_sqt: ellipsoid transition of v
    w: weight of cost at each time step, w_k*g(x_s[k],x_t[k]) 
    gamma: weight of velocity cost
    """
        
    for opt in opts:
        opt.zero_grad()
   
    loss = forward(sensor_torch,target_torch,x_s,x_t,u_s,u_t,v,rs,rt,rv,C_sqt,w,W,regularize=regularize,Q = Q, c_hat=c_hat)
        
    loss.backward()
    
    for opt in opts:
        opt.step()
    
    with torch.no_grad():
        loss_val = loss.item()
            
    return loss_val

def construct_c_hat(K,C_sqt,rt,target_torch):
    n_target = C_sqt.shape[0]
    d = C_sqt.shape[1]//2
    c_hat = zeros((K,*C_sqt.shape))
    
    
    Q = zeros((2*d,2*d))
    for it in range(n_target):
        #variance of a uniform random variable
        Q[d:,d:] = eye(d)*((rt[1]-rt[0])**2)/12
        
        for k in range(K):
            if k==0:
                c = torch.as_tensor(inv(C_sqt[it,:,:]@C_sqt[it,:,:].T))
            else:
                c = torch.as_tensor(c_hat[k-1,it])
                
            c_hat[k,it] = torch.as_tensor(target_torch.predictC(c,Q))
    return c_hat

def construct_W(dim,gamma=1):
    W = eye(dim)
    if gamma!=1:
        W[dim//2:,dim//2:] = gamma
    return W

def move(K,sensor_torch,target_torch,x_s,x_t,lr,rs,rt,C_sqt,w,n_dynamic_sensors,is_target=True,n_iter = 1000, gamma = 1, regularize = False,Q = None, c_hat = None):
    """
     if is_target = True, rt works, otherwise rs works. 
     n_static only works when is_target=False
    """
    s_shape = x_s.shape
    t_shape = x_t.shape
    d = s_shape[0]//2

    if is_target:
        u_s = torch.zeros((K,*s_shape))
        init_u_t = 0.01*np.random.randn(K,*t_shape)
        init_u_t[:,:d,:] = 0
        u_t = torch.tensor(init_u_t, requires_grad = True)
    else:
        init_u_s = 0.01*np.random.randn(K,s_shape[0],n_dynamic_sensors)
        init_u_s[:,:d,:] = 0
        u_s = torch.tensor(init_u_s, requires_grad = True)
        u_t = torch.zeros((K,*t_shape))
        
    v = torch.zeros(t_shape)
    rv = torch.zeros(2)
    
    losses = []
    
    opts = [Adam([u_t if is_target else u_s],lr = lr)]
    
    if is_target:
        opts[0].param_groups[0]['lr'] *= -1 #change descending to ascending
    
    if regularize:
        c_hat = construct_c_hat(K,C_sqt,rt,target_torch)
        W = construct_W(2*d,gamma)
    elif is_ranging(sensor_torch):
        W = construct_W(d,gamma)
    else:
        W = construct_W(2*d,gamma)
    
    for i in range (n_iter):
        loss_val = step(sensor_torch,target_torch,opts,x_s,x_t,u_s,u_t,v,rs,rt,rv,C_sqt,w, W, regularize = regularize,c_hat = c_hat)
        losses.append(loss_val)
        if loss_val==np.inf:
            break
            
    if is_target:
        u = reparam(u_t.detach(),*rt)
    else:
        u = reparam(u_s.detach(),*rs)
        
    return u,losses

def robustmove(K,sensor_torch,target_torch,x_s,x_t,lr_s,lr_t,rs,rt,rv,C_sqt,w,n_dynamic_sensors,n_iter = 1000,gamma = 1, regularize = False, c_hat = None):
    s_shape = x_s.shape
    t_shape = x_t.shape
    d = s_shape[0]//2
    
    init_u_s = 0.01*np.random.randn(K,s_shape[0],n_dynamic_sensors)
    init_u_s[:,:d,:] = 0
    u_s = torch.tensor(init_u_s, requires_grad=True)
    
    init_u_t = 0.01*np.random.randn(K,*t_shape)
    init_u_t[:,:d,:] = 0
    u_t = torch.tensor(init_u_t, requires_grad=True)
    
    #uncertainty variable
    v = torch.zeros(t_shape,requires_grad=True)
    
    losses = []
    
    opt_t = Adam([u_t,v],lr = lr_t) 
    opt_t.param_groups[0]['lr'] *= -1
    
    opt_s = Adam([u_s],lr = lr_s)
    
    opts = [opt_s,opt_t]
    
    if regularize:
        c_hat = construct_c_hat(K,C_sqt,rt,target_torch)
        W = construct_W(2*d,gamma)
    elif is_ranging(sensor_torch):
        W = construct_W(d,gamma)
    else:
        W = construct_W(2*d,gamma)
        
    for i in range (n_iter):
        loss_val = step(sensor_torch,target_torch,opts,x_s,x_t,u_s,u_t,v,rs,rt,rv,C_sqt,w,W,regularize=regularize,c_hat=c_hat)
        losses.append(loss_val)
        
    u_s_val = reparam(u_s.detach(),*rs)
    
    return u_s_val,losses
